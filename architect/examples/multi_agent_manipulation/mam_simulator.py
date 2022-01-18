"""Define a simulator for the multi-agent manipulation (MAM) task"""
from functools import partial
from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp

from architect.components.geometry.transforms_2d import rotation_matrix_2d


@jax.jit
def softnorm(x):
    """Compute the 2-norm, but if x is too small replace it with the squared 2-norm
    to make sure it's differentiable. This function is continuous and has a derivative
    that is defined everywhere, but its derivative is discontinuous.
    """
    eps = 1e-5
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)


@jax.jit
def calc_ground_wrench_on_box(
    box_state: jnp.ndarray,
    mu: jnp.ndarray,
    mass: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the wrench on the box as a result of contact with the ground

    args:
        box_state: (6,) array of (x, y, theta, vx, vy, omega) state of the box
        mu: ground-box coefficient of friction (1-element, 0-dimensional array)
        mass: box mass (1-element, 0-dimensional array)
    """
    # Get the normal force of the box resting on the ground
    g = 9.81  # m/s^2 -- gravitational acceleration
    f_normal = g * mass

    # Get the translational and angular velocity of the box
    v_WB = box_state[3:5]
    w_WB = box_state[5]

    # Set some constants to enable a continuous friction model
    psi_s = 0.3  # tangential velocity at which slipping occurs
    c = mu / psi_s  # coefficient of tangential velocity in sticking friction

    # Compute translational friction
    psi = softnorm(v_WB)  # Magnitude of tangential velocity
    norm_factor = (psi < 1e-5) * 1.0 + (psi >= 1e-5) * psi
    sliding_direction = v_WB / norm_factor
    sticking = psi <= psi_s
    slipping = jnp.logical_not(sticking)
    friction_coefficient = (
        sticking * c * psi * sliding_direction + slipping * mu * sliding_direction
    )
    translational_friction_force = -friction_coefficient * f_normal

    # Compute rotational friction (this is an approximation)
    psi = softnorm(w_WB)  # Magnitude of angular velocity
    sliding_direction = jnp.sign(w_WB)
    sticking = psi <= psi_s
    slipping = jnp.logical_not(sticking)
    friction_coefficient = (
        sticking * c * psi * sliding_direction + slipping * mu * sliding_direction
    )
    # Make rotational friction less than translational
    rotational_friction_torque = -0.1 * friction_coefficient * f_normal

    ground_wrench_on_box = jnp.zeros((3,))
    ground_wrench_on_box = ground_wrench_on_box.at[:2].set(translational_friction_force)
    ground_wrench_on_box = ground_wrench_on_box.at[2].set(rotational_friction_torque)

    return ground_wrench_on_box


@jax.jit
def box_turtle_signed_distance(
    box_pose: jnp.ndarray,
    turtle_pose: jnp.ndarray,
    box_size: jnp.ndarray,
    turtle_radius: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the signed distance from the box to the turtlebot

    args:
        box_pose: (3,) array with current (x, y, theta) pose of the box
        turtle_pose: (3,) array with current (x, y, theta) state of the turtlebot
        box_size: side length of box (1-element, 0-dimensional array)
        turtle_radius: chassis radius of turtlebot (1-element, 0-dimensional array)
    returns:
        signed distance
    """
    # Credit to this stackoverflow answer for the inspiration for this code:
    # stackoverflow.com/questions/30545052/
    # calculate-signed-distance-between-point-and-rectangle

    # First transform the turtlebot (x, y) into the box frame
    p_WT = turtle_pose[:2]
    p_WB = box_pose[:2]
    theta_B = box_pose[2]

    p_BT_W = p_WT - p_WB
    # Rotate p_BT_W by -theta about the z axis to get position in box frame
    R_WB = rotation_matrix_2d(theta_B)
    R_BW = R_WB.T  # type: ignore
    p_BT = R_BW @ p_BT_W

    # Now get the signed distance
    x_dist = jnp.maximum(-(p_BT[0] + box_size / 2.0), p_BT[0] - box_size / 2.0)
    y_dist = jnp.maximum(-(p_BT[1] + box_size / 2.0), p_BT[1] - box_size / 2.0)

    # phi = signed distance.
    phi = jnp.minimum(0.0, jnp.maximum(x_dist, y_dist))
    phi = phi + softnorm(
        jnp.maximum(jnp.array([1e-3, 1e-3]), jnp.array([x_dist, y_dist]))
    )

    # Subtract the radius of the turtlebot
    phi -= turtle_radius

    return phi


@jax.jit
def calc_box_turtle_contact_wrench(
    box_state: jnp.ndarray,
    turtle_state: jnp.ndarray,
    box_size: jnp.ndarray,
    box_mass: jnp.ndarray,
    mu: jnp.ndarray,
    turtle_radius: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the contact wrench between the box and a turtlebot.

    args:
        box_state: current (x, z, theta, vx, vz, omega) state of the box
        turtle_state: current (x, z, theta, vx, vz, omega) state of the turtlebot
        box_size: side length of box (1-element, 0-dimensional array)
        box_mass: mass of box (1-element, 0-dimensional array)
        mu: coefficient of friction (1-element, 0-dimensional array)
        turtle_radius: turtlebot chassis radius (1-element, 0-dimensional array)
    returns:
        Tuple of
            - contact wrench on box in x, z, and theta.
            - contact force on turtlebot in x and z.
    """
    # Define constants
    psi_s = 0.3  # tangential velocity at which slipping occurs
    c = mu / psi_s  # coefficient of tangential velocity in sticking friction
    contact_k = 300  # spring constant for contact
    contact_d = 2 * jnp.sqrt(box_mass * contact_k)  # critical damping

    # Contact point is approximated as the center of the turtlebot in the box frame
    p_WT = turtle_state[:2]
    p_WB = box_state[:2]
    p_BT_W = p_WT - p_WB
    R_WB = rotation_matrix_2d(box_state[2])
    p_BT = R_WB.T @ p_BT_W  # type: ignore

    # Get velocity of the turtlebot in box frame
    v_WT = turtle_state[3:5]
    v_WB = box_state[3:5]
    v_BT_W = v_WT - v_WB
    v_BT = R_WB.T @ v_BT_W  # type: ignore

    # Get velocity of contact point in box frame
    v_Bcontact = box_state[5] * jnp.array([[0, -1], [1, 0]]) @ p_BT

    # Get velocity of turtlebot relative to contact pt in box frame
    v_contactT_B = v_BT - v_Bcontact

    # Get the normal vector of the contact in the box frame
    right_or_up = p_BT[1] > -p_BT[0]
    left_or_up = p_BT[1] > p_BT[0]
    normal_right = jnp.logical_and(right_or_up, jnp.logical_not(left_or_up))
    normal_up = jnp.logical_and(right_or_up, left_or_up)
    normal_left = jnp.logical_and(jnp.logical_not(right_or_up), left_or_up)
    normal_down = jnp.logical_and(
        jnp.logical_not(right_or_up), jnp.logical_not(left_or_up)
    )
    normal = normal_right * jnp.array([1.0, 0.0])
    normal += normal_left * jnp.array([-1.0, 0.0])
    normal += normal_up * jnp.array([0.0, 1.0])
    normal += normal_down * jnp.array([0.0, -1.0])

    # Get the tangent vector, which is orthogonal to the normal vector
    # and points in the same direction as the relative velocity
    tangential_velocity = (
        v_contactT_B - v_contactT_B.dot(normal) * normal
    )  # relative velocity in tangent direction
    normal_velocity = v_contactT_B.dot(normal)  # scalar, along the normal vector
    tangent = tangential_velocity / (softnorm(tangential_velocity) + 1e-3)

    # Get signed distance
    phi_turtle_box = box_turtle_signed_distance(
        box_state[:3],
        turtle_state[:3],
        box_size,
        turtle_radius,
    )
    # Clip to only consider negative values
    phi_turtle_box = jnp.minimum(0, phi_turtle_box)

    # Use the same simplified friction model as used for ground contact
    normal_force = -contact_k * phi_turtle_box  # scalar, in normal direction
    normal_force = normal_force - contact_d * normal_velocity * (phi_turtle_box < 0)
    sticking = softnorm(tangential_velocity) <= psi_s
    slipping = jnp.logical_not(sticking)
    friction_coefficient = sticking * c * tangential_velocity + slipping * mu * tangent
    tangent_force = -friction_coefficient * normal_force  # vector!

    # Sum up the contact forces in the box frame
    contact_force_B = normal_force * normal + tangent_force
    # transform into the world frame
    contact_force_W = R_WB @ contact_force_B

    # Add the contact force to the box and turtlebot
    p_Tcontact_W = -R_WB @ normal * turtle_radius
    p_Bcontact_W = p_BT_W + p_Tcontact_W
    box_wrench = jnp.zeros(3)
    box_wrench = box_wrench.at[:2].add(-contact_force_W)
    box_wrench = box_wrench.at[2].add(jnp.cross(p_Bcontact_W, -contact_force_W))

    turtle_wrench = jnp.zeros(3)
    turtle_wrench = turtle_wrench.at[:2].add(contact_force_W)
    turtle_wrench = turtle_wrench.at[2].add(jnp.cross(p_Tcontact_W, contact_force_W))

    return box_wrench, turtle_wrench


@jax.jit
def turtlebot_dynamics_step(
    current_state: jnp.ndarray,
    control_input: jnp.ndarray,
    low_level_control_gains: jnp.ndarray,
    external_wrench: jnp.ndarray,
    mu: jnp.ndarray,
    mass: jnp.ndarray,
    chassis_radius: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute the one-step dynamics update for a turtlebot

    args:
        current_state: (6,) array of (x, y, theta, vx, vy, omega) state of the turtlebot
        control_input: (2,) array of (v_d, omega_d) desired velocities of the turtlebot
        low_level_control_gains: (2,) array of proportional gains for v and omega.
        external_wrench (3,) array of external forces and torque (fx, fy, tau)
        mu: 1-element 0-dimensional array of friction coefficient
        mass: 1-element 0-dimensional array of robot mass
        chassis_radius: 1-element 0-dimensional array of robot chassis radius
        dt: the timestep at which to simulate

    returns: the new state of the turtlebot
    """
    # Define constants
    g = 9.81  # m/s^2 -- gravitational acceleration

    # Compute the control force and torque
    theta = current_state[2]
    current_velocity = current_state[3:5]
    current_omega = current_state[5]
    desired_velocity = control_input[0] * jnp.array([jnp.cos(theta), jnp.sin(theta)])
    desired_omega = control_input[1]
    k_v, k_w = low_level_control_gains
    f_control = k_v * (desired_velocity - current_velocity)
    tau_control = k_w * (desired_omega - current_omega)

    # Saturate the control force at the friction limit
    f_normal = g * mass
    f_control_limit = mu * f_normal
    f_control_scale = jnp.minimum(softnorm(f_control), f_control_limit)
    # When we compute norms, we need to add a small amount to make the gradient
    # defined at zero. We also need to add a small bit to avoid dividing by zero
    f_control_scale /= softnorm(f_control) + 1e-3

    # Sum the forces
    f_sum = f_control + external_wrench[:2]
    tau_sum = tau_control + external_wrench[2]

    # Update the state
    new_state = jnp.zeros((6,)) + current_state
    new_state = new_state.at[:2].add(dt * current_velocity)  # position update
    new_state = new_state.at[2].add(dt * current_omega)  # orientation update
    new_state = new_state.at[3:5].add(dt * f_sum / mass)  # xy velocity update
    new_state = new_state.at[5].add(
        dt * tau_sum / (0.5 * mass * chassis_radius ** 2)  # angular velocity update
    )

    return new_state


@jax.jit
def evaluate_quadratic_spline(
    start_pt: jnp.ndarray,
    control_pt: jnp.ndarray,
    end_pt: jnp.ndarray,
    t: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate the quadratic spline defined by the given points at the specified time t.
    This curve starts at start_pt (t=0) and ends at end_pt (t=1) and is continuously
    differentiable.

    The position along the curve at time t [0, 1] is given by

        p(t) = (1 - t)^2 * start_pt + 2 * (1 - t) * t * control_pt + t ^ 2 * end_pt

    the tangent velocity is given by

        v(t) = 2 * (1 - t) * (control_pt - start_pt) + 2 * t * (end_pt - control_pt)

    args:
        start_pt: the starting point of the spline
        control_pt: the control point of the spline
        end_pt: the end point of the spline.
        t: the time at which to evaluate the curve. Must satisfy 0 <= t <= 1
    returns:
        the position and velocity along the curve at the specified time
    """
    # Compute position
    position = (1 - t) ** 2 * start_pt + 2 * (1 - t) * t * control_pt + t ** 2 * end_pt
    # Compute velocity
    velocity = 2 * (1 - t) * (control_pt - start_pt) + 2 * t * (end_pt - control_pt)

    return position, velocity


@jax.jit
def turtlebot_position_velocity_tracking_controller(
    turtle_state: jnp.ndarray,
    tracking_position: jnp.ndarray,
    tracking_velocity: jnp.ndarray,
    high_level_control_gains: jnp.ndarray,
) -> jnp.ndarray:
    """
    Evaluate a tracking controller that steers the turtlebot to follow the indicated
    position and velocity.

    args:
        turtle_state: the current state (x, y, theta, vx, vy, omega) of the turtlebot
        tracking_position: the (x, y) position to track
        tracking_velocity: the (vx, vy) velocity to track
        high_level_control_gains: (2,) array of linear feedback gains for the tracking
            control law.
    returns:
        an array of control inputs (v, omega)
    """
    # Compute desired heading and velocity
    tracking_theta = jax.lax.atan2(tracking_velocity[1], tracking_velocity[0])
    tracking_v = softnorm(tracking_velocity)

    # Also compute the along-track and cross-track error
    turtle_position = turtle_state[:2]
    position_error = turtle_position - tracking_position
    turtle_theta = turtle_state[2]
    turtle_tangent = jnp.array([jnp.cos(turtle_theta), jnp.sin(turtle_theta)])
    R = rotation_matrix_2d(jnp.array(jnp.pi / 2.0))
    turtle_normal = R @ turtle_tangent
    # Along-track is positive if the turtlebot is ahead
    along_track_error = turtle_tangent.dot(position_error)
    # Cross-track is positive if the turtlebot is to the left
    cross_track_error = turtle_normal.dot(position_error)

    # Extract control gains
    k_v, k_w = high_level_control_gains

    # v is set based on tracking velocity and along-track error
    turtle_velocity = turtle_state[3:5]
    turtle_v = softnorm(turtle_velocity)
    v = -0.0 * (turtle_v - tracking_v) - k_v * along_track_error

    # w is set based on the cross-track error and the heading difference
    w = -0.0 * (turtle_theta - tracking_theta) - k_w * cross_track_error

    return jnp.stack((v, w))


@jax.jit
def box_dynamics_step(
    current_state: jnp.ndarray,
    external_wrench: jnp.ndarray,
    mu: jnp.ndarray,
    mass: jnp.ndarray,
    box_size: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute the one-step dynamics update for a box

    args:
        current_state: (6,) array of (x, y, theta, vx, vy, omega) state of the box
        external_wrench (3,) array of external forces and torque (fx, fy, tau)
        mu: 1-element 0-dimensional array of friction coefficient
        mass: 1-element 0-dimensional array of box mass
        box_size: 1-element 0-dimensional array of box size
        dt: the timestep at which to simulate

    returns: the new state of the turtlebot
    """
    # Get the force and torque from the ground
    ground_wrench_on_box = calc_ground_wrench_on_box(current_state, mu, mass)

    # Sum the forces
    f_sum = ground_wrench_on_box[:2] + external_wrench[:2]
    tau_sum = ground_wrench_on_box[2] + external_wrench[2]

    # Update the state
    current_velocity = current_state[3:5]
    current_omega = current_state[5]
    new_state = jnp.zeros((6,)) + current_state
    new_state = new_state.at[:2].add(dt * current_velocity)  # position update
    new_state = new_state.at[2].add(dt * current_omega)  # orientation update
    new_state = new_state.at[3:5].add(dt * f_sum / mass)  # xy velocity update
    new_state = new_state.at[5].add(
        dt * tau_sum / (mass * box_size ** 2 / 6.0)  # angular velocity update
    )

    return new_state


@partial(jax.jit, static_argnames=["n_turtles"])
def multi_agent_box_dynamics_step(
    current_turtlebot_states: jnp.ndarray,
    current_box_state: jnp.ndarray,
    start_pts: jnp.ndarray,
    control_pts: jnp.ndarray,
    end_pts: jnp.ndarray,
    low_level_control_gains: jnp.ndarray,
    high_level_control_gains: jnp.ndarray,
    mu_turtle_ground: jnp.ndarray,
    mu_box_ground: jnp.ndarray,
    mu_box_turtle: jnp.ndarray,
    box_mass: jnp.ndarray,
    turtle_mass: jnp.ndarray,
    box_size: jnp.ndarray,
    turtle_radius: jnp.ndarray,
    t: jnp.ndarray,
    t_final: jnp.ndarray,
    dt: float,
    n_turtles: int,
    desired_box_pose: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the one-step dynamics update for a box and multiple turtlebots

    args:
        current_turtlebot_states: (n_turtles, 6) array of (x, y, theta, vx, vy, omega)
                                 state for each of n_turtles turtlebots
        current_box_state: (6,) array of (x, y, theta, vx, vy, omega) state of the box
        start_pts: (n_turtles, 2) array of (x, y) spline start points for each turtlebot
        control_pts: (n_turtles, 2) array of (x, y) spline control points for each
                     turtlebot
        end_pts: (n_turtles, 2) array of (x, y) spline end points for each turtlebot
        low_level_control_gains: (2,) array of control gains for tracking commanded
            v and omega.
        high_level_control_gains: (2,) array of control gains for tracking a path.
        mu_turtle_ground: 1-element 0-dimensional array of friction coefficient between
                          turtlebot and ground
        mu_box_ground: 1-element 0-dimensional array of friction coefficient between
                       box and ground
        mu_box_turtle: 1-element 0-dimensional array of friction coefficient between
                       box and turtlebot
        box_mass: 1-element 0-dimensional array of box mass
        turtle_mass: 1-element 0-dimensional array of turtlebot mass
        box_size: 1-element 0-dimensional array of box side length
        turtle_radius: 1-element 0-dimensional array of robot chassis radius
        t: 1-element 0-dimensional array of current time
        t_final: 1-element 0-dimensional array of final time for the spline.
        dt: the timestep at which to simulate
        n_turtles: integer number of turtlebots
        desired_box_pose: if provided, stop turtles if we get this close

    returns: the new state of the turtlebots and the new state of the box
    """
    # Loop through each turtlebot and obtain its contact force with the box
    new_turtlebot_states = jnp.zeros_like(current_turtlebot_states)
    total_box_wrench = jnp.zeros((3,))
    for i in range(n_turtles):
        box_contact_wrench, turtle_i_contact_wrench = calc_box_turtle_contact_wrench(
            current_box_state,
            current_turtlebot_states[i],
            box_size,
            box_mass,
            mu_box_turtle,
            turtle_radius,
        )

        # Accumulate the box wrench
        total_box_wrench += box_contact_wrench

        # Get the control for this turtlebot based on the spline
        spline_t = jnp.minimum(t / t_final, 1.0)
        tracking_position, tracking_velocity = evaluate_quadratic_spline(
            start_pts[i], control_pts[i], end_pts[i], spline_t
        )
        control_input = turtlebot_position_velocity_tracking_controller(
            current_turtlebot_states[i],
            tracking_position,
            tracking_velocity,
            high_level_control_gains,
        )

        # Zero the control input if we're close enough to the goal
        if desired_box_pose is not None:
            not_close_enough = softnorm(desired_box_pose - current_box_state[:3]) > 0.1
            control_input = not_close_enough * control_input

        # Update the state of the turtlebot
        new_turtlebot_states = new_turtlebot_states.at[i].set(
            turtlebot_dynamics_step(
                current_turtlebot_states[i],
                control_input,
                low_level_control_gains,
                turtle_i_contact_wrench,
                mu_turtle_ground,
                turtle_mass,
                turtle_radius,
                dt,
            )
        )

    # Update the box state
    new_box_state = box_dynamics_step(
        current_box_state,
        total_box_wrench,
        mu_box_ground,
        box_mass,
        box_size,
        dt,
    )

    return new_turtlebot_states, new_box_state


@jax.jit
def turtle_nn_planner(
    controller_params: List[Tuple[jnp.ndarray, jnp.ndarray]],
    turtle_position: jnp.ndarray,
    final_box_location: jnp.ndarray,
):
    """
    Compute the control inputs for a fleet of turtlebots using a neural network.

    Uses tanh activations everywhere except the final layer

    args:
        controller_params: list of tuples with weights and biases for each layer.
                Weights should be (n_out, n_in) arrays and corresponding biases should
                be (n_out). The first layer should have n_in = n_turtle * 6 + 3, and the
                last layer should have n_out = n_turtle * 3
        turtle_position: (n_turtle, 3) array of current turtlebot (x, y, theta) in the
                         current box frame
        final_box_location: (3,) array of desired x, y, and theta for the box, expressed
                            in the current box frame.
    returns
        an (n_turtles, 2) array of (x, y) spline control points for each turtlebot.
    """
    # Concatenate all inputs into a single feature array
    activations = jnp.concatenate((turtle_position.reshape(-1), final_box_location))

    # Loop through all layers except the final one
    for weight, bias in controller_params[:-1]:
        activations = jnp.dot(weight, activations)
        activations = jnp.tanh(activations) + bias

    # The last layer is just linear
    weight, bias = controller_params[-1]
    output = jnp.dot(weight, activations) + bias

    return output


def mam_simulate_single_push_two_turtles(
    design_params: jnp.ndarray,
    exogenous_sample: jnp.ndarray,
    layer_widths: Tuple[int],
    dt: float,
):
    """Simulate the multi-agent manipulation system executing a single push lasting 1
    second (includes an 0.5 second setup time, so simulates for 1.5 seconds in total).
    Assumes two turtlebots.

    args:
        design_params: an array containing the weights and biases of a controller neural
            network.
        exogenous_sample: (12,) array of
            (mu_turtle_ground, mu_box_ground, mu_box_turtle, desired_box_x,
            desired_box_y, desired_box_theta, turtle_1_x0, turtle_1_y0,
            turtle_1_theta_0, turtle_2_x0, turtle_2_y0, turtle_2_theta0)

            where the turtlebot states are displacements from nominal.
        layer_widths: number of units in each layer. First element should be
                      15. Last element should be 6
        dt: the duration of each time step. Causes an error if time_steps * dt < 1.0 s
        n_turtles: integer number of turtlebots

    returns:
        (6,) array containing the final location of the box relative to the desired
        location
    """
    # Set constants
    box_size = jnp.array(0.61)
    turtle_mass = jnp.array(2.7)
    chassis_radius = jnp.array(0.08)
    low_level_control_gains = jnp.array([5.0, 0.1])
    n_turtles = 2

    # Extract the network weights and biases and control gains (design parameters)
    high_level_control_gains = design_params[:2]
    controller_params = []
    n_layers = len(layer_widths)
    assert layer_widths[0] == n_turtles * 3 + 3
    assert layer_widths[-1] == n_turtles * 2
    start_index = 2
    for i in range(1, n_layers):
        input_width = layer_widths[i - 1]
        output_width = layer_widths[i]

        num_weight_values = input_width * output_width
        weight = design_params[start_index : start_index + num_weight_values]
        weight = weight.reshape(output_width, input_width)
        start_index += num_weight_values

        num_bias_values = output_width
        bias = design_params[start_index : start_index + num_bias_values]
        start_index += num_bias_values

        controller_params.append((weight, bias))

    # Extract the exogenous parameters
    mu_turtle_ground = exogenous_sample[0]
    mu_box_ground = exogenous_sample[1]
    mu_box_turtle = exogenous_sample[2]
    box_mass = exogenous_sample[3]
    desired_box_pose = exogenous_sample[4:7]
    # There are still exogenous parameters for the initial pose of each turtlebot, which
    # we deal with next.

    # Let's work in coordinates relative to the initial position of the box. As an extra
    # bonus, since the box is symmetric, let's rotate those coordinates so that the
    # desired location of the box is in the first quadrant.
    initial_box_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Assume we can get to a starting position with the turtlebots near the middle of
    # the bottom and left faces of the box, but with some exogenous offset.
    initial_turtle_state = jnp.array(
        [
            [-(box_size / 2 + chassis_radius), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -(box_size / 2 + chassis_radius), jnp.pi / 2, 0.0, 0.0, 0.0],
        ]
    )
    for i in range(n_turtles):
        initial_turtle_state = initial_turtle_state.at[i, :3].add(
            exogenous_sample[7 + i * 3 : 7 + (i + 1) * 3]
        )

    # This random displacement may have caused some overlap between the turtlebots and
    # the box, so simulate for some settling time to let them move out of overlap,
    # then reset the coordinate system to the box pose
    settle_start_pts = initial_turtle_state[:, :2]
    settle_control_pts = initial_turtle_state[:, :2]
    settle_end_pts = initial_turtle_state[:, :2]
    settle_time = 0.5
    settle_steps = int(settle_time // dt)
    for t in range(settle_steps):
        initial_turtle_state, initial_box_state = multi_agent_box_dynamics_step(
            initial_turtle_state,
            initial_box_state,
            settle_start_pts,
            settle_control_pts,
            settle_end_pts,
            low_level_control_gains,
            high_level_control_gains,
            mu_turtle_ground,
            mu_box_ground,
            mu_box_turtle,
            box_mass,
            turtle_mass,
            box_size,
            chassis_radius,
            t * dt,
            settle_time,
            dt,
            n_turtles,
        )

    # reset coordinates by zero-ing velocity,
    initial_turtle_state = initial_turtle_state.at[:, 3:].set(0.0)
    initial_box_state = initial_box_state.at[3:].set(0.0)
    # setting the new box pose to the origin,
    initial_turtle_state = initial_turtle_state.at[:, :2].add(-initial_box_state[:2])
    initial_box_state = initial_box_state.at[:2].add(-initial_box_state[:2])
    # and rotating everything into the box frame
    theta = initial_box_state[2]
    initial_turtle_state = initial_turtle_state.at[:, 2].add(-theta)
    initial_box_state = initial_box_state.at[2].add(-theta)
    R_WB = rotation_matrix_2d(theta)
    R_BW = R_WB.T  # type: ignore
    new_turtle_xy = R_BW @ initial_turtle_state[:, :2].T
    new_turtle_xy = new_turtle_xy.T
    initial_turtle_state = initial_turtle_state.at[:, :2].set(new_turtle_xy)

    # Set the spline start points based on the initial conditions
    start_pts = initial_turtle_state[:, :2]
    R_WBfinal = rotation_matrix_2d(desired_box_pose[2])
    R_BfinalW = R_WBfinal.T  # type: ignore
    p_BfinalEndpts = jnp.array(
        [
            [-(box_size / 2 + chassis_radius), 0.0],
            [0.0, -(box_size / 2 + chassis_radius)],
        ]
    )
    # The spatial algebra feels screwy here but it works??
    end_pts = desired_box_pose[:2] + R_BfinalW @ p_BfinalEndpts

    # Now get the spline control points from the network (we treat these as additions
    # to the midpoint of the start and end points).
    control_pts = turtle_nn_planner(
        controller_params, initial_turtle_state[:, :3], desired_box_pose
    )
    control_pts = control_pts.reshape(n_turtles, 2)
    control_pts = (start_pts + end_pts) / 2.0 + control_pts

    # Simulate! This simulation proceeds as follows:
    #   2.) Set v and omega based on the spline for push_time.
    push_time = 4.0
    push_steps = int(push_time // dt)

    # Set up arrays to store the simulation trace
    turtle_states = jnp.zeros((push_steps, 2, 6))
    turtle_states = turtle_states.at[0].set(initial_turtle_state)
    box_states = jnp.zeros((push_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)

    # Simulate the push phase
    for t in range(push_steps):
        new_turtle_state, new_box_state = multi_agent_box_dynamics_step(
            turtle_states[t],
            box_states[t],
            start_pts,
            control_pts,
            end_pts,
            low_level_control_gains,
            high_level_control_gains,
            mu_turtle_ground,
            mu_box_ground,
            mu_box_turtle,
            box_mass,
            turtle_mass,
            box_size,
            chassis_radius,
            t * dt,
            push_time * 0.5,
            dt,
            n_turtles,
            desired_box_pose=desired_box_pose,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    # Return the trace of box and turtlebot states
    return turtle_states, box_states
