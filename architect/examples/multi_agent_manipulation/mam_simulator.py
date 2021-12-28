"""Define a simulator for the multi-agent manipulation (MAM) task"""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.components.geometry.transforms_2d import rotation_matrix_2d


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
    psi = jnp.linalg.norm(v_WB + 1e-3)  # Magnitude of tangential velocity
    sliding_direction = psi / (jnp.linalg.norm(psi + 1e-3) + 1e-3)
    sticking = psi <= psi_s
    slipping = jnp.logical_not(sticking)
    friction_coefficient = sticking * c * psi + slipping * mu * sliding_direction
    translational_friction_force = -friction_coefficient * f_normal

    # Compute rotational friction (this is an approximation)
    psi = jnp.linalg.norm(w_WB + 1e-3)  # Magnitude of angular velocity
    sliding_direction = jnp.sign(w_WB)
    sticking = psi <= psi_s
    slipping = jnp.logical_not(sticking)
    friction_coefficient = sticking * c * psi + slipping * mu * sliding_direction
    rotational_friction_torque = -friction_coefficient * f_normal

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
    phi = phi + jnp.linalg.norm(
        jnp.maximum(jnp.array([1e-3, 1e-3]), jnp.array([x_dist, y_dist]))
    )

    # Subtract the radius of the turtlebot
    phi -= chassis_radius

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
    contact_k = 1000  # spring constant for contact
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
    tangent = tangential_velocity / (jnp.linalg.norm(tangential_velocity + 1e-3) + 1e-3)

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
    sticking = jnp.linalg.norm(tangential_velocity + 1e-3) <= psi_s
    slipping = jnp.logical_not(sticking)
    friction_coefficient = sticking * c * tangential_velocity + slipping * mu * tangent
    tangent_force = -friction_coefficient * normal_force  # vector!

    # Sum up the contact forces in the box frame
    contact_force_B = normal_force * normal + tangent_force
    # transform into the world frame
    contact_force_W = R_WB @ contact_force_B

    # Add the contact force to the box and turtlebot
    box_wrench = jnp.zeros(3)
    box_wrench = box_wrench.at[:2].add(-contact_force_W)
    box_wrench = box_wrench.at[2].add(jnp.cross(p_BT_W, -contact_force_W))

    turtle_forces = contact_force_W

    return box_wrench, turtle_forces


@jax.jit
def turtlebot_dynamics_step(
    current_state: jnp.ndarray,
    control_input: jnp.ndarray,
    control_gains: jnp.ndarray,
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
        control_gains: (2,) array of proportional control gains for v and omega.
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
    k_v, k_w = control_gains
    f_control = k_v * (desired_velocity - current_velocity)
    tau_control = k_w * (desired_omega - current_omega)

    # Saturate the control force at the friction limit
    f_normal = g * mass
    f_control_limit = mu * f_normal
    f_control_scale = jnp.minimum(jnp.linalg.norm(f_control), f_control_limit)
    # When we compute norms, we need to add a small amount to make the gradient
    # defined at zero. We also need to add a small bit to avoid dividing by zero
    f_control_scale /= jnp.linalg.norm(f_control + 1e-3) + 1e-3

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
def box_dynamics_step(
    current_state: jnp.ndarray,
    external_wrench: jnp.ndarray,
    mu: jnp.ndarray,
    mass: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute the one-step dynamics update for a box

    args:
        current_state: (6,) array of (x, y, theta, vx, vy, omega) state of the box
        external_wrench (3,) array of external forces and torque (fx, fy, tau)
        mu: 1-element 0-dimensional array of friction coefficient
        mass: 1-element 0-dimensional array of box mass
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
    current_turtlebot_state: jnp.ndarray,
    current_box_state: jnp.ndarray,
    control_input: jnp.ndarray,
    control_gains: jnp.ndarray,
    mu: jnp.ndarray,
    box_mass: jnp.ndarray,
    turtle_mass: jnp.ndarray,
    box_size: jnp.ndarray,
    turtle_radius: jnp.ndarray,
    dt: float,
    n_turtles: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the one-step dynamics update for a box and multiple turtlebots

    args:
        current_turtlebot_state: (n_turtles, 6) array of (x, y, theta, vx, vy, omega)
                                 state for each of n_turtles turtlebots
        current_box_state: (6,) array of (x, y, theta, vx, vy, omega) state of the box
        control_input: (n_turtles, 2) array of (v_d, omega_d) desired velocities for
                       each turtlebot
        control_gains: (2,) array of proportional control gains for v and omega.
        mu: 1-element 0-dimensional array of friction coefficient
        box_mass: 1-element 0-dimensional array of box mass
        turtle_mass: 1-element 0-dimensional array of turtlebot mass
        box_size: 1-element 0-dimensional array of box side length
        turtle_radius: 1-element 0-dimensional array of robot chassis radius
        dt: the timestep at which to simulate
        n_turtles: integer number of turtlebots

    returns: the new state of the turtlebots and the new state of the box
    """
    # Loop through each turtlebot and obtain its contact force with the box
    new_turtlebot_state = jnp.zeros_like(current_turtlebot_state)
    total_box_wrench = jnp.zeros((3,))
    for i in range(n_turtles):
        box_contact_wrench, turtle_i_contact_force = calc_box_turtle_contact_wrench(
            current_box_state,
            current_turtlebot_state[i],
            box_size,
            box_mass,
            mu,
            turtle_radius,
        )

        # Accumulate the box wrench
        total_box_wrench += box_contact_wrench

        # Update the state of the turtlebot
        turtle_i_contact_wrench = jnp.zeros((3,))
        turtle_i_contact_wrench = turtle_i_contact_wrench.at[:2].set(
            turtle_i_contact_force
        )
        new_turtlebot_state = new_turtlebot_state.at[i].set(
            turtlebot_dynamics_step(
                current_turtlebot_state[i],
                control_input[i],
                control_gains,
                turtle_i_contact_force,
                mu,
                turtle_mass,
                turtle_radius,
                dt,
            )
        )

    # Update the box state
    new_box_state = box_dynamics_step(
        current_box_state,
        total_box_wrench,
        mu,
        box_mass,
        dt,
    )

    return new_turtlebot_state, new_box_state


if __name__ == "__main__":
    # # Test turtlebot dynamics
    # T = 10
    # dt = 0.01
    # n_steps = int(T // dt)
    # mu = jnp.array(1.0)
    # mass = jnp.array(1.0)
    # chassis_radius = jnp.array(0.1)
    # control_gains = jnp.array([5.0, 0.1])
    # initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # states = jnp.zeros((n_steps, 6))
    # states = states.at[0].set(initial_state)
    # for t in range(n_steps - 1):
    #     control_input = jnp.array([2.0, 1.0])
    #     external_wrench = jnp.zeros((3,))
    #     new_state = turtlebot_dynamics_step(
    #         states[t],
    #         control_input,
    #         control_gains,
    #         external_wrench,
    #         mu,
    #         mass,
    #         chassis_radius,
    #         dt,
    #     )
    #     states = states.at[t + 1].set(new_state)

    # plt.plot(states[:, 0], states[:, 1])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.gca().set_aspect("equal")
    # plt.show()

    # # Test signed distance calculation
    # box_pose = jnp.array([-0.4, 0.2, -0.1])
    # box_size = jnp.array(0.5)

    # turtle_x = jnp.linspace(-1, 1, 100)
    # turtle_z = jnp.linspace(-1, 1, 100)

    # turtle_X, turtle_Z = jnp.meshgrid(turtle_x, turtle_z)
    # turtle_XZ = jnp.stack((turtle_X, turtle_Z)).reshape(2, 10000).T

    # f_phi = lambda turtle_pose: box_turtle_signed_distance(
    #     box_pose, turtle_pose, box_size, chassis_radius
    # )
    # Phi = jax.vmap(f_phi, in_axes=0)(turtle_XZ).reshape(100, 100)
    # fig, ax = plt.subplots()
    # contours = ax.contourf(turtle_X, turtle_Z, Phi, levels=10)
    # zero_contour = ax.contour(turtle_X, turtle_Z, Phi, levels=[0], colors="r")
    # ax.set_xlabel(r"$x$")
    # ax.set_ylabel(r"$z$")
    # ax.set_title(
    #     r"SDF for a cube at $(x, z, \theta)$ = "
    #     + "({:.2f}, {:.2f}, {:.2f})".format(box_pose[0], box_pose[1], box_pose[2])
    # )
    # ax.set_aspect("equal")
    # fig.colorbar(contours)
    # plt.show()

    # Test box and 1 turtlebot
    T = 1
    dt = 0.01
    n_steps = int(T // dt)
    mu = jnp.array(1.0)
    turtle_mass = jnp.array(1.0)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.5)
    chassis_radius = jnp.array(0.1)
    control_gains = jnp.array([5.0, 0.1])
    initial_turtle_state = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_box_state = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    turtle_states = jnp.zeros((n_steps, 1, 6))
    turtle_states = turtle_states.at[0].set(initial_turtle_state)
    box_states = jnp.zeros((n_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)
    for t in range(n_steps - 1):
        control_input = jnp.array([[2.0, 0.0]])
        external_wrench = jnp.zeros((3,))
        new_turtle_state, new_box_state = multi_agent_box_dynamics_step(
            turtle_states[t],
            box_states[t],
            control_input,
            control_gains,
            mu,
            box_mass,
            turtle_mass,
            box_size,
            chassis_radius,
            dt,
            1,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    plt.plot(turtle_states[:, 0, 0], turtle_states[:, 0, 1])
    plt.plot(box_states[:, 0], box_states[:, 1], "s-")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()
