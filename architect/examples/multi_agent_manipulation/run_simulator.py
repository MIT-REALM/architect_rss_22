import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from architect.examples.multi_agent_manipulation.mam_design_problem import (
    make_mam_design_problem,
)
from architect.examples.multi_agent_manipulation.mam_plotting import (
    plot_box_trajectory,
    plot_turtle_trajectory,
    make_box_patches,
)
from architect.examples.multi_agent_manipulation.mam_simulator import (
    turtlebot_dynamics_step,
    box_turtle_signed_distance,
    box_dynamics_step,
    multi_agent_box_dynamics_step,
    mam_simulate_single_push_two_turtles,
    evaluate_quadratic_spline,
)


def test_turtlebot_dynamics():
    T = 5
    dt = 0.01
    n_steps = int(T // dt)
    mu = jnp.array(1.0)
    mass = jnp.array(2.7)
    chassis_radius = jnp.array(0.08)
    low_level_control_gains = jnp.array([5.0, 0.1])
    initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states = jnp.zeros((n_steps, 6))
    states = states.at[0].set(initial_state)
    for t in range(n_steps - 1):
        control_input = jnp.array([2.0, 1.0])
        external_wrench = jnp.zeros((3,))
        new_state = turtlebot_dynamics_step(
            states[t],
            control_input,
            low_level_control_gains,
            external_wrench,
            mu,
            mass,
            chassis_radius,
            dt,
        )
        states = states.at[t + 1].set(new_state)

    plot_turtle_trajectory(states, chassis_radius, 20, plt.gca())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def test_signed_distance():
    # Test signed distance calculation
    box_pose = jnp.array([-0.4, 0.2, -0.1])
    box_size = jnp.array(0.61)

    turtle_x = jnp.linspace(-1, 1, 100)
    turtle_z = jnp.linspace(-1, 1, 100)

    turtle_X, turtle_Z = jnp.meshgrid(turtle_x, turtle_z)
    turtle_XZ = jnp.stack((turtle_X, turtle_Z)).reshape(2, 10000).T

    chassis_radius = jnp.array(0.08)
    f_phi = lambda turtle_pose: box_turtle_signed_distance(
        box_pose, turtle_pose, box_size, chassis_radius
    )
    Phi = jax.vmap(f_phi, in_axes=0)(turtle_XZ).reshape(100, 100)
    fig, ax = plt.subplots()
    contours = ax.contourf(turtle_X, turtle_Z, Phi, levels=10)
    ax.contour(turtle_X, turtle_Z, Phi, levels=[0], colors="r")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$z$")
    ax.set_title(
        r"SDF for a cube at $(x, z, \theta)$ = "
        + "({:.2f}, {:.2f}, {:.2f})".format(box_pose[0], box_pose[1], box_pose[2])
    )
    ax.set_aspect("equal")
    fig.colorbar(contours)
    plt.show()


def test_box_dynamics():
    # Test 1 box
    T = 1.0
    dt = 0.01
    n_steps = int(T // dt)
    mu = jnp.array(1.0)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.61)
    initial_box_state = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    box_states = jnp.zeros((n_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)
    for t in range(n_steps - 1):
        external_wrench = jnp.zeros((3,))
        new_box_state = box_dynamics_step(
            box_states[t],
            external_wrench,
            mu,
            box_mass,
            box_size,
            dt,
        )
        box_states = box_states.at[t + 1].set(new_box_state)

    plot_box_trajectory(box_states, box_size, 20, plt.gca())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def test_turtle_spline_following():
    # Test box and 1 turtlebot, but make sure they don't make contact
    T = 10.0
    dt = 0.01
    n_steps = int(T // dt)
    mu_box_turtle = jnp.array(0.2)
    mu_turtle_ground = jnp.array(0.7)
    mu_box_ground = jnp.array(0.5)
    turtle_mass = jnp.array(2.7)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.61)
    chassis_radius = jnp.array(0.08)
    low_level_control_gains = jnp.array([5.0, 0.1])
    high_level_control_gains = jnp.array([12.0, 5.0])
    initial_turtle_state = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_box_state = jnp.array([-2, 0.0, 0.0, 0.0, 0.0, 0.0])
    turtle_states = jnp.zeros((n_steps, 1, 6))
    turtle_states = turtle_states.at[0].set(initial_turtle_state)
    box_states = jnp.zeros((n_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)

    # Define the spline from the start point to and end point
    start_pts = jnp.zeros((1, 2))
    control_pts = jnp.array([[0.0, 2.0]])
    end_pts = jnp.ones((1, 2))

    for t in range(n_steps - 1):
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
            T * 0.5,
            dt,
            1,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    plot_turtle_trajectory(turtle_states[:, 0, :], chassis_radius, 20, plt.gca())

    spline_fn = lambda t: evaluate_quadratic_spline(
        start_pts[0], control_pts[0], end_pts[0], t
    )
    spline_pts, _ = jax.vmap(spline_fn, in_axes=0)(jnp.linspace(0.0, 1.0, 100))
    plt.plot(spline_pts[:, 0], spline_pts[:, 1], "k:")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def test_box_turtle_dynamics():
    # Test box and 1 turtlebot
    T = 10.0
    dt = 0.01
    n_steps = int(T // dt)
    mu_box_turtle = jnp.array(0.2)
    mu_turtle_ground = jnp.array(0.7)
    mu_box_ground = jnp.array(0.5)
    turtle_mass = jnp.array(2.7)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.61)
    chassis_radius = jnp.array(0.08)
    low_level_control_gains = jnp.array([5.0, 0.1])
    high_level_control_gains = jnp.array([12.0, 5.0])
    initial_turtle_state = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_box_state = jnp.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0])
    turtle_states = jnp.zeros((n_steps, 1, 6))
    turtle_states = turtle_states.at[0].set(initial_turtle_state)
    box_states = jnp.zeros((n_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)

    # Define the spline from the start point to and end point
    start_pts = jnp.zeros((1, 2))
    control_pts = jnp.array([[2.0, 0.0]])
    end_pts = jnp.ones((1, 2))

    for t in range(n_steps - 1):
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
            T * 0.5,
            dt,
            1,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    plot_box_trajectory(box_states, box_size, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], chassis_radius, 20, plt.gca())

    spline_fn = lambda t: evaluate_quadratic_spline(
        start_pts[0], control_pts[0], end_pts[0], t
    )
    spline_pts, _ = jax.vmap(spline_fn, in_axes=0)(jnp.linspace(0.0, 1.0, 100))
    plt.plot(spline_pts[:, 0], spline_pts[:, 1], "k:")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def test_box_two_turtles_dynamics():
    # Test box and 2 turtlebot
    T = 10.0
    dt = 0.01
    n_steps = int(T // dt)
    mu_box_turtle = jnp.array(0.2)
    mu_turtle_ground = jnp.array(0.7)
    mu_box_ground = jnp.array(0.5)
    turtle_mass = jnp.array(2.7)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.61)
    chassis_radius = jnp.array(0.08)
    low_level_control_gains = jnp.array([5.0, 0.1])
    high_level_control_gains = jnp.array([2.0, 5.0])
    initial_turtle_state = jnp.array(
        [
            [-0.35, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.35, jnp.pi / 2, 0.0, 0.0, 0.0],
        ]
    )
    initial_box_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    turtle_states = jnp.zeros((n_steps, 2, 6))
    turtle_states = turtle_states.at[0].set(initial_turtle_state)
    box_states = jnp.zeros((n_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)

    # Define the spline from the start point to and end point
    start_pts = jnp.array([[-0.35, 0.0], [0.0, -0.35]])
    control_pts = jnp.array([[0.7, 0.0], [0.0, 0.7]])
    end_pts = jnp.array([[1.0 - 0.35, 1.0], [1.0, 1.0 - 0.35]])

    for t in range(n_steps - 1):
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
            T * 0.5,
            dt,
            2,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    plot_box_trajectory(box_states, box_size, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], chassis_radius, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 1, :], chassis_radius, 20, plt.gca())

    spline0_fn = lambda t: evaluate_quadratic_spline(
        start_pts[0], control_pts[0], end_pts[0], t
    )
    spline1_fn = lambda t: evaluate_quadratic_spline(
        start_pts[1], control_pts[1], end_pts[1], t
    )
    spline_pts, _ = jax.vmap(spline0_fn, in_axes=0)(jnp.linspace(0.0, 1.0, 100))
    plt.plot(spline_pts[:, 0], spline_pts[:, 1], "k:")
    spline_pts, _ = jax.vmap(spline1_fn, in_axes=0)(jnp.linspace(0.0, 1.0, 100))
    plt.plot(spline_pts[:, 0], spline_pts[:, 1], "k:")

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.gca().set_aspect("equal")
    # plt.show()


def test_push():
    # Test mam_simulate_single_push_two_turtles function
    prng_key = jax.random.PRNGKey(0)

    # Make the design problem
    layer_widths = (2 * 3 + 3, 32, 2 * 2)
    dt = 0.01
    prng_key, subkey = jax.random.split(prng_key)
    mam_design_problem = make_mam_design_problem(layer_widths, dt, subkey)

    logfile = (
        "logs/multi_agent_manipulation/real_turtle_dimensions/"
        "design_optimization_512_samples_0p5x0p5xpi_4_target_"
        "9x32x4_network_spline_1e-1_variance_weight_solution.csv"
    )
    design_param_values = jnp.array(
        np.loadtxt(
            logfile,
            delimiter=",",
        )
    )

    # Sample exogenous parameters
    prng_key, subkey = jax.random.split(prng_key)
    exogenous_sample = mam_design_problem.exogenous_params.sample(subkey)

    # Simulate
    turtle_states, box_states = mam_simulate_single_push_two_turtles(
        design_param_values,
        exogenous_sample,
        layer_widths,
        dt,
    )

    plot_box_trajectory(box_states, 0.5, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], 0.1, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 1, :], 0.1, 20, plt.gca())
    desired_box_pose = exogenous_sample[4:7]
    make_box_patches(desired_box_pose, 1.0, 0.5, plt.gca(), hatch=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-0.75, 1.0])
    plt.ylim([-0.75, 1.0])
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    # test_turtlebot_dynamics()
    # test_box_dynamics()
    # test_turtle_spline_following()
    # test_box_turtle_dynamics()
    # test_box_two_turtles_dynamics()
    test_push()
