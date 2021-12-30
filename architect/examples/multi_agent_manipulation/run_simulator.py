import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.examples.multi_agent_manipulation.mam_design_parameters import (
    MAMDesignParameters,
)
from architect.examples.multi_agent_manipulation.mam_exogenous_parameters import (
    MAMExogenousParameters,
)
from architect.examples.multi_agent_manipulation.mam_plotting import (
    plot_box_trajectory,
    plot_turtle_trajectory,
)
from architect.examples.multi_agent_manipulation.mam_simulator import (
    turtlebot_dynamics_step,
    box_turtle_signed_distance,
    box_dynamics_step,
    multi_agent_box_dynamics_step,
    mam_simulate_single_push_two_turtles,
)


def test_turtlebot_dynamics():
    T = 5
    dt = 0.01
    n_steps = int(T // dt)
    mu = jnp.array(1.0)
    mass = jnp.array(1.0)
    chassis_radius = jnp.array(0.1)
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
    box_size = jnp.array(0.5)

    turtle_x = jnp.linspace(-1, 1, 100)
    turtle_z = jnp.linspace(-1, 1, 100)

    turtle_X, turtle_Z = jnp.meshgrid(turtle_x, turtle_z)
    turtle_XZ = jnp.stack((turtle_X, turtle_Z)).reshape(2, 10000).T

    chassis_radius = jnp.array(0.1)
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
    box_size = jnp.array(0.5)
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


def test_box_turtle_dynamics():
    # Test box and 1 turtlebot
    T = 3.0
    dt = 0.01
    n_steps = int(T // dt)
    mu_box_turtle = jnp.array(0.1)
    mu_turtle_ground = jnp.array(0.7)
    mu_box_ground = jnp.array(0.5)
    turtle_mass = jnp.array(1.0)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.5)
    chassis_radius = jnp.array(0.1)
    low_level_control_gains = jnp.array([5.0, 0.1])
    initial_turtle_state = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_box_state = jnp.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0])
    turtle_states = jnp.zeros((n_steps, 1, 6))
    turtle_states = turtle_states.at[0].set(initial_turtle_state)
    box_states = jnp.zeros((n_steps, 6))
    box_states = box_states.at[0].set(initial_box_state)
    for t in range(n_steps - 1):
        control_input = jnp.array([[1.0, 0.0]])
        new_turtle_state, new_box_state = multi_agent_box_dynamics_step(
            turtle_states[t],
            box_states[t],
            control_input,
            low_level_control_gains,
            mu_turtle_ground,
            mu_box_ground,
            mu_box_turtle,
            box_mass,
            turtle_mass,
            box_size,
            chassis_radius,
            dt,
            1,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    plot_box_trajectory(box_states, box_size, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], chassis_radius, 20, plt.gca())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def test_box_two_turtles_dynamics():
    # Test box and 2 turtlebot
    T = 1.0
    dt = 0.01
    n_steps = int(T // dt)
    mu_box_turtle = jnp.array(0.1)
    mu_turtle_ground = jnp.array(0.7)
    mu_box_ground = jnp.array(0.5)
    turtle_mass = jnp.array(1.0)
    box_mass = jnp.array(1.0)
    box_size = jnp.array(0.5)
    chassis_radius = jnp.array(0.1)
    low_level_control_gains = jnp.array([5.0, 0.1])
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
    for t in range(n_steps - 1):
        control_input = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        new_turtle_state, new_box_state = multi_agent_box_dynamics_step(
            turtle_states[t],
            box_states[t],
            control_input,
            low_level_control_gains,
            mu_turtle_ground,
            mu_box_ground,
            mu_box_turtle,
            box_mass,
            turtle_mass,
            box_size,
            chassis_radius,
            dt,
            2,
        )
        turtle_states = turtle_states.at[t + 1].set(new_turtle_state)
        box_states = box_states.at[t + 1].set(new_box_state)

    plot_box_trajectory(box_states, box_size, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], chassis_radius, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 1, :], chassis_radius, 20, plt.gca())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def test_push():
    # Test mam_simulate_single_push_two_turtles function
    dt = 0.01
    mu_box_turtle_range = jnp.array([0.05, 0.2])
    mu_turtle_ground_range = jnp.array([0.6, 0.8])
    mu_box_ground_range = jnp.array([0.4, 0.6])
    box_mass_range = jnp.array([0.9, 1.1])
    desired_box_pose_range = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    turtlebot_displacement_covariance = (0.1 ** 2) * jnp.eye(3)
    exogenous_params = MAMExogenousParameters(
        mu_turtle_ground_range,
        mu_box_ground_range,
        mu_box_turtle_range,
        box_mass_range,
        desired_box_pose_range,
        turtlebot_displacement_covariance,
        2,
    )

    layer_widths = [15, 32, 32, 6]
    prng_key = jax.random.PRNGKey(0)
    prng_key, subkey = jax.random.split(prng_key)
    design_params = MAMDesignParameters(subkey, layer_widths)

    # Sample exogenous parameters
    prng_key, subkey = jax.random.split(prng_key)
    exogenous_sample = exogenous_params.sample(subkey)

    # Simulate
    turtle_states, box_states = mam_simulate_single_push_two_turtles(
        design_params.get_values(),
        exogenous_sample,
        layer_widths,
        dt,
    )

    plot_box_trajectory(box_states, 0.5, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], 0.1, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 1, :], 0.1, 20, plt.gca())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    # test_turtlebot_dynamics()
    # test_box_dynamics()
    # test_box_turtle_dynamics()
    # test_box_two_turtles_dynamics()
    test_push()
