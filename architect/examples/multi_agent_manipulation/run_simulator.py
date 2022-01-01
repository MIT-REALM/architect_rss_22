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
    make_box_patches
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

    layer_widths = [15, 6]
    prng_key = jax.random.PRNGKey(0)
    prng_key, subkey = jax.random.split(prng_key)
    design_params = MAMDesignParameters(subkey, layer_widths)

    # Load values from optimization
    design_params.set_values(
        jnp.array(
            [
                -7.04576671e-02,
                -9.21882749e-01,
                -1.62223041e-01,
                -2.47734152e-02,
                6.42141923e-02,
                -1.16768099e-01,
                3.97659421e-01,
                -2.97620744e-01,
                7.34597147e-01,
                8.36179126e-04,
                8.69541392e-02,
                -4.81038913e-03,
                1.45657770e-02,
                1.25289336e-01,
                -4.49741870e-01,
                -4.87457156e-01,
                2.53358305e-01,
                -2.85509288e-01,
                -9.68895480e-02,
                3.88645269e-02,
                1.98535010e-01,
                -3.48910280e-02,
                1.17699109e-01,
                -2.01122597e-01,
                1.91382423e-01,
                1.66218638e-01,
                -4.42797830e-03,
                4.40929443e-01,
                2.42168158e-01,
                5.35399199e-01,
                3.66655290e-01,
                -1.16849697e00,
                -3.17824066e-01,
                -1.11400425e-01,
                1.19768791e-01,
                -2.43930593e-02,
                6.20200932e-01,
                -1.47774622e-01,
                -1.99982271e-01,
                -4.38794158e-02,
                -1.83652937e-02,
                7.43897483e-02,
                5.50693367e-03,
                6.20339159e-03,
                -2.49915794e-02,
                -1.31198717e-02,
                -6.55680895e-01,
                1.31707042e-01,
                -7.78124928e-02,
                3.08433245e-03,
                1.14277817e-01,
                1.02980518e00,
                -5.12877882e-01,
                -2.67077200e-02,
                8.53846781e-03,
                -1.43913552e-01,
                -1.78483292e-01,
                -5.64252883e-02,
                -1.21457420e-01,
                -2.07552454e-03,
                4.76801284e-02,
                4.78325248e-01,
                1.20024085e-01,
                1.51159957e-01,
                -9.13563073e-02,
                -3.08981743e-02,
                -5.68452552e-02,
                -3.30677569e-01,
                -1.92424640e-01,
                1.14437558e-01,
                -1.45058617e-01,
                2.33762369e-01,
                5.51105976e-01,
                6.47466302e-01,
                1.27754295e00,
                2.18899593e-01,
                -2.97684282e-01,
                -4.02013585e-02,
                -5.57296239e-02,
                1.94659501e-01,
                2.74614133e-02,
                6.00912213e-01,
                -1.40517667e-01,
                -2.18123347e-01,
                2.18643829e-01,
                -5.49983159e-02,
                1.60712376e-02,
                -2.05764323e-02,
                1.10168651e-01,
                -1.78522468e-01,
                4.82704282e-01,
                1.01096421e-01,
                3.06500643e-01,
                1.66432306e-01,
                -3.11918974e-01,
                3.31087261e-01,
            ]
        )
    )

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
    desired_box_pose = exogenous_sample[4:7]
    make_box_patches(desired_box_pose, 1.0, 0.5, plt.gca(), hatch=True)
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
