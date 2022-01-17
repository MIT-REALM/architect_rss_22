"""Define a simulator for the AGV"""
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.components.dynamics.dubins import (
    dubins_next_state,
    dubins_linearized_dynamics,
)
from architect.components.estimation.ekf import dt_ekf_predict_covariance, dt_ekf_update
from architect.components.sensing.range_beacons import (
    beacon_range_measurements,
    beacon_range_measurement_lin,
)
from architect.components.sensing.compass import (
    compass_measurements,
    compass_measurement_lin,
)
from architect.design import DesignParameters

from architect.examples.agv_localization.agv_exogenous_parameters import (
    AGVExogenousParameters,
)


@jax.jit
def navigation_function(position_xy: jnp.ndarray) -> jnp.ndarray:
    """Computes the navigation function for moving to the right through a door.

    args:
        position_xy: the (2,) array containing the current estimate of (x, y) position
    returns:
        the value of the navigation function
    """
    # As a base, the navigation function decreases around the origin
    V = 2 * (position_xy ** 2).sum()

    # For each obstacle (a bunch of discs) we add a repulsive term
    obstacle_radius = 0.2
    obstacle_factor = 0.05
    door_width = 0.4
    obstacle_locations = jnp.array(
        [
            [-1.5, door_width],
            [-1.51, door_width + 0.1],
            [-1.52, door_width + 0.2],
            [-1.53, door_width + 0.3],
            [-1.54, door_width + 0.4],
            [-1.55, door_width + 0.5],
            [-1.5, -door_width],
            [-1.51, -door_width - 0.1],
            [-1.52, -door_width - 0.2],
            [-1.53, -door_width - 0.3],
            [-1.54, -door_width - 0.4],
            [-1.55, -door_width - 0.5],
        ]
    )
    distances = ((position_xy - obstacle_locations) ** 2).sum(
        axis=-1
    ) - obstacle_radius ** 2
    distances = jnp.clip(distances, a_min=0.005)
    V = V + obstacle_factor / jnp.min(distances)

    return V


@jax.jit
def navigate(state_estimate: jnp.ndarray, control_gains: jnp.ndarray) -> jnp.ndarray:
    """Compute the control input for the current state.

    Tries to follow the negative gradient of the navigation function.

    args:
        state_estimate: the (3,) array containing the current state estimate
        control_gains: (2,) array containing control gains. [0.5, 0.1] is a sensible
                       default
    returns:
        control_input: the (2,) array containing the actions for the dubins vehicle
    """
    # Get the gradient of the navigation function at the current point
    position = state_estimate[:2]
    grad_V = jax.grad(navigation_function)(position)
    grad_V_norm = jnp.linalg.norm(grad_V)
    grad_V_norm = jnp.where(grad_V_norm < 1e-2, 1.0, grad_V_norm)

    # Get the unit vector in the direction of the robot's current heading
    theta = state_estimate[2]
    heading = jnp.array([jnp.cos(theta), jnp.sin(theta)])

    # Steer so that the heading vector aligns with -grad_V. We can do this with P
    # control on the normalized cross product between the heading and -grad_V
    kp = control_gains[0]
    w = kp * jnp.cross(heading, -grad_V) / grad_V_norm

    # Try to move forward at a constant velocity, but slow down if heading is not
    # aligned with -grad_V
    v_scale = control_gains[1]
    alignment = jnp.dot(heading, -grad_V) / grad_V_norm  # +1 when aligned, -1 otherwise
    v = jnp.clip(v_scale * jnp.clip(alignment, a_min=0.0), a_max=1.0)

    return jnp.array([v, w])


@jax.jit
def get_observations(
    state: jnp.ndarray,
    beacon_locations: jnp.ndarray,
    observation_noise: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get the observations, including an angle measurement and squared range to beacons.

    To allow randomness in the JAX paradigm, we need to pass in all randomness
    explicitly, so the observation_noise argument should contain a gaussian vector
    generated using the observation noise covariance and zero mean.

    args:
        state: the (3,) array containing the state at which to measure
        beacon_locations: the (n_beacons, 2) array containing the beacon locations
        observation_noise: the (1 + n_beacons) array of noise to be added to the
                           measurement.
    returns:
        a (1 + n_beacons) array containing the observations
    """
    # Get the range and compass measurements and combine them
    ranges = beacon_range_measurements(
        state[:2], beacon_locations, observation_noise[1:]
    )
    heading = compass_measurements(state[2], observation_noise[0])

    return jnp.concatenate((heading, ranges))


@jax.jit
def get_observations_jacobian(
    state: jnp.ndarray,
    beacon_locations: jnp.ndarray,
) -> jnp.ndarray:
    """
    Return the jacobian of the observation model

    args:
        state: the (3,) array containing the estimated state at which to linearize
        beacon_locations: the (n_beacons, 2) array containing the beacon locations
    """
    # Get the range measurements, which we need for linearizing
    n_beacons = beacon_locations.shape[0]
    ranges = beacon_range_measurements(
        state[:2], beacon_locations, jnp.zeros(n_beacons)
    )

    # Get both jacobians and combine them as a block-diagonal
    range_jac = beacon_range_measurement_lin(state[:2], beacon_locations, ranges)
    heading_jac = compass_measurement_lin(state[2])

    observations_jac = jnp.zeros((n_beacons + 1, 3))
    observations_jac = observations_jac.at[1:, :2].set(range_jac)
    observations_jac = observations_jac.at[0, 2].set(heading_jac[0, 0])

    return observations_jac


@jax.jit
def step(
    current_state_true: jnp.ndarray,
    current_state_estimate: jnp.ndarray,
    current_state_estimate_covariance: jnp.ndarray,
    control_gains: jnp.ndarray,
    beacon_locations: jnp.ndarray,
    observation_noise: jnp.ndarray,
    observation_noise_covariance: jnp.ndarray,
    actuation_noise: jnp.ndarray,
    actuation_noise_covariance: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate one discrete time step for the AGV system

    args:
        current_state_true: (3) array of current true state
        current_state_estimate: (3) array of current state estimate
        current_state_estimate_covariance: (3, 3) current state estimate covariance
        control_gains: a (2) array of controller gains
        beacon_locations: a (n_beacons, 2) array of beacon locations
        observation_noise: a (1 + n_beacons) matrix of observation noise values
        observation_noise_covariance: a (1 + n_beacons, 1 + n_beacons) covariance matrix
                                      for the observation noise.
        actuation_noise: a (3) matrix of actuation noise values
        actuation_noise_covariance: a (3, 3) covariance matrix for the actuation noise.
        dt: timestep

    returns:
        a tuple of
            - the new true state in a (3) array
            - the new state estimate in a (3) array
            - the new state estimate covariance in a (3, 3) matrix
    """
    # Get the control input based on the current state estimate
    control_input = navigate(current_state_estimate, control_gains)

    # Update the true state with some random actuation noise
    new_state_true = dubins_next_state(
        current_state_true, control_input, actuation_noise, dt
    )

    # Update the state estimate and covariance with the dynamics
    new_state_estimate = dubins_next_state(
        current_state_estimate,
        control_input,
        jnp.zeros(current_state_estimate.shape),
        dt,
    )
    dynamics_jac = dubins_linearized_dynamics(current_state_estimate, control_input, dt)
    new_state_estimate_covariance = dt_ekf_predict_covariance(
        current_state_estimate_covariance,
        dynamics_jac,
        actuation_noise_covariance,
        dt,
    )

    # Get the new observations with some random observation noise
    observations = get_observations(
        current_state_true,
        beacon_locations,
        observation_noise,
    )
    expected_observations = get_observations(
        new_state_estimate,
        beacon_locations,
        jnp.zeros_like(observation_noise),
    )

    # Use those observations to update the EKF estimate
    observations_jacobian = get_observations_jacobian(
        new_state_estimate, beacon_locations
    )
    new_state_estimate, new_state_estimate_covariance = dt_ekf_update(
        new_state_estimate,
        new_state_estimate_covariance,
        observations,
        expected_observations,
        observations_jacobian,
        observation_noise_covariance,
    )

    return new_state_true, new_state_estimate, new_state_estimate_covariance


def agv_simulate(
    design_params: jnp.ndarray,
    exogenous_sample: jnp.ndarray,
    observation_noise_covariance,
    actuation_noise_covariance,
    initial_state_mean,
    initial_state_covariance,
    time_steps: int,
    dt: float,
):
    """Simulate the performance of the EKF + navigation function system.

    To make this function pure, we need to pass in all sources of randomness used.

    args:
        design_params: a (2 + n_beacons * 2) array of design parameters. The first two
                       entries are the controller gains and the rest are the beacon
                       locations when reshaped to (n_beacons, 2)
        exogenous_sample: (2 + T * (3 + n_beacons + 1)) array containing all randomness
                          used for the observation, actuation, and initial state noise.
                          Can be generated by AGVExogenousParameters.sample
        observation_noise_covariance: a (1 + n_beacons, 1 + n_beacons) covariance matrix
                                      for the observation noise.
        actuation_noise_covariance: a (3, 3) covariance matrix for the actuation noise.
        initial_state_mean: a (3,) array of the initial mean state
        initial_state_covariance: a (3, 3) covariance matrix for the initial state.
        time_steps: the number of steps to simulate
        dt: the duration of each time step

    returns:
        a tuple of
            - the true state trace in a (T, 3) matrix
            - the state estimate trace in a (T, 3) matrix
            - the state estimate covariance trace in a (T, 3, 3) matrix
            - the navigation function trace in a (T) matrix
    """
    # Extract design parameters
    control_gains = design_params[:2]
    beacon_locations = design_params[2:]
    n_beacons = beacon_locations.shape[0] // 2
    beacon_locations = beacon_locations.reshape(n_beacons, 2)

    # Extract the exogenous parameters
    initial_state = exogenous_sample[:3]
    observation_noises = exogenous_sample[3 : (3 + time_steps * (n_beacons + 1))]
    observation_noises = observation_noises.reshape(time_steps, 1 + n_beacons)
    actuation_noises = exogenous_sample[(3 + time_steps * (n_beacons + 1)) :]
    actuation_noises = actuation_noises.reshape(time_steps, 3)

    # Set up the matrices to store the simulation traces
    state_true = jnp.zeros((time_steps, 3))
    state_estimate_mean = jnp.zeros((time_steps, 3))
    state_estimate_covariance = jnp.zeros((time_steps, 3, 3))
    true_navigation_function = jnp.zeros(time_steps)

    # Add the initial values
    state_true = state_true.at[0, :].set(initial_state)
    state_estimate_mean = state_estimate_mean.at[0, :].set(initial_state_mean)
    state_estimate_covariance = state_estimate_covariance.at[0, :, :].set(
        initial_state_covariance
    )
    true_navigation_function = true_navigation_function.at[0].set(
        navigation_function(initial_state[:2])
    )

    # Simulate forward
    for t in range(time_steps - 1):
        current_state_true = state_true[t]
        current_state_estimate = state_estimate_mean[t]
        current_state_estimate_covariance = state_estimate_covariance[t]

        new_state_true, new_state_estimate, new_state_estimate_covariance = step(
            current_state_true,
            current_state_estimate,
            current_state_estimate_covariance,
            control_gains,
            beacon_locations,
            observation_noises[t],
            observation_noise_covariance,
            actuation_noises[t],
            actuation_noise_covariance,
            dt,
        )

        # Save the new state, estimate, and navigation function value
        state_true = state_true.at[t + 1].set(new_state_true)
        state_estimate_mean = state_estimate_mean.at[t + 1].set(new_state_estimate)
        state_estimate_covariance = state_estimate_covariance.at[t + 1].set(
            new_state_estimate_covariance
        )
        V = navigation_function(new_state_true[:2])
        true_navigation_function = true_navigation_function.at[t + 1].set(V)

    # Return the state and estimate traces
    return (
        state_true,
        state_estimate_mean,
        state_estimate_covariance,
        true_navigation_function,
    )


if __name__ == "__main__":
    # Plot the navigation function
    x = jnp.linspace(-3, 0.1, 100)
    # y = jnp.linspace(-1.1, 1.1, 100)
    y = jnp.linspace(-1.1, 0.75, 100)

    X, Y = jnp.meshgrid(x, y)
    XY = jnp.stack((X, Y)).reshape(2, 10000).T

    V = jax.vmap(navigation_function, in_axes=0)(XY).reshape(100, 100)

    # Test out the simulation
    T = 30
    dt = 0.5
    time_steps = int(T / dt)

    # Define the exogenous parameters
    observation_noise_covariance = jnp.diag(jnp.array([0.1, 0.01, 0.01]))
    actuation_noise_covariance = dt ** 2 * jnp.diag(jnp.array([0.001, 0.001, 0.01]))
    initial_state_mean = jnp.array([-2.2, 0.5, 0.0])
    initial_state_covariance = 0.001 * jnp.eye(3)
    ep = AGVExogenousParameters(
        time_steps,
        initial_state_mean,
        initial_state_covariance,
        actuation_noise_covariance,
        observation_noise_covariance,
    )

    # Define the design parameters

    # # Initial
    # beacon_locations = jnp.array([[-2.0, 0.0], [-0.1, 0.0]])
    # control_gains = jnp.array([0.5, 0.1])

    # Optimized
    beacon_locations = jnp.array([-1.6945883, -1.0, 0.0, -0.8280163]).reshape(2, 2)
    control_gains = jnp.array([2.535058, 0.09306894])

    dp = DesignParameters(control_gains.size + beacon_locations.size)
    dp.set_values(jnp.concatenate((control_gains, beacon_locations.reshape(-1))))

    # Sample some exogenous parameters
    prng_key = jax.random.PRNGKey(0)
    prng_key, subkey = jax.random.split(prng_key)
    exogenous_sample = ep.sample(subkey)

    true_states, state_estimates, state_estimate_covariances, V_trace = agv_simulate(
        dp.get_values(),
        exogenous_sample,
        observation_noise_covariance,
        actuation_noise_covariance,
        initial_state_mean,
        initial_state_covariance,
        time_steps,
        dt,
    )

    # Plot overlaid on the navigation function, with beacon locations shown as
    # red triangles
    # fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # ax = axs[0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    contours = ax.contourf(X, Y, V, levels=10)
    cbar = plt.colorbar(contours, location="top")
    cbar.set_label("V", rotation="horizontal", fontsize=15)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.plot(
        beacon_locations[:, 0],
        beacon_locations[:, 1],
        "r^",
        label="Beacons",
        markersize=20,
    )
    ax.plot(
        true_states[:, 0],
        true_states[:, 1],
        "silver",
        marker="x",
        label="True trajectory",
        markersize=10,
    )
    ax.plot(
        state_estimates[:, 0],
        state_estimates[:, 1],
        "cyan",
        marker="x",
        label="Estimated trajectory",
        markersize=10,
    )
    ax.legend(prop={"size": 24})

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(15)

    ax.set_aspect("equal")
    # fig.tight_layout()

    # # Plot the true navigation function value over time
    # ax = axs[1]
    # ax.plot(jnp.arange(0, T, dt), V_trace)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Navigation function value")

    # # Plot the estimation error over time
    # ax = axs[2]
    # ax.plot(
    #     jnp.arange(0, T, dt), jnp.linalg.norm(true_states - state_estimates, axis=-1)
    # )
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel(r"$||x - \hat{x}||$")
    plt.show()
