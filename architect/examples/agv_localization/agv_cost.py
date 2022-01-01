import jax.numpy as jnp

from .agv_simulator import agv_simulate


def agv_cost(
    design_params: jnp.ndarray,
    exogenous_sample: jnp.ndarray,
    observation_noise_covariance,
    actuation_noise_covariance,
    initial_state_mean,
    initial_state_covariance,
    time_steps: int,
    dt: float,
) -> jnp.ndarray:
    """Compute the cost based on given beacon locations.

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
        scalar cost, given by the sum of the navigation function over time
    """
    true_states, state_estimates, state_estimate_covariances, V_trace = agv_simulate(
        design_params,
        exogenous_sample,
        observation_noise_covariance,
        actuation_noise_covariance,
        initial_state_mean,
        initial_state_covariance,
        time_steps,
        dt,
    )

    # Compute cost based:
    #   1) Estimation error
    #   2) LQR-style cost: mean squared states
    #   3) Navigation cost: mean navigation function value
    #   4) Collision penalty: max navigation function value
    cost = (
        100 * ((true_states - state_estimates) ** 2).mean()
        + (true_states ** 2).mean()
        + 0.1 * V_trace.mean()
        + 0.1 * V_trace.max()
    )
    return cost