"""Define some estimation algorithms, like EKF"""
from typing import Tuple

import jax
import jax.numpy as jnp


@jax.jit
def dt_ekf_predict_covariance(
    state_covariance: jnp.ndarray,
    dynamics_jac: jnp.ndarray,
    actuation_noise_covariance: jnp.ndarray,
    dt,
) -> jnp.ndarray:
    """Compute the one-step EKF prediction for discrete-time to update the state estimate
    covariance. Does not update the estimate itself.

    args:
        state_covariance: the (n_states, n_states) matrix containing the covariance of
                          the current state estimate error
        dynamics_jac: the (n_states, n_states) jacobian matrix for the dynamics
        actuation_noise_covariance: the (n_states, n_states) matrix containing the
                                    covariance of the state update noise
        dt: the length of the discrete-time update (scalar)
    returns:
        the new state error covariance
    """
    # Update the covariance
    F = dynamics_jac
    next_state_covariance = (
        F @ state_covariance @ jnp.transpose(F) + actuation_noise_covariance
    )

    return next_state_covariance


# @jax.jit
def dt_ekf_update(
    state_mean: jnp.ndarray,
    state_covariance: jnp.ndarray,
    observations: jnp.ndarray,
    expected_observations: jnp.ndarray,
    observations_jacobian: jnp.ndarray,
    observation_noise_covariance: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the discrete-time one-step EKF update.

    args:
        state_mean: the (n_states,) array containing the current estimate of state
        state_covariance: the (n_states, n_states) matrix containing the covariance of
                          the current state estimate error
        observations: the (n_obs,) array containing the measurements
        expected_observations: the (n_obs,) array containing the measurements expected
                               at the current state mean
        observations_jacobian: the (n_obs, n_states) array containing jacobian of the
                               measurements with respect to the states
        observation_noise_covariance: the (n_obs, n_obs) matrix
                                      containing the covariance of the observation noise
    returns:
        a tuple of the new mean state estimate and error covariance
    """
    # Compute measurement residual and residual covariance
    n_states = state_mean.shape[0]
    residual = observations - expected_observations
    residual_covariance = (
        observations_jacobian @ state_covariance @ jnp.transpose(observations_jacobian)
        + observation_noise_covariance
    )

    # Get the EKF Kalman gain
    K = (
        state_covariance
        @ jnp.transpose(observations_jacobian)
        @ jnp.linalg.inv(residual_covariance)
    )

    # Update the state estimate and covariance
    new_state_mean = state_mean + K @ residual
    new_state_covariance = (
        jnp.eye(n_states) - K @ observations_jacobian
    ) @ state_covariance

    return new_state_mean, new_state_covariance
