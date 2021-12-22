"""Define some estimation algorithms, like EKF"""
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy

from ..dynamics.types import (
    DiscreteTimeDynamicsCallable,
    DiscreteTimeDynamicsJacobianCallable,
)


@jax.jit
def dt_ekf_predict(
    state_mean: jnp.ndarray,
    state_covariance: jnp.ndarray,
    control_input: jnp.ndarray,
    dynamics_fn: DiscreteTimeDynamicsCallable,
    dynamics_jac_fn: DiscreteTimeDynamicsJacobianCallable,
    actuation_noise_covariance: jnp.ndarray,
    dt,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the one-step EKF prediction for discrete-time

    args:
        state_mean: the (n_states,) array containing the current estimate of state
        state_covariance: the (n_states, n_states) matrix containing the covariance of
                          the current state estimate error
        control_input: the (n_controls,) array containing the control input at this step
        dynamics_fn: a function taking current state, control input, noise, and a
                     timestep and returning the next state
        actuation_noise_covariance: the (n_states, n_states) matrix containing the
                                    covariance of the state update noise
        dt: the length of the discrete-time update (scalar)
    returns:
        a tuple of the new mean state estimate and error covariance
    """
    # Update the mean (dynamics with zero noise)
    zero_noise = jnp.zeros(state_mean.shape)
    next_state_mean = dynamics_fn(state_mean, control_input, zero_noise, dt)

    # Update the covariance
    F = dynamics_jac_fn(state_mean, control_input, dt)
    next_state_covariance = (
        F @ state_covariance @ jnp.transpose(F) + actuation_noise_covariance
    )

    return next_state_mean, next_state_covariance


@jax.jit
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
