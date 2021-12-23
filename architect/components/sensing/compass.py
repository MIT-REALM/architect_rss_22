"""Define some useful sensor models"""
import jax
import jax.numpy as jnp


@jax.jit
def compass_measurements(
    true_heading: jnp.ndarray,
    observation_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Measure the heading using a noisy compass measurement

    We need to pass in all randomness explicitly as an exogenous parameter (here, this
    is observation_noise).

    args:
        true_heading: the (1,) array containing the current (true) heading
        observation_noise: the (1,) array of noise to be added to the measurement
    returns:
        a (1,) array containing the observations
    """
    # Add the observation noise
    observations = true_heading + observation_noise

    return observations.reshape(1)


@jax.jit
def compass_measurement_lin(
    estimated_heading: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the linearized observation matrix for the compass measurement.

    args:
        estimated_heading: the (2,) array containing the current (estimated) heading
        measurements: the (n_beacons,) array of measurements about which to linearize
    returns:
        the (n_beacons, 2) matrix linearizing the relationship between the measurement
        and the heading of the robot
    """
    H = jnp.eye(1)
    return H
