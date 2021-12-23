"""Define some useful sensor models"""
import jax
import jax.numpy as jnp


@jax.jit
def beacon_range_measurements(
    measurement_xy: jnp.ndarray,
    beacon_locations_xy: jnp.ndarray,
    observation_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Measure the squared range to each beacon.

    We need to pass in all randomness explicitly as an exogenous parameter (here, this
    is observation_noise).

    args:
        measurement_xy: the (2,) array containing the current (true) xy state
        beacon_locations_xy: the (n_beacons, 2) array containing the beacon locations
        observation_noise: the (n_beacons) array of noise to be added to the measurement
    returns:
        a (n_beacons,) array containing the observations
    """
    # Compute the squared range (this is analogous to measuring e.g. the decay in a
    # radio or acoustic signal).
    ranges = ((measurement_xy - beacon_locations_xy) ** 2).sum(axis=-1)

    # Add the observation noise
    observations = ranges + observation_noise

    return observations


@jax.jit
def beacon_range_measurement_lin(
    estimated_measurement_xy: jnp.ndarray,
    beacon_locations_xy: jnp.ndarray,
    measurements: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the linearized observation matrix for the beacon range measurement.

    args:
        estimated_measurement_xy: the (2,) array containing the current (estimated)
                                  xy state
        beacon_locations_xy: the (n_beacons, 2) array containing the beacon locations
        measurements: the (n_beacons,) array of measurements about which to linearize
    returns:
        the (n_beacons, 2) matrix linearizing the relationship between the measurement
        and the xy state of the robot
    """
    n_beacons = measurements.shape[0]

    # The range measurements have derivatives:
    #
    #     dz/dx = 2 * z * (x - x_bi)
    #     dz/dy = 2 * z * (y - y_bi)
    #
    # where (x_bi, y_bi) is the location of the i-th beacon
    xy_diff = estimated_measurement_xy - beacon_locations_xy
    H = 2 * measurements.reshape(n_beacons, 1) * xy_diff

    return H
