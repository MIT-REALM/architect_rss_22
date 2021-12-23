from typing import Optional

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import ExogenousParameters


class AGVExogenousParameters(ExogenousParameters):
    """ExogenousParameters for the AGV localization task"""

    def __init__(
        self,
        time_steps: int,
        initial_state_mean: jnp.ndarray,
        initial_state_covariance: jnp.ndarray,
        actuation_noise_covariance: jnp.ndarray,
        observation_noise_covariance: jnp.ndarray,
    ):
        """
        Initialize the exogenous parameters for the AGV localization porblem.

        args:
            time_steps: the number of discrete timesteps
            initial_state_mean: a (3,) array of the initial mean state
            initial_state_covariance: a (3, 3) covariance matrix for the initial state.
            actuation_noise_covariance: (3, 3) covariance matrix for the actuation noise
            observation_noise_covariance: (1 + n_beacons, 1 + n_beacons) covariance
                                          matrix for the observation noise.
        """
        # 3 for initial state, 3 for actuation noise at each timestep, and some for
        # the observations
        self.n_beacons = observation_noise_covariance.shape[0] - 1
        n_vars = 3 + time_steps * (3 + self.n_beacons + 1)
        names = ["initial_x", "initial_y"]
        for t in range(time_steps):
            names += [
                f"actuation_noise_{t}_x",
                f"actuation_noise_{t}_y",
                f"actuation_noise_{t}_theta",
            ]
            names += [f"observation_noise_{t}_theta"]
            names += [
                f"observation_noise_{t}_beacon_{i}" for i in range(self.n_beacons)
            ]

        self.time_steps = time_steps
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.actuation_noise_covariance = actuation_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance

        super(AGVExogenousParameters, self).__init__(n_vars, names)

    def sample(
        self, prng_key: PRNGKeyArray, batch_size: Optional[int] = None
    ) -> jnp.ndarray:
        """Sample values for these exogenous parameters from this distribution.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
                      This method will not split the key, it will be consumed.
            batch_size: if None (default), return a 1D JAX array with self.size
                        elements; otherwise, return a 2D JAX array with size
                        (batch_size, self.size)
        """
        if batch_size is None:
            batch_size = 1

        sampled_phi = jnp.zeros((batch_size, self.size))

        # Sample initial state
        prng_key, subkey = jax.random.split(prng_key)
        initial_state = jax.random.multivariate_normal(
            subkey,
            self.initial_state_mean,
            self.initial_state_covariance,
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 0:3].set(initial_state)

        # Sample observation noise
        prng_key, subkey = jax.random.split(prng_key)
        observation_noise = jax.random.multivariate_normal(
            subkey,
            jnp.zeros(self.observation_noise_covariance.shape[0]),
            self.observation_noise_covariance,
            shape=(batch_size, self.time_steps),
        )
        sampled_phi = sampled_phi.at[
            :, 3 : (3 + self.time_steps * (self.n_beacons + 1))
        ].set(observation_noise.reshape(batch_size, -1))

        # Sample actuation noise
        prng_key, subkey = jax.random.split(prng_key)
        actuation_noise = jax.random.multivariate_normal(
            subkey,
            jnp.zeros(self.actuation_noise_covariance.shape[0]),
            self.actuation_noise_covariance,
            shape=(batch_size, self.time_steps),
        )
        sampled_phi = sampled_phi.at[
            :, (3 + self.time_steps * (self.n_beacons + 1)) :
        ].set(actuation_noise.reshape(batch_size, -1))

        # Remove batch dimension if we only have one batch
        if batch_size == 1:
            sampled_phi = sampled_phi.reshape(-1)

        return sampled_phi
