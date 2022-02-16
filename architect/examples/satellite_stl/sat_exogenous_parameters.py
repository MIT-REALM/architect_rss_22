from typing import Optional

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import ExogenousParameters


class SatExogenousParameters(ExogenousParameters):
    """ExogenousParameters for the multi-agent manipulation task"""

    def __init__(self):
        """
        Initialize the exogenous parameters for the satellite STL problem.
        """
        n_vars = 6
        names = ["px0", "py0", "pz0", "vx0", "vy0", "vz0"]

        super(SatExogenousParameters, self).__init__(n_vars, names)

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
        # Handle default if no batch size is given
        shape: tuple[int, ...] = (self.size,)
        if batch_size is not None:
            shape = (batch_size, self.size)

        # Initial states are sampled from a normal distribution with mean position
        # [15, 15, 0] and zero velocity
        mean_state = jnp.zeros((6,)).at[:2].add(15.0)

        # Use standard deviation of 2
        return mean_state + 2 * jax.random.normal(prng_key, shape=shape)
