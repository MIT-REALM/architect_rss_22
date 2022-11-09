"""Exogenous parameters are anything "uncontrollable" that affect the design; these are
what we consider robustness against and are typically drawn from some distribution
"""
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import numpy as np

from .exogenous_parameters import ExogenousParameters


class BoundedExogenousParameters(ExogenousParameters):
    """A subclass of ExogenousParameters that includes hyperrectangle bounds for each
    dimension.
    """

    def __init__(
        self,
        size: int,
        bounds: jnp.ndarray,
        names: Optional[list[str]] = None,
    ):
        """
        Initialize the ExogenousParameters object.

        args:
            size: the number of design variables
            bounds: a (size, 2) array of upper and lower bounds for each parameter.
            names: a list of names for variables. If not provided, defaults to
                   "phi_0", "phi_1", ...
        """
        super(BoundedExogenousParameters, self).__init__(size, names)
        self.bounds = bounds

        # Bounded exogenous parameters can have specific values
        self._values = jnp.zeros(self.size)
        for idx, bound in enumerate(self.bounds_list):
            lb, ub = bound
            if lb is not None and ub is not None:
                center = (lb + ub) / 2.0
            elif lb is not None:
                center = lb
            elif ub is not None:
                center = ub
            else:
                center = 0.0

            self._values = self._values.at[idx].set(center)

    def set_values(self, new_values: Union[jnp.ndarray, np.ndarray]):
        """Set the values of these design parameters using the given values.

        args:
            new_values: the array of new values
        """
        self._values = new_values

    def get_values(self) -> jnp.ndarray:
        """Return the values of these design parameters."""
        return self._values

    def get_values_np(self) -> np.ndarray:
        """Return the values of these design parameters."""
        return np.array(self._values)

    @property
    def bounds_list(self) -> Sequence[Tuple[float, float]]:
        """Returns the bounds on the design parameters as a list. Each element
        of the list should be a tuple of (lower, upper) bounds.
        """
        return [(lb.item(), ub.item()) for lb, ub in self.bounds]

    def sample(
        self, prng_key: PRNGKeyArray, batch_size: Optional[int] = None
    ) -> jnp.ndarray:
        """Sample values for these exogenous parameters uniformly from the bounded
        region

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
                      This method will not split the key, it will be consumed.
            batch_size: if None (default), return a 1D JAX array with self.size
                        elements; otherwise, return a 2D JAX array with size
                        (batch_size, self.size)
        """
        # Handle default if no batch size is given
        if batch_size is None:
            batch_size = 1
        shape: Tuple[int, ...] = (batch_size, self.size)

        # Sample uniformly on [0, 1), then re-scale and shift to satisfy bounds
        sample = jax.random.uniform(prng_key, shape=shape, minval=0, maxval=1)
        for dim_idx in range(self.size):
            lower, upper = self.bounds[dim_idx]
            spread = upper - lower
            sample = sample.at[:, dim_idx].set(sample[:, dim_idx] * spread + lower)

        # Squeeze to 1 dimension if batch_size is 1
        if batch_size == 1:
            sample = sample.reshape(-1)

        return sample
