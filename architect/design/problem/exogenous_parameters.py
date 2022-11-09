"""Exogenous parameters are anything "uncontrollable" that affect the design; these are
what we consider robustness against and are typically drawn from some distribution
"""
from typing import Optional

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray


class ExogenousParameters(object):
    """ExogenousParameters represents a vector of parameters over which the designer has
    *no* control: they represent disturbances from the environment, uncertainties in
    sensing or actuation, etc.. The design task involves ensuring robustness to
    variation in these parameters, and the analysis tasks involves measuring sensitivity
    to changes in these parameters.

    You can think of this class as defining a distribution over these parameters.

    Implemented as a generic vector of m parameters drawn independently from standard
    normal distributions with zero mean and variance 1. If a specific distribution (or
    combination of distributions) is needed for your design problem, you should make a
    custom subclass.
    """

    def __init__(
        self,
        size: int,
        names: Optional[list[str]] = None,
    ):
        """
        Initialize the ExogenousParameters object.

        args:
            size: the number of design variables
            names: a list of names for variables. If not provided, defaults to
                   "phi_0", "phi_1", ...
        """
        super(ExogenousParameters, self).__init__()
        self.size = size

        # Specify default behavior
        if names is None:
            names = [f"phi_{i}" for i in range(self.size)]
        self.names = names

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

        # Default behavior is sampling from standard normals; subclasses can do more
        # interesting things
        return jax.random.normal(prng_key, shape=shape)
