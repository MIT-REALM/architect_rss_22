"""Define useful types for dynamics callables"""
from typing import Callable

import jax.numpy as jnp


# Dynamics should take current state, control input, noise, and timestep
DiscreteTimeDynamicsCallable = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray
]

# The dynamics jacobian has a similar signature, but no noise
DiscreteTimeDynamicsJacobianCallable = Callable[
    [jnp.ndarray, jnp.ndarray, float], jnp.ndarray
]
