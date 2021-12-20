"""Define some type for simulator and cost function callables"""
from typing import Callable, Tuple

import jax.numpy as jnp


"""
The simulator needs to take specific values for design and exogenous parameters (as
JAX arrays) and returns one or more JAX arrays with the results of running the
simulation. These JAX arrays should have time as the first dimension
"""
Simulator = Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, ...]]

"""
The cost function takes the output of the simulator and returns a scalar value (in a
1-element JAX array).
"""
CostFunction = Callable[[Tuple[jnp.ndarray, ...]], jnp.ndarray]
