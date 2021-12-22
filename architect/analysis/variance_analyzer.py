"""Variation in the exogenous parameters induces variance in the performance of the
design, which we quantify using this analysis"""
from typing import Tuple

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import DesignProblem


class VarianceAnalyzer(object):
    """VarianceAnalyzer conducts a statistical analysis of the variance in performance
    of a design induced by variance in the exogenous parameters"""

    def __init__(self, design_problem: DesignProblem, sample_size: int):
        """Initialize a VarianceAnalyzer.

        args:
            design_problem: the design problem we seek to analyze
            sample_size: the number of points used to estimate the mean and variance
        """
        super(VarianceAnalyzer, self).__init__()
        self.design_problem = design_problem
        self.sample_size = sample_size

    def analyze(self, prng_key: PRNGKeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the variance analysis

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        returns: a tuple
            a JAX array with 1 element containing the mean of the cost
            a JAX array with 1 element containing the variance of the cost
        """
        # Compose the simulator and cost function and vectorize
        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            simulation_trace = self.design_problem.simulator(
                design_params, exogenous_params
            )
            return self.design_problem.cost_fn(simulation_trace)

        costv = jax.vmap(cost, (None, 0))

        # Sample exogenous parameters
        exogenous_sample = self.design_problem.exogenous_params.sample(
            prng_key, self.sample_size
        )

        # Get the cost on this sample
        sample_cost = costv(
            self.design_problem.design_params.get_values(), exogenous_sample
        )

        # Get the mean and variance
        sample_cost_mean = sample_cost.mean()
        sample_cost_var = sample_cost.var(ddof=1)

        return sample_cost_mean, sample_cost_var
