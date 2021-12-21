from typing import Tuple

import arviz as az
import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import pandas as pd
import pymc3 as pm
from pymc3.distributions.dist_math import bound
import theano.tensor as tt

from architect.design import DesignProblem


class LipschitzAnalyzer(object):
    """LipschitzAnalyzer conducts a statistical analysis of the design's sensitivity to
    variantion in the exogenous parameters. This is related to estimating the Lipschitz
    constant, but does not require that the performance be Lipschitz (the analysis will
    tell us whether it is likely Lipschitz or not)."""

    def __init__(
        self, design_problem: DesignProblem, sample_size: int, block_size: int
    ):
        """Initialize a LipschitzAnalyzer.

        args:
            design_problem: the design problem we seek to analyze
            sample_size: the number of blocks used to estimate the distribution of
                         the maximum sensitivity. Larger sample sizes will decrease
                         the variance of the estimate.
            block_size: the number of samples used to compute the max in each block.
                        Larger block sizes will decrease the variance of the estimate,
                        but will be more expensive to compute.
        """
        super(LipschitzAnalyzer, self).__init__()
        self.design_problem = design_problem
        self.sample_size = sample_size
        self.block_size = block_size

    def analyze(
        self, prng_key: PRNGKeyArray
    ) -> Tuple[pd.DataFrame, az.data.inference_data.InferenceData]:
        """Conduct the variance analysis

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        returns: a tuple
            - a pandas DataFrame containing summary statistics for fitting a GEVD
              to the observed sensitivities
            - an arviz InferenceData object including the raw results of the inference
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

        # Construct a dataset of the maximum sensitivities for sample_size blocks,
        # each of size block_size.

        # This starts by sampling two sets of exogenous parameters
        prng_key, prng_subkey = jax.random.split(prng_key)
        exogenous_sample_1 = self.design_problem.exogenous_params.sample(
            prng_subkey, self.sample_size * self.block_size
        )
        prng_key, prng_subkey = jax.random.split(prng_key)
        exogenous_sample_2 = self.design_problem.exogenous_params.sample(
            prng_subkey, self.sample_size * self.block_size
        )

        # Get the costs on each sample
        sample_1_cost = costv(
            self.design_problem.design_params.get_values(), exogenous_sample_1
        )
        sample_2_cost = costv(
            self.design_problem.design_params.get_values(), exogenous_sample_2
        )

        # Get the change in cost for each pair
        abs_cost_diff = jnp.abs(sample_1_cost - sample_2_cost)

        # And compute the slope
        abs_param_diff = jnp.linalg.norm(
            exogenous_sample_1 - exogenous_sample_2, axis=-1
        )
        slope = abs_cost_diff / abs_param_diff

        # slope should now be (sample_size * block_size), reshape
        slope = slope.reshape(self.sample_size, self.block_size)

        # Add some noise to allow the MAP to work. This should not change the estimated
        # maximum
        slope -= jax.random.truncated_normal(prng_key, 0.0, 0.1, shape=slope.shape)

        # Get the maximum in each block
        block_maxes = slope.max(axis=-1)

        # Fit a generalized extreme value distribution to these data using PyMC3
        # (a Bayesian analysis using MCMC for inference).
        # Source for snippet:
        # https://discourse.pymc.io/t/generalized-extreme-value-analysis-in-pymc3/7433
        with pm.Model() as model_gevd:  # noqa: F841
            mu = pm.Normal("mu", mu=3, sigma=10)
            sigma = pm.Normal("sigma", mu=0.24, sigma=10)
            xi = pm.Normal("xi", mu=0, sigma=10)

            def gev_logp(value):
                scaled = (value - mu) / sigma
                logp = -(
                    tt.log(sigma)
                    + ((xi + 1) / xi) * tt.log1p(xi * scaled)
                    + (1 + xi * scaled) ** (-1 / xi)
                )
                alpha = mu - sigma / xi
                bounds = tt.switch(xi > 0, xi > alpha, xi < alpha)
                return bound(logp, bounds, xi != 0)

            gevd = pm.DensityDist("gevd", gev_logp, observed=block_maxes)  # noqa: F841
            step = pm.NUTS()
            idata = pm.sample(
                10000,
                chains=4,
                tune=3000,
                step=step,
                start={"mu": 1.0, "sigma": 1.0, "xi": -0.1},
                return_inferencedata=True,
                progressbar=True,
            )

        return az.summary(idata), idata
