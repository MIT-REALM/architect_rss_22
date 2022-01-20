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


class WorstCaseCostAnalyzer(object):
    """
    WorstCaseCostAnalyzer conducts a statistical analysis of the design's worst case
    performance
    """

    def __init__(
        self, design_problem: DesignProblem, sample_size: int, block_size: int
    ):
        """Initialize a WorstCaseCostAnalyzer.

        args:
            design_problem: the design problem we seek to analyze
            sample_size: the number of blocks used to estimate the distribution of
                         the maximum cost. Larger sample sizes will decrease
                         the variance of the estimate.
            block_size: the number of samples used to compute the max in each block.
                        Larger block sizes will decrease the variance of the estimate,
                        but will be more expensive to compute.
        """
        super(WorstCaseCostAnalyzer, self).__init__()
        self.design_problem = design_problem
        self.sample_size = sample_size
        self.block_size = block_size

    def analyze(
        self, prng_key: PRNGKeyArray
    ) -> Tuple[pd.DataFrame, az.data.inference_data.InferenceData]:
        """Conduct the analysis

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        returns: a tuple
            - a pandas DataFrame containing summary statistics for fitting a GEVD
              to the observed sensitivities
            - an arviz InferenceData object including the raw results of the inference
        """
        # Wrap the cost function and vectorize
        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            return self.design_problem.cost_fn(design_params, exogenous_params)

        costv = jax.vmap(cost, (None, 0))

        # Construct a dataset of the maximum sensitivities for sample_size blocks,
        # each of size block_size.

        # This starts by sampling two sets of exogenous parameters
        prng_key, prng_subkey = jax.random.split(prng_key)
        exogenous_sample = self.design_problem.exogenous_params.sample(
            prng_subkey, self.sample_size * self.block_size
        )

        # Get the costs on each sample
        print(f"Design params: {self.design_problem.design_params.get_values()}")
        sample_cost = costv(
            self.design_problem.design_params.get_values(), exogenous_sample
        )
        # Reshape to be (sample_size, block_size) instead of (sample_size * block_size)
        sample_cost = sample_cost.reshape(self.sample_size, self.block_size)

        # Get the maximum in each block
        block_maxes = sample_cost.max(axis=-1)

        total_num_samples = self.sample_size * self.block_size
        print(f"Max observed cost over {total_num_samples}: {block_maxes.max()}")

        # Fit a generalized extreme value distribution to these data using PyMC3
        # (a Bayesian analysis using MCMC for inference).
        # Source for snippet:
        # https://discourse.pymc.io/t/generalized-extreme-value-analysis-in-pymc3/7433
        with pm.Model() as model_gevd:  # noqa: F841
            mu = pm.Normal("mu", mu=0, sigma=10)
            sigma = pm.Exponential("sigma", lam=1.0)
            # sigma = pm.TruncatedNormal("sigma", mu=1, sigma=10, lower=0.0)
            xi = pm.Normal("xi", mu=0, sigma=10)

            def logp(value):
                scaled = (value - mu) / sigma

                logp_expression = tt.switch(
                    tt.isclose(xi, 0.0),
                    -tt.log(sigma) - scaled - tt.exp(-scaled),
                    -tt.log(sigma)
                    - ((xi + 1) / xi) * tt.log1p(xi * scaled)
                    - tt.pow(1 + xi * scaled, -1 / xi),
                )
                return bound(
                    logp_expression,
                    1 + xi * (value - mu) / sigma > 0,
                    sigma > 0,
                )

            # Make some guesses for starting parameters
            mu_start = block_maxes.mean()
            sigma_start = block_maxes.std()

            gevd = pm.DensityDist("gevd", logp, observed=block_maxes)  # noqa: F841
            step = pm.NUTS()
            idata = pm.sample(
                10000,
                chains=4,
                tune=3000,
                step=step,
                start={"mu": mu_start, "sigma": sigma_start, "xi": -0.1},
                return_inferencedata=True,
                progressbar=True,
            )

        return az.summary(idata), idata
