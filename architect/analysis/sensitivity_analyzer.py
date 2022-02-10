from typing import Tuple

from tqdm import tqdm
import arviz as az
import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import pandas as pd
import pymc3 as pm
from pymc3.distributions.dist_math import bound
import theano.tensor as tt

from architect.design import DesignProblem


class SensitivityAnalyzer(object):
    """SensitivityAnalyzer conducts a statistical analysis of the design's sensitivity to
    variantion in the exogenous parameters. This is related to estimating the Lipschitz
    constant, but does not require that the performance be Lipschitz (the analysis will
    tell us whether it is likely Lipschitz or not)."""

    def __init__(
        self,
        design_problem: DesignProblem,
        sample_size: int,
        block_size: int,
        stride_length: int = 1,
    ):
        """Initialize a SensitivityAnalyzer.

        args:
            design_problem: the design problem we seek to analyze
            sample_size: the number of blocks used to estimate the distribution of
                         the maximum sensitivity. Larger sample sizes will decrease
                         the variance of the estimate.
            block_size: the number of samples used to compute the max in each block.
                        Larger block sizes will decrease the variance of the estimate,
                        but will be more expensive to compute.
            stride_length: batch this many samples at once
        """
        super(SensitivityAnalyzer, self).__init__()
        self.design_problem = design_problem
        self.sample_size = sample_size
        self.block_size = block_size
        self.stride_length = stride_length

    def analyze(
        self, prng_key: PRNGKeyArray, inject_noise: bool = False,
    ) -> Tuple[pd.DataFrame, az.data.inference_data.InferenceData]:
        """Conduct the sensitivity analysis

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            inject_noise: if True, a small amount of (strictly negative) noise will
                be added to the measured sensitivities. This can help in simple cases
                where the sensitivity is constant (and the corresponding extreme value
                distribution is degenerate).
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
        block_maxes = jnp.zeros(self.sample_size)
        print("Gathering samples...")
        progress_bar = tqdm(range(0, self.sample_size, self.stride_length))
        for i in progress_bar:
            # This starts by sampling two sets of exogenous parameters
            prng_key, prng_subkey = jax.random.split(prng_key)
            exogenous_sample_1 = self.design_problem.exogenous_params.sample(
                prng_subkey, self.stride_length * self.block_size
            )
            prng_key, prng_subkey = jax.random.split(prng_key)
            exogenous_sample_2 = self.design_problem.exogenous_params.sample(
                prng_subkey, self.stride_length * self.block_size
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

            # Slope should now have (self.stride_length * self.block_size) entries,
            # so reshape and save the maximum for each block
            assert slope.shape == (self.stride_length * self.block_size,)
            slope = slope.reshape(self.stride_length, self.block_size)

            if inject_noise:
                # Add some noise to make the observed sensitivities non-degenerate.
                # This should not change the estimated maximum b/c we truncate the noise
                slope -= jax.random.truncated_normal(
                    prng_key, 0.0, 0.1, shape=slope.shape
                )

            # Get the maximum in each block
            block_maxes = block_maxes.at[i : i + self.stride_length].set(
                slope.max(axis=-1)
            )

        # Fit a generalized extreme value distribution to these data using PyMC3
        # (a Bayesian analysis using MCMC for inference).
        # Source for snippet from PR#5085 on the PyMC3 GitHub
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
                    1 + xi * scaled > 0,
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
                tune=5000,
                step=step,
                start={"mu": mu_start, "sigma": sigma_start, "xi": -0.1},
                return_inferencedata=True,
                progressbar=True,
            )

        return az.summary(idata), idata
