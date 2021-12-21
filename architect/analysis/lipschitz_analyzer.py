from typing import Tuple
import warnings

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import numpy as np
import scipy.optimize as sciopt

from architect.design import DesignProblem


def gevd_neg_loglikelihood(params, x):
    """Returns the negative log likelihood of the GEVD with params = [mu, sigma, xi]
    on data x

    args:
        params: [mu, sigma, xi], corresponding to the location, scale, and shape
                parameters of the GEVD
        x: length-m JAX array of the data to fit
    """
    # Extract parameters
    mu, sigma, xi = params

    # Normalize input
    z = (x - mu) / sigma

    # Get number of data points
    sample_size = x.shape[0]

    # Compute the log likelihood based on 3.3.2 in Coles.
    # This proceeds in cases on xi; if xi is too small we need to use the Gumbel
    # equation
    if jnp.abs(xi) < 1e-5:
        ll = -sample_size * jnp.log(sigma) - z.sum() - jnp.exp(z).sum()
    else:
        # Otherwise, use the Frechet/Weibull equation. These distributions have bounded
        # support, so we first need to make sure we're in-bounds
        out_of_support = jnp.abs(jnp.minimum(1 + xi * z, 0))
        penalty = 1e3
        ll = -(penalty * out_of_support).sum()

        # Now, we can add the true negative log likelihood for any points that
        # are in bounds
        in_support = 1 + xi * z > 0
        ll -= sample_size * jnp.log(sigma)
        ll -= (1 + 1 / xi) * jnp.log(1 + xi * z)[in_support].sum()
        ll -= ((1 + xi * z) ** (-1 / xi))[in_support].sum()

    # Return the negative
    return -ll


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

    def analyze(self, prng_key: PRNGKeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the variance analysis

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        returns: a tuple
            a JAX array with 3 elements containing the [mu, sigma, xi] parameters of
                the GEVD fit to the maximum sensitivities
            a JAX array with 1 element containing the standard errors associated with
                the estimates of the parameters.
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

        # Get the maximum in each block
        block_maxes = slope.max(axis=-1)

        # Fit a generalized extreme value distribution to these data by minimizing the
        # negative log likelihood.
        negative_ll_fn = lambda params: gevd_neg_loglikelihood(params, block_maxes)
        cost_and_grad = jax.value_and_grad(negative_ll_fn)

        def cost_and_grad_np(params_np):
            cost, grad = cost_and_grad(jnp.array(params_np))
            print("params")
            print(params_np)
            print(f"cost: {cost}")
            return cost.item(), np.array(grad, dtype=np.float64)

        bounds = (
            (-np.inf, np.inf),  # no bounds on mu
            (1e-3, np.inf),  # sigma must be positive
            (-0.5, np.inf),  # MLE needs xi > -1.0 to be valid, and this is a pretty
            # reasonable assumption in practice according to Coles
        )
        initial_guess = np.array([1.0, 1.0, 0.1])
        result = sciopt.minimize(
            cost_and_grad_np,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
            options={"disp": True},
        )

        if not result.success:
            warnings.warn("GVED MLE estimation failure!" + result.message)

        # Extract the parameters, then construct the observed information matrix
        # so we can estimate the error in these parameters
        params = jnp.array(result.x)
        hessian = jax.hessian(negative_ll_fn)(params)
        param_standard_error = jnp.sqrt(jnp.diag(jnp.linalg.inv(hessian)))

        return params, param_standard_error
