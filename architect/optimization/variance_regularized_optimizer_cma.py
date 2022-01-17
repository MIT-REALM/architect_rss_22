"""
Optimizes a design to achieve minimal cost, regularized by the variance of the cost
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import numpy as np
import nevergrad as ng

from architect.design import DesignProblem


class VarianceRegularizedOptimizerCMA(object):
    """
    VarianceRegularizedOptimizerCMA implements (as the name implies) variance-
    regularized design optimization, where we seek to minimize the cost metric plus an
    added regularization term proportional to the variance in the cost metric.

    The CMA suffix indicates that this optimizer relies on covariant matrix adaptation,
    a derivative-free algorithm.

    Formally: given a cost function J(theta, phi) (a function of design and exogenous
    parameters), solve

        min_theta E_phi [J(theta, phi)] + variance_weight * var_phi [J(theta, phi)]
        s.t. constraints on theta (specified in the DesignParameters property of the
             DesignProblem)

    The expectation and variance are estimated over a sample of size sample_size
    """

    def __init__(
        self, design_problem: DesignProblem, variance_weight: float, sample_size: int
    ):
        """Initialize the optimizer.

        args:
            design_problem: the design problem we seek to optimize
            variance_weight: the weight used to penalize the cost variance in the
                             objective function
            sample_size: the number of points used to estimate the mean and variance
        """
        super(VarianceRegularizedOptimizerCMA, self).__init__()
        self.design_problem = design_problem
        self.variance_weight = variance_weight
        self.sample_size = sample_size

    def compile_cost_fn(
        self, prng_key: PRNGKeyArray
    ) -> Tuple[
        Callable[[np.ndarray], float],
        Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    ]:
        """Compile the variance-regularized cost function. This involves:
            1. Vectorize w.r.t. exogenous parameters to enable efficient computation of
               sample mean and variance.
            2. Wrap cost with mean and variance.
            3. Wrap with numpy arrays

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        returns: two functions in a Tuple
            - a function that takes an array of values for the design parameters and
              returns the variance-regularized cost.
            - a function that takes a JAX array of values for the design parameters
              and returns a tuple of sample mean and variance of the cost
        """
        # Wrap the cost function
        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            return self.design_problem.cost_fn(design_params, exogenous_params)

        # Vectorize the cost function with respect to the exogenous parameters
        # None indicates that we do not vectorize wrt design parameters
        # 0 indicates that we add a batch dimension as the 0-th dimension of the
        # exogenous parameters
        costv = jax.vmap(cost, (None, 0))

        # Define the variance-regularized cost as a function of design parameters
        def cost_mean_and_variance(
            design_params: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Sample a number of exogenous parameters. This will generate the same
            # sample on each call, since we use the same prng_key
            exogenous_sample = self.design_problem.exogenous_params.sample(
                prng_key, self.sample_size
            )

            # Evaluate the cost on this sample
            sample_cost = costv(design_params, exogenous_sample)

            # Compute the mean and variance of the sample cost
            sample_mean = sample_cost.mean()
            sample_variance = sample_cost.var(ddof=1)

            return sample_mean, sample_variance

        def variance_regularized_cost(design_params: jnp.ndarray) -> jnp.ndarray:
            sample_mean, sample_variance = cost_mean_and_variance(design_params)

            # Return the variance-regularized cost
            return sample_mean + self.variance_weight * sample_variance

        # Wrap in numpy access
        def vr_cost_np(design_params_np: np.ndarray):
            vr_cost = variance_regularized_cost(jnp.array(design_params_np))

            return vr_cost.item()

        # Return the needed functions
        return vr_cost_np, cost_mean_and_variance

    def optimize(
        self,
        prng_key: PRNGKeyArray,
        budget: int = 100,
        verbosity: int = 0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Optimize the design problem. Does not return status because it is difficult
        to infer convergence from derivative-free methods

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            budget: the budget for CMA (maximum number of function evaluations)
            verbosity: print information about the optimization
                (0: None, 1: fitness values, 2: fitness values and recommendation)
        returns:
            a JAX array of the optimal values of the design parameters
            a single-element JAX array of the expected cost at the optimal parameters
            a single-element JAX array of the cost variance at the optimal parameters
        """
        # Compile the cost function
        f, cost_mean_and_variance = self.compile_cost_fn(prng_key)

        # Get the bounds on the design parameters
        bounds = self.design_problem.design_params.bounds
        # We need to place bounds on all variables to match the nevergrad syntax.
        # If a variable is unbounded, bound it to +/- 50 arbitrarily
        lower_bounds = np.zeros(self.design_problem.design_params.size)
        upper_bounds = np.zeros(self.design_problem.design_params.size)
        for idx, bound in enumerate(bounds):
            lower, upper = bound
            if lower is None:
                lower_bounds[idx] = -50
            else:
                lower_bounds[idx] = lower

            if upper is None:
                upper_bounds[idx] = 50
            else:
                upper_bounds[idx] = upper

        # This method does not support constraints
        constraints = self.design_problem.design_params.constraints
        if constraints:
            raise NotImplementedError(
                "CMA-based optimization does not support constraints"
            )

        # Minimize! Use the nevergrad interface for CMA
        design_params_ng = ng.p.Array(
            shape=(self.design_problem.design_params.size,)
        ).set_bounds(lower_bounds, upper_bounds)
        instrum = ng.p.Instrumentation(design_params_ng)
        optimizer = ng.optimizers.CMA(parametrization=instrum, budget=budget)
        recommendation = optimizer.minimize(f)

        # Extract the solution and get the cost and variance at that point
        design_params_opt = jnp.array(recommendation.args[0])
        opt_cost_mean, opt_cost_variance = cost_mean_and_variance(design_params_opt)

        return (
            design_params_opt,
            opt_cost_mean,
            opt_cost_variance,
        )
