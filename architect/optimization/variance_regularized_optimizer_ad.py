"""
Optimizes a design to achieve minimal cost, regularized by the variance of the cost
"""
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import numpy as np
import scipy.optimize as sciopt

from architect.design import DesignProblem


class VarianceRegularizedOptimizerAD(object):
    """
    VarianceRegularizedOptimizerAD implements (as the name implies) variance-
    regularized design optimization, where we seek to minimize the cost metric plus an
    added regularization term proportional to the variance in the cost metric.

    The AD suffix indicates that this optimizer relies on automatic differentiation.

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
        super(VarianceRegularizedOptimizerAD, self).__init__()
        self.design_problem = design_problem
        self.variance_weight = variance_weight
        self.sample_size = sample_size

    def compile_cost_fn(
        self, prng_key: PRNGKeyArray, jit: bool = False
    ) -> Tuple[
        Callable[[np.ndarray], Tuple[float, np.ndarray]],
        Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    ]:
        """Compile the variance-regularized cost function. This involves:
            1. Vectorize w.r.t. exogenous parameters to enable efficient computation of
               sample mean and variance.
            2. Wrap cost with mean and variance.
            3. Automatically differentiate
            4. Wrap with numpy arrays

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            jit: if True, jit the cost and gradient function
        returns: two functions in a Tuple
            - a function that takes an array of values for the design parameters and
              returns a tuple of variance-regularized cost and the gradient of that
              cost w.r.t. the design parameters.
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

        # Automatically differentiate
        vr_cost_and_grad = jax.value_and_grad(variance_regularized_cost)

        if jit:
            vr_cost_and_grad = jax.jit(vr_cost_and_grad)

        # Wrap in numpy access
        def vr_cost_and_grad_np(design_params_np: np.ndarray):
            vr_cost, grad = vr_cost_and_grad(jnp.array(design_params_np))

            return vr_cost.item(), np.array(grad, dtype=np.float64)

        # Return the needed functions
        return vr_cost_and_grad_np, cost_mean_and_variance

    def optimize(
        self,
        prng_key: PRNGKeyArray,
        disp: bool = False,
        maxiter: Optional[int] = None,
        jit: bool = False,
    ) -> Tuple[bool, str, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Optimize the design problem, starting with the initial values stored in
        self.design_problem.design_params.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            disp: if True, display optimization progress
            maxiter: maximum number of optimization iterations to perform. Defaults to
                no limit.
            jit: if True, JIT the cost and gradient function.
        returns:
            a boolean that is true if the optimization suceeded,
            a string containing the termination message from the optimization engine,
            a JAX array of the optimal values of the design parameters
            a single-element JAX array of the expected cost at the optimal parameters
            a single-element JAX array of the cost variance at the optimal parameters
        """
        # Compile the cost function
        f, cost_mean_and_variance = self.compile_cost_fn(prng_key, jit=jit)

        # Get the bounds on the design parameters
        bounds = self.design_problem.design_params.bounds_list

        # Get the constraints
        constraints = self.design_problem.design_params.constraints

        # If there are no constraints, we can use the L-BFGS-B backend; otherwise, use
        # sequential least-squares
        if not constraints:
            method = "L-BFGS-B"
        else:
            method = "SLSQP"

        # Get the initial guess stored in the design parameters
        initial_guess = self.design_problem.design_params.get_values_np()

        # Minimize! Use the scipy interface
        opts: Dict[str, Any] = {"disp": disp}
        if maxiter is not None:
            opts["maxiter"] = maxiter

        result = sciopt.minimize(
            f,
            initial_guess,
            method=method,
            jac=True,  # f returns both cost and gradient in a tuple
            bounds=bounds,
            constraints=constraints,
            options=opts,
        )

        # Extract the solution and get the cost and variance at that point
        design_params_opt = jnp.array(result.x)
        opt_cost_mean, opt_cost_variance = cost_mean_and_variance(design_params_opt)

        return (
            result.success,
            result.message,
            design_params_opt,
            opt_cost_mean,
            opt_cost_variance,
        )
