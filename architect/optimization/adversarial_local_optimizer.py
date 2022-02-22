"""
Optimizes a design to achieve minimal cost, regularized by the variance of the cost
"""
import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import numpy as np
import scipy.optimize as sciopt

from architect.design import BoundedDesignProblem


class AdversarialLocalOptimizer(object):
    """
    AdversarialLocalOptimizer implements adversarial local optimization, where we seek
    to modify the design parameters to minimize the objective but modify the exogenous
    parameters to maximize the objective.

    This proceeds in two phases. In the first phase, the designer gets to choose
    design parameters to minimize the objective. They do this by regularizing
    sensitivity about some nominal phi

        min_theta J(theta, phi) + c * ||grad_phi J(theta, phi)||^2

    The adversary then responds by maximizing phi:

        max_phi J(theta, phi)

    Both players are subject to bounds on the given variables. These rounds can repeat
    up to a specified number of rounds, but the adversary always gets the last response.
    """

    def __init__(self, design_problem: BoundedDesignProblem):
        """Initialize the optimizer.

        args:
            design_problem: the design problem we seek to optimize
        """
        super(AdversarialLocalOptimizer, self).__init__()
        self.design_problem = design_problem

    def compile_cost_dp_single_phi(
        self, prng_key: PRNGKeyArray, jit: bool = True
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Compile the cost function for the design parameters (to minimize cost)

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            jit: if True, jit the cost and gradient function
        returns:
            - a function that takes parameter values and returns a tuple of cost and the
                gradient of that cost w.r.t. the design parameters.
        """
        # Wrap the cost function
        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            return self.design_problem.cost_fn(design_params, exogenous_params)

        # Differentiate wrt theta
        cost_and_grad = jax.value_and_grad(cost, argnums=(0))

        if jit:
            cost_and_grad = jax.jit(cost_and_grad)

        # Wrap in numpy access
        def cost_and_grad_np(
            design_params_np: np.ndarray, exogenous_params_np: np.ndarray
        ):
            cost, grad = cost_and_grad(
                jnp.array(design_params_np), jnp.array(exogenous_params_np)
            )

            return cost.item(), np.array(grad, dtype=np.float64)

        # Return the needed functions
        return cost_and_grad_np

    def compile_cost_dp_multi_phi(
        self, prng_key: PRNGKeyArray, n_phi: int, vw: float = 0.0, jit: bool = True
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Compile the cost function for the design parameters (to minimize cost)

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            n_phi: number of samples from phi to consider
            vw: variance weight to use
            jit: if True, jit the cost and gradient function
        returns:
            - a function that takes parameter values and returns a tuple of cost and the
                gradient of that cost w.r.t. the design parameters.
        """

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
                prng_key, n_phi
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
            return sample_mean + vw * sample_variance

        # Automatically differentiate
        vr_cost_and_grad = jax.value_and_grad(variance_regularized_cost)

        if jit:
            vr_cost_and_grad = jax.jit(vr_cost_and_grad)

        # Wrap in numpy access
        def vr_cost_and_grad_np(design_params_np: np.ndarray, ep_np: np.ndarray):
            # We don't use these values of the exogenous params, since we've already
            # sampled a bunch of them
            vr_cost, grad = vr_cost_and_grad(jnp.array(design_params_np))

            return vr_cost.item(), np.array(grad, dtype=np.float64)

        # Return the needed function
        return vr_cost_and_grad_np

    def compile_cost_dp_sens_single_phi(
        self, prng_key: PRNGKeyArray, sensitivity_weight: float, jit: bool = True
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Compile the cost function for the design parameters (to minimize cost)

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            sensitivity_weight: regularization weight for sensitivity to exogenous
                parameters
            jit: if True, jit the cost and gradient function
        returns:
            - a function that takes parameter values and returns a tuple of cost and the
                gradient of that cost w.r.t. the design parameters.
        """
        # Wrap the cost function
        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            return self.design_problem.cost_fn(design_params, exogenous_params)

        # Automatically differentiate wrt phi
        cost_and_grad_phi = jax.value_and_grad(cost, argnums=(1))

        # Make the sensitivity-regularized cost
        def sensitivity_regularized_cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            cost, grad_phi = cost_and_grad_phi(design_params, exogenous_params)
            return cost + (grad_phi ** 2).sum() * sensitivity_weight

        # Differentiate again wrt theta
        cost_and_grad = jax.value_and_grad(sensitivity_regularized_cost, argnums=(0))

        if jit:
            cost_and_grad = jax.jit(cost_and_grad)

        # Wrap in numpy access
        def cost_and_grad_np(
            design_params_np: np.ndarray, exogenous_params_np: np.ndarray
        ):
            cost, grad = cost_and_grad(
                jnp.array(design_params_np), jnp.array(exogenous_params_np)
            )

            return cost.item(), np.array(grad, dtype=np.float64)

        # Return the needed functions
        return cost_and_grad_np

    def compile_cost_dp_sens_multi_phi(
        self,
        prng_key: PRNGKeyArray,
        n_phi: int,
        sw: float = 0.0,
        vw: float = 0.0,
        jit: bool = True,
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Compile the cost function for the design parameters (to minimize cost)

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            n_phi: number of samples from phi to consider
            sw: sensitivity weight to use
            vw: variance weight to use
            jit: if True, jit the cost and gradient function
        returns:
            - a function that takes parameter values and returns a tuple of cost and the
                gradient of that cost w.r.t. the design parameters.
        """

        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            return self.design_problem.cost_fn(design_params, exogenous_params)

        # Automatically differentiate wrt phi
        cost_and_grad_phi = jax.value_and_grad(cost, argnums=(1))

        # Make the sensitivity-regularized cost
        def sensitivity_regularized_cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            cost, grad_phi = cost_and_grad_phi(design_params, exogenous_params)
            return cost + (grad_phi ** 2).sum() * sw

        # Vectorize the cost function with respect to the exogenous parameters
        # None indicates that we do not vectorize wrt design parameters
        # 0 indicates that we add a batch dimension as the 0-th dimension of the
        # exogenous parameters
        costv = jax.vmap(sensitivity_regularized_cost, (None, 0))

        # Define the variance-regularized cost as a function of design parameters
        def cost_mean_and_variance(
            design_params: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Sample a number of exogenous parameters. This will generate the same
            # sample on each call, since we use the same prng_key
            exogenous_sample = self.design_problem.exogenous_params.sample(
                prng_key, n_phi
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
            return sample_mean + vw * sample_variance

        # Automatically differentiate
        vr_cost_and_grad = jax.value_and_grad(variance_regularized_cost)

        if jit:
            vr_cost_and_grad = jax.jit(vr_cost_and_grad)

        # Wrap in numpy access
        def vr_cost_and_grad_np(design_params_np: np.ndarray, ep_np: np.ndarray):
            # We don't use these values of the exogenous params, since we've already
            # sampled a bunch of them
            vr_cost, grad = vr_cost_and_grad(jnp.array(design_params_np))

            return vr_cost.item(), np.array(grad, dtype=np.float64)

        # Return the needed function
        return vr_cost_and_grad_np

    def compile_cost_ep(
        self, prng_key: PRNGKeyArray, jit: bool = True
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Compile the cost function for the exogenous parameters (to maximize cost)

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            jit: if True, jit the cost and gradient function
        returns:
            - a function that takes parameter values and returns a tuple of cost and the
                gradient of that cost w.r.t. the design parameters.
        """
        # Wrap the cost function to MAXIMIZE the cost
        def cost(
            design_params: jnp.ndarray, exogenous_params: jnp.ndarray
        ) -> jnp.ndarray:
            return -self.design_problem.cost_fn(design_params, exogenous_params)

        # Automatically differentiate wrt the exogenous parameters
        cost_and_grad = jax.value_and_grad(cost, argnums=(1))

        if jit:
            cost_and_grad = jax.jit(cost_and_grad)

        # Wrap in numpy access
        def cost_and_grad_np(
            design_params_np: np.ndarray, exogenous_params_np: np.ndarray
        ):
            cost, grad = cost_and_grad(
                jnp.array(design_params_np), jnp.array(exogenous_params_np)
            )

            return cost.item(), np.array(grad, dtype=np.float64)

        # Return the needed functions
        return cost_and_grad_np

    def optimize(
        self,
        prng_key: PRNGKeyArray,
        disp: bool = False,
        maxiter: int = 300,
        rounds: int = 1,
        jit: bool = True,
    ) -> Tuple[bool, str, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Optimize the design problem, starting with the initial values stored in
        self.design_problem.design_params.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            disp: if True, display optimization progress
            maxiter: maximum number of optimization iterations to perform. Defaults to
                no limit.
            rounds: number of rounds to play between design and exogenous parameters
            jit: if True, JIT the cost and gradient function.
        returns:
            a JAX array of the optimal values of the design parameters
            a JAX array of the optimal values of the exogenous parameters
            a single-element JAX array of the cost at the optimal parameters
            a single-element JAX array of the cost sensitivity at the optimal parameters
        """
        # Compile the cost functions for both design and exogenous parameters
        # Single phi, vanilla cost
        f_dp = self.compile_cost_dp_single_phi(prng_key, jit=jit)

        # # Multiple phi, vanilla cost (variance weight = 0.0)
        # f_dp = self.compile_cost_dp_multi_phi(prng_key, 32, 0.0, jit=jit)

        # # Multiple phi, variance-regularized cost (variance weight = 0.1)
        # f_dp = self.compile_cost_dp_multi_phi(prng_key, 64, 0.1, jit=jit)

        # # # Single phi, sensitivity-regularized cost (weight = 1e-3)
        # f_dp = self.compile_cost_dp_sens_single_phi(prng_key, 1e-3, jit=jit)

        # # Multiple phi, sensitivity- and variance-regularized cost (sw=1e-3, vw=0.0)
        # f_dp = self.compile_cost_dp_sens_multi_phi(prng_key, 128, 1e-3, 0.0, jit=jit)

        f_ep = self.compile_cost_ep(prng_key, jit=jit)

        # Get the bounds on the parameters
        dp_bounds = self.design_problem.design_params.bounds_list
        ep_bounds = self.design_problem.exogenous_params.bounds_list

        # Set the optimization method to support bounds
        method = "L-BFGS-B"
        opts: Dict[str, Any] = {"disp": disp, "maxiter": maxiter}

        # Get the initial guess stored in the design and exogenous parameters
        design_params = self.design_problem.design_params.get_values_np()
        exogenous_params = self.design_problem.exogenous_params.get_values_np()

        # JIT functions and save the time required
        jit_start = time.perf_counter()
        f_dp(design_params, exogenous_params)
        dp_jit_end = time.perf_counter()
        f_ep(design_params, exogenous_params)
        jit_end = time.perf_counter()
        print(
            (
                f"JIT took {jit_end - jit_start:.4f} s "
                f"({dp_jit_end - jit_start:.4f} s for dp, "
                f"{jit_end - dp_jit_end:.4f} s for ep)"
            )
        )

        for i in range(rounds):
            start = time.perf_counter()
            # First minimize cost by changing the design parameters
            f = lambda dp: f_dp(dp, exogenous_params)
            dp_result = sciopt.minimize(
                f,
                design_params,
                method=method,
                jac=True,  # f returns both cost and gradient in a tuple
                bounds=dp_bounds,
                options=opts,
            )
            end = time.perf_counter()
            # Extract the result and get the cost
            design_params = np.array(dp_result.x)
            cost, _ = f_ep(design_params, exogenous_params)
            print(f"[Round {i}]: Optimized design params, cost {cost:.4f}")
            print(f"[Round {i}]: Optimizing design params took {end - start:.4f} s")

            # Then maximize the cost by changing the exogenous parameters
            f = lambda ep: f_ep(design_params, ep)
            ep_result = sciopt.minimize(
                f,
                exogenous_params,
                method=method,
                jac=True,  # f returns both cost and gradient in a tuple
                bounds=ep_bounds,
                options=opts,
            )
            # Extract the result
            exogenous_params = np.array(ep_result.x)
            print(f"[Round {i}]: Optimized exogenous params, cost {ep_result.fun:.4f}")

        return (
            jnp.array(dp_result.x),
            jnp.array(ep_result.x),
            jnp.array(ep_result.fun),
        )
