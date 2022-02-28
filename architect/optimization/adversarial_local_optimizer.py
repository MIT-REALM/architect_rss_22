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

    def compile_cost_dp(
        self, prng_key: PRNGKeyArray, jit: bool = True
    ) -> Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]:
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

        # Wrap in numpy access and take the mean
        def cost_and_grad_np(
            design_params_np: np.ndarray, exogenous_params_np: np.ndarray
        ):
            # Manually batch to avoid re-jitting
            exogenous_params = jnp.array(exogenous_params_np).reshape(
                -1, exogenous_params_np.shape[-1]
            )
            cost = jnp.zeros((exogenous_params.shape[0]))
            grad = jnp.zeros((exogenous_params.shape[0], design_params_np.size))

            for i, ep in enumerate(exogenous_params):
                cost_i, grad_i = cost_and_grad(
                    jnp.array(design_params_np),
                    ep,
                )
                cost = cost.at[i].set(cost_i)
                grad = grad.at[i].set(grad_i)

            return cost.mean().item(), np.array(grad.mean(axis=0), dtype=np.float64)

        # Return the needed functions
        return cost_and_grad_np

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
        n_init: int = 4,
        stopping_tolerance: float = 0.1,
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
            n_init: number of initial exogenous samples to start with
            stopping_tolerance: stop when the difference between successive adversarial
                examples is less than this value
            jit: if True, JIT the cost and gradient function.
        returns:
            a JAX array of the optimal values of the design parameters
            a JAX array of the optimal values of the exogenous parameters
            a single-element JAX array of the cost at the optimal parameters. This cost
                is measured after the final response by the adversary.
            a single-element JAX array of the difference between the cost after the
                adversary's last response and the cost just before that response. This
                effectively measures the brittleness of the optimized design parameters.
            float of the time spent running optimization routines.
            float of the time spent running JIT
            int of number of rounds used
            int number of exogenous samples used to optimized design
        """
        # Compile the cost functions for both design and exogenous parameters
        f_dp = self.compile_cost_dp(prng_key, jit=jit)
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
        jit_time = jit_end - jit_start
        # print(
        #     (
        #         f"JIT took {jit_time:.4f} s "
        #         f"({dp_jit_end - jit_start:.4f} s for dp, "
        #         f"{jit_end - dp_jit_end:.4f} s for ep)"
        #     )
        # )

        # Maintain a population of exogenous parameters
        exogenous_pop = self.design_problem.exogenous_params.sample(
            prng_key,
            batch_size=n_init,
        )
        # exogenous_pop = exogenous_params.reshape(1, -1)

        total_time = 0.0
        for i in range(rounds):
            dp_start = time.perf_counter()
            # First minimize cost by changing the design parameters
            f = lambda dp: f_dp(dp, exogenous_pop)
            dp_result = sciopt.minimize(
                f,
                design_params,
                method=method,
                jac=True,  # f returns both cost and gradient in a tuple
                bounds=dp_bounds,
                options=opts,
            )
            dp_end = time.perf_counter()
            # Extract the result and get the cost
            design_params = np.array(dp_result.x)
            dp_cost, _ = f_ep(design_params, exogenous_params)
            # print(f"[Round {i}]: Optimized design params, dp_cost {dp_cost:.4f}")
            # print(f"[Round {i}]: Optimizing dp took {dp_end - dp_start:.4f} s")

            # Then maximize the cost by changing the exogenous parameters
            ep_start = time.perf_counter()
            f = lambda ep: f_ep(design_params, ep)
            ep_result = sciopt.minimize(
                f,
                exogenous_params,
                method=method,
                jac=True,  # f returns both cost and gradient in a tuple
                bounds=ep_bounds,
                options=opts,
            )
            ep_end = time.perf_counter()
            total_time += (ep_end - ep_start) + (dp_end - dp_start)
            # print(f"total time: {total_time}")
            # Stop if we've converged
            # print(f"Adversary moved {np.linalg.norm(ep_result.x - exogenous_params)}")
            if np.linalg.norm(ep_result.x - exogenous_params) < stopping_tolerance:
                break

            # Otherwise, extract the result and add it to the population
            exogenous_params = np.array(ep_result.x)
            exogenous_pop = jnp.vstack((exogenous_pop, exogenous_params))
            # print(f"[Round {i}]: Optimized exogenous params, cost {ep_result.fun:.4f}")

        # print(f"Overall, optimization took {total_time:.4f} s")

        pop_size = exogenous_pop.shape[0]
        if i == rounds - 1:
            pop_size -= 1  # don't count the sample added after the last round

        return (
            jnp.array(dp_result.x),
            jnp.array(ep_result.x),
            -jnp.array(ep_result.fun),
            -jnp.array(ep_result.fun - dp_cost),
            total_time,
            jit_time,
            i,  # number of rounds
            pop_size,
        )
