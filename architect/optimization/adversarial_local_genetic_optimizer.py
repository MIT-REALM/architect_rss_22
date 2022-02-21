"""
Optimizes a design to achieve minimal cost subject to adversarial changes to
exogenous parameters
"""
from functools import partial
import time
from typing import Callable, Tuple

import jax
from jax.nn import logsumexp, softmax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import BoundedDesignProblem


# Define some useful static methods to JIT compile
@jax.jit
def mutate_population(
    prng_key: PRNGKeyArray, pop: jnp.ndarray, n_mutations: int, mutation_rate: float
) -> jnp.ndarray:
    """Mutate a random selection of n_mutations elements from pop with a perturbation
    of size mutation_rate.

    args:
        prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        pop: the population to mutate
        n_mutations: how many samples to mutate
        mutation_rate: size of mutation
    returns:
        mutated population
    """
    # Get indices to mutate
    prng_key, prng_subkey = jax.random.split(prng_key)
    mutation_indices = jax.random.permutation(prng_subkey, pop.shape[0])
    mutation_indices = mutation_indices[:n_mutations]

    # Add the perturbation
    prng_key, prng_subkey = jax.random.split(prng_key)
    perturbation = mutation_rate * jax.random.normal(
        prng_subkey, shape=pop[mutation_indices].shape
    )
    return pop.at[mutation_indices].add(perturbation)


@jax.jit
def crossover(
    prng_key: PRNGKeyArray, pop: jnp.ndarray, new_samples: int
) -> jnp.ndarray:
    """Generate new samples by randomly combining two others.

    args:
        prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        pop: the population to cross over
    returns:
        the crossed-over population with pop.shape[0] + new_samples samples
    """
    # Figure out which things to cross
    prng_key, prng_subkey = jax.random.split(prng_key)
    parent_1 = pop[:new_samples]
    parent_2 = jax.random.choice(prng_subkey, pop, shape=(new_samples,))

    # Figure out how much to cross them
    prng_key, prng_subkey = jax.random.split(prng_key)
    crossover_ratio = jax.random.uniform(prng_subkey, (new_samples, 1))
    crossover_ratio = jnp.tile(crossover_ratio, (1, pop.shape[1]))

    # Construct the children by crossing over
    children = crossover_ratio * parent_1 + (1 - crossover_ratio) * parent_2

    # Return the augmented population
    return jnp.concatenate((pop, children))


@jax.jit
def downselect_and_cross(
    prng_key: PRNGKeyArray,
    pop: jnp.ndarray,
    pop_costs: jnp.ndarray,
    n_keep_samples: int,
) -> jnp.ndarray:
    """Generate a new population by downselecting and replenishing with crossover.

    args:
        prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        pop: the population to downselect and cross
        pop_costs: the scores of each sample in pop (lower is better)
        n_keep_samples: how many samples from the initial population to retain
    returns:
        the crossed-over population with pop.shape[0] + new_samples samples
    """
    # Figure out how many we'll need to regenerate
    new_samples = pop.shape[0] - n_keep_samples

    # Downselect by costs (normalize costs to probabilities)
    selection_probabilities = softmax(-pop_costs)
    prng_key, prng_subkey = jax.random.split(prng_key)
    pop = jax.random.choice(
        prng_subkey,
        pop,
        shape=(n_keep_samples,),
        replace=False,
        p=selection_probabilities,
    )

    # Add samples using crossover
    pop = crossover(prng_key, pop, new_samples)

    return pop


# @partial(jax.jit, static_argnames=["n_steps", "cost_and_grad"])
def gradient_descent(
    params: jnp.ndarray,
    other_params: jnp.ndarray,
    cost_and_grad: Callable[
        [jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]
    ],
    param_bounds: jnp.ndarray,
    learning_rate: float,
    n_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run gradient descent for a fixed number of steps.

    args:
        params: (n,) array initial guess for parameters being optimized
        other_params: (batch, m) array values of parameters not being optimized
        cost_and_grad: function taking (n,) x (batch, m) -> (1,) x (n,)
        param_bounds: (n, 2) array of lower and upper bounds on each parameter
        learning_rate: gradient descent step size
        n_steps: number of steps to take
    returns
        - optimized parameters
        - cost of optimized parameters
    """
    for i in range(n_steps):
        # Get the gradient
        _, grad = cost_and_grad(params, other_params)

        # Step
        params = params - learning_rate * grad

        # Project to bounds
        params = jnp.clip(params, param_bounds[:, 0], param_bounds[:, 0])

    # Return optimized params and cost
    cost, _ = cost_and_grad(params, other_params)
    return params, cost


class AdversarialLocalGeneticOptimizer(object):
    """
    AdversarialLocalGeneticOptimizer implements a local genetic algorithm (local in the
    sense of using Lamarckian evolution to locally optimize samples using gradient
    ascent/descent) that alternates between optimizing the design and exogenous
    parameters to solve

        min_theta max_phi J(theta, phi)

    subject to bounds on phi and optional bounds on theta
    """

    def __init__(
        self,
        design_problem: BoundedDesignProblem,
        N_dp: int,
        N_ep: int,
        N_generations: int,
        learning_rate: float,
        learning_steps: int,
        selection_fraction: float,
        mutation_rate: float,
    ):
        """Initialize the optimizer.

        args:
            design_problem: the design problem (with bounded exogeneous parameters) we
                seek to optimize
            N_dp: number of design parameter samples to include in the population
            N_ep: number of exogenous parameter samples to include in the population
            N_generations: number of generations to run
            learning_rate: gradient ascent/descent step size
            learning_steps: number of gradient ascent/descent steps in each generation
            selection_fraction: the fraction of samples to keep after each generation
                (the rest will be re-generated using crossovers)
            mutation_rate: the fraction of samples that will be mutated after each
                generation
        """
        super(AdversarialLocalGeneticOptimizer, self).__init__()
        self.design_problem = design_problem
        self.N_dp = N_dp
        self.N_ep = N_ep
        self.N_generations = N_generations
        self.learning_rate = learning_rate
        self.learning_steps = learning_steps
        self.selection_fraction = selection_fraction
        self.mutation_rate = mutation_rate

        # Pre-compute some things for convenience later
        self.n_keep_dp = int(self.selection_fraction * self.N_dp)
        self.n_keep_ep = int(self.selection_fraction * self.N_ep)

        # Initialize arrays to hold the populations
        self._dp_pop = jnp.zeros((self.N_dp, self.design_problem.design_params.size))
        self._ep_pop = jnp.zeros((self.N_ep, self.design_problem.exogenous_params.size))

    def initialize_populations(self, prng_key: PRNGKeyArray):
        """Randomly initialize the populations of design and exogenous parameters.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
        """
        # Initialize design parameters uniformly within their bounds
        prng_key, prng_subkey = jax.random.split(prng_key)
        self._dp_pop = jax.random.uniform(prng_key, shape=self._dp_pop.shape)
        for dim_idx in range(self.design_problem.design_params.size):
            lower, upper = self.design_problem.design_params.bounds[dim_idx]
            spread = upper - lower
            self._dp_pop = self._dp_pop.at[:, dim_idx].set(
                self._dp_pop[:, dim_idx] * spread + lower
            )

        # Initialize exogenous parameters uniformly within their bounds
        prng_key, prng_subkey = jax.random.split(prng_key)
        self._ep_pop = self.design_problem.exogenous_params.sample(
            prng_subkey, batch_size=self.N_ep
        )

    def compile_cost_grad_fns(
        self, smoothing: float = 1.0
    ) -> Tuple[
        Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
        Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    ]:
        """Compile the cost and gradient functions for local optimization of both
        design and exogenous parameters.

        args:
            smoothing: parameter determining how smooth the approximate maximum is.
                Larger = more exact.

        returns:
            - Jitted cost and gradient function for optimizing design parameters
            - Jitted cost and gradient function for optimizing exogenous parameters

        both returned functions are vmapped to be run on the entire population at once
        """
        # Design parameters are changed to minimize the cost
        def dp_objective(dp: jnp.ndarray, ep: jnp.ndarray) -> jnp.ndarray:
            return self.design_problem.cost_fn(dp, ep)

        # Exogenous parameters are changed to maximize the cost
        def ep_objective(dp: jnp.ndarray, ep: jnp.ndarray) -> jnp.ndarray:
            return -self.design_problem.cost_fn(dp, ep)

        # Vectorize each with respect to the other
        dp_objective_v = jax.vmap(dp_objective, (None, 0))  # no batch on dp, batch ep
        ep_objective_v = jax.vmap(ep_objective, (0, None))  # batch on dp, no batch ep

        # Take the maximum of each objective with respect to the other set of parameters
        def dp_max_objective(dp: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
            return 1 / smoothing * logsumexp(smoothing * dp_objective_v(dp, eps))

        # Flip argument order so it's "what's being optimized" first and a batch
        # of fixed stuff afterwards
        def ep_max_objective(ep: jnp.ndarray, dps: jnp.ndarray) -> jnp.ndarray:
            return 1 / smoothing * logsumexp(smoothing * ep_objective_v(dps, ep))

        # Get the cost and grad
        dp_max_objective_grad = jax.value_and_grad(dp_max_objective)
        ep_max_objective_grad = jax.value_and_grad(ep_max_objective)

        # Return these objective functions
        return dp_max_objective_grad, ep_max_objective_grad

    def optimize(
        self,
        prng_key: PRNGKeyArray,
        smoothing: float = 1.0,
        disp: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Optimize the design problem using the adversarial local genetic method

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            smoothing: parameter determining how smooth the approximate maximum is.
                Larger = more exact.
            disp: if True, display optimization progress
        returns:
            a JAX array of the optimal values of the design parameters
            a JAX array of the population of the exogenous parameters
            a single-element JAX array of the optimal cost
        """
        # Initialize populations
        print("Initializing population... ", end="", flush=True)
        self.initialize_populations(prng_key)
        print("Done.", flush=True)

        # Compile objectives
        print("Constructing objectives... ", end="", flush=True)
        dp_cost_and_grad, ep_cost_and_grad = self.compile_cost_grad_fns(smoothing)
        print("Done.", flush=True)

        # Vectorize and compile gradient descent functions
        def dp_gradient_descent(
            dp: jnp.ndarray, eps: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return gradient_descent(
                dp,
                eps,
                dp_cost_and_grad,
                self.design_problem.design_params.bounds,
                self.learning_rate,
                self.learning_steps,
            )

        def ep_gradient_descent(
            ep: jnp.ndarray, dps: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return gradient_descent(
                ep,
                dps,
                ep_cost_and_grad,
                self.design_problem.exogenous_params.bounds,
                self.learning_rate,
                self.learning_steps,
            )

        print("Vectorizing... ", end="", flush=True)
        dp_gradient_descent_v = jax.vmap(dp_gradient_descent, (0, None))
        ep_gradient_descent_v = jax.vmap(ep_gradient_descent, (0, None))
        print("Done.", flush=True)

        # Run each once to activate JIT
        print("Burn-in... ", end="", flush=True)
        start = time.perf_counter()
        dp_gradient_descent_v(self._dp_pop, self._ep_pop)
        ep_gradient_descent_v(self._ep_pop, self._dp_pop)
        end = time.perf_counter()
        print(f"Done (took {end - start} s)", flush=True)

        # For each generation
        print("Starting evolution...", flush=True)
        for generation_i in range(self.N_generations):
            # Optimize DP, downselect, crossover, and mutate
            gen_tag = f"[Generation {generation_i}][design] "
            print(gen_tag + "Starting gradient descent... ", end="", flush=True)
            self._dp_pop, dp_costs = dp_gradient_descent_v(self._dp_pop, self._ep_pop)
            print(f"Done; best (min) cost {dp_costs.min()}", flush=True)

            print(gen_tag + "Downselecting and crossing... ", end="", flush=True)
            self._dp_pop = downselect_and_cross(
                prng_key, self._dp_pop, dp_costs, self.n_keep_dp
            )
            print("Done", flush=True)

            print(gen_tag + "Mutating...", end="", flush=True)
            self._dp_pop = mutate_population(
                prng_key,
                self._dp_pop,
                self.n_keep_dp,
                self.mutation_rate,
            )
            print("Done", flush=True)

            # Optimize EP, downselect, crossover, and mutate
            gen_tag = f"[Generation {generation_i}][exogenous] "
            print(gen_tag + "Starting gradient descent... ", end="", flush=True)
            self._ep_pop, ep_costs = ep_gradient_descent_v(self._ep_pop, self._dp_pop)
            print(f"Done; best (max) cost {ep_costs.max()}", flush=True)

            print(gen_tag + "Downselecting and crossing... ", end="", flush=True)
            self._ep_pop = downselect_and_cross(
                prng_key, self._ep_pop, ep_costs, self.n_keep_ep
            )
            print("Done", flush=True)

            print(gen_tag + "Mutating...", end="", flush=True)
            self._ep_pop = mutate_population(
                prng_key,
                self._ep_pop,
                self.n_keep_ep,
                self.mutation_rate,
            )
            print("Done", flush=True)

        print("Evolution done!", flush=True)

        # Return the best set of design parameters, the population of exogenous
        # parameters, and the optimal cost (max over _ep_pop)
        dp_costs, _ = jax.vmap(dp_cost_and_grad, (0, None))(self._dp_pop, self._ep_pop)
        dp_opt_idx = jnp.argmin(dp_costs)
        dp_opt = self._dp_pop[dp_opt_idx]
        dp_opt_cost = dp_costs[dp_opt_idx]

        return dp_opt, self._ep_pop, dp_opt_cost
