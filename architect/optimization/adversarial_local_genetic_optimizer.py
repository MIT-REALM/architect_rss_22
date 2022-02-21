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
@partial(jax.jit, static_argnames=["n_mutations"])
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


@partial(jax.jit, static_argnames=["new_samples"])
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
    parent_1 = jax.random.choice(prng_subkey, pop, shape=(new_samples,))
    prng_key, prng_subkey = jax.random.split(prng_key)
    parent_2 = jax.random.choice(prng_subkey, pop, shape=(new_samples,))

    # Figure out how much to cross them
    prng_key, prng_subkey = jax.random.split(prng_key)
    crossover_ratio = jax.random.uniform(prng_subkey, (new_samples, 1))
    crossover_ratio = jnp.tile(crossover_ratio, (1, pop.shape[1]))

    # Construct the children by crossing over
    children = crossover_ratio * parent_1 + (1 - crossover_ratio) * parent_2

    # Return the augmented population
    return jnp.concatenate((pop, children))


@partial(jax.jit, static_argnames=["n_keep_samples"])
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
    design_params: jnp.ndarray,
    exogenous_params: jnp.ndarray,
    cost_and_grad: Callable[
        [jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
    ],
    design_param_bounds: jnp.ndarray,
    exogenous_param_bounds: jnp.ndarray,
    learning_rate: float,
    n_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run gradient descent for a fixed number of steps.

    args:
        design_params: (n,) array initial guess for parameters being optimized
        exogenous_params: (batch, m) array values of parameters not being optimized
        cost_and_grad: function taking (n,) x (m,) -> (1,) x (n,) x (m,)
            (i.e. returns the cost and gradients wrt design and exogenous params).
        design_param_bounds: (n, 2) array of lower and upper bounds on each parameter
        exogenous_param_bounds: (n, 2) array of lower and upper bounds on each parameter
        learning_rate: gradient descent step size
        n_steps: number of steps to take
    returns
        - optimized design parameters
        - optimized exogenous parameters
        - cost with optimized parameters
    """
    for i in range(n_steps):
        # Get the gradient
        _, (grad_dp, grad_ep) = cost_and_grad(design_params, exogenous_params)

        # Step (design params minimize, but exogenous params maximize)
        design_params = design_params - learning_rate * grad_dp
        exogenous_params = exogenous_params + learning_rate * grad_ep

        # Project to bounds
        design_params = jnp.clip(
            design_params, design_param_bounds[:, 0], design_param_bounds[:, 1]
        )
        exogenous_params = jnp.clip(
            exogenous_params, exogenous_param_bounds[:, 0], exogenous_param_bounds[:, 1]
        )

    # Return optimized params and cost
    cost, _ = cost_and_grad(design_params, exogenous_params)
    return design_params, exogenous_params, cost


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

    def initialize_populations(
        self, prng_key: PRNGKeyArray, initial_design_params: jnp.ndarray
    ):
        """Randomly initialize the populations of design and exogenous parameters.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            initial_design_params: initial values for design params
        """
        # Initialize design parameters uniformly spread around the initial guess
        # spread = 10% of bounds
        prng_key, prng_subkey = jax.random.split(prng_key)
        self._dp_pop = jax.random.uniform(
            prng_key, shape=self._dp_pop.shape, minval=-1.0, maxval=1.0
        )
        for dim_idx in range(self.design_problem.design_params.size):
            lower, upper = self.design_problem.design_params.bounds[dim_idx]
            spread = 0.1 * (upper - lower)
            self._dp_pop = self._dp_pop.at[:, dim_idx].set(
                self._dp_pop[:, dim_idx] * spread + initial_design_params[dim_idx]
            )

        # Initialize exogenous parameters uniformly within their bounds
        prng_key, prng_subkey = jax.random.split(prng_key)
        self._ep_pop = self.design_problem.exogenous_params.sample(
            prng_subkey, batch_size=self.N_ep
        )

    def optimize(
        self,
        prng_key: PRNGKeyArray,
        initial_design_params: jnp.ndarray,
        smoothing: float = 1.0,
        disp: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Optimize the design problem using the adversarial local genetic method

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
            initial_design_params: array of single initial value for the design params
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
        start = time.perf_counter()
        self.initialize_populations(prng_key, initial_design_params)
        end = time.perf_counter()
        print(f"Done (took {(end - start):0.3f} s)", flush=True)

        # Compile objectives
        print("Constructing objectives... ", end="", flush=True)
        start = time.perf_counter()
        objective_and_grad = jax.jit(
            jax.value_and_grad(self.design_problem.cost_fn, argnums=(0, 1))
        )
        # Run once to activate any JAX transforms
        objective_and_grad(self._dp_pop[0, :], self._ep_pop[0, :])
        end = time.perf_counter()
        print(f"Done (took {(end - start):0.3f} s)", flush=True)

        N_tests = 5
        total_time = 0.0
        for i in range(N_tests):
            start = time.perf_counter()
            objective_and_grad(self._dp_pop[0, :], self._ep_pop[0, :])
            end = time.perf_counter()
            total_time += end - start
        print(f"Running both objectives takes {total_time / N_tests} s on average")

        # Vectorize and compile gradient descent functions
        def wrapped_gradient_descent(
            dp: jnp.ndarray, ep: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return gradient_descent(
                dp,
                ep,
                objective_and_grad,
                self.design_problem.design_params.bounds,
                self.design_problem.exogenous_params.bounds,
                self.learning_rate,
                self.learning_steps,
            )

        print("Constructing gradient descent functions... ", end="", flush=True)
        start = time.perf_counter()
        wrapped_gradient_descent(self._dp_pop[0, :], self._ep_pop[0, :])
        end = time.perf_counter()
        print(f"Done (took {(end - start):0.3f} s)", flush=True)

        print("Vectorizing... ", end="", flush=True)
        start = time.perf_counter()
        gradient_descent_v = jax.vmap(wrapped_gradient_descent, (0, 0))
        end = time.perf_counter()
        print(f"Done (took {(end - start):0.3f} s)", flush=True)

        # Run once to make sure JIT is active
        print("Burn-in... ", end="", flush=True)
        start = time.perf_counter()
        gradient_descent_v(self._dp_pop, self._ep_pop)
        end = time.perf_counter()
        print(f"Done (took {(end - start):0.3f} s)", flush=True)

        # For each generation
        print("Starting evolution...", flush=True)
        for generation_i in range(self.N_generations):
            # Optimize DP, downselect, crossover, and mutate
            gen_tag = f"[Generation {generation_i}][design] "
            print(gen_tag + "Starting gradient descent... ", end="", flush=True)
            self._dp_pop, dp_costs = dp_gradient_descent_v(self._dp_pop, self._ep_pop)
            print(f"Done; best (min) cost {dp_costs.min()}", flush=True)

            self._dp_pop = downselect_and_cross(
                prng_key, self._dp_pop, dp_costs, self.n_keep_dp
            )

            self._dp_pop = mutate_population(
                prng_key,
                self._dp_pop,
                self.n_keep_dp,
                self.mutation_rate,
            )

            # Optimize EP, downselect, crossover, and mutate
            gen_tag = f"[Generation {generation_i}][exogenous] "
            print(gen_tag + "Starting gradient descent... ", end="", flush=True)
            self._ep_pop, ep_costs = ep_gradient_descent_v(self._ep_pop, self._dp_pop)
            print(f"Done; best (max) cost {ep_costs.max()}", flush=True)

            self._ep_pop = downselect_and_cross(
                prng_key, self._ep_pop, ep_costs, self.n_keep_ep
            )

            self._ep_pop = mutate_population(
                prng_key,
                self._ep_pop,
                self.n_keep_ep,
                self.mutation_rate,
            )

        print("Evolution done!", flush=True)

        # Return the best set of design parameters, the population of exogenous
        # parameters, and the optimal cost (max over _ep_pop)
        dp_costs, _ = jax.vmap(objective_and_grad, (0, None))(
            self._dp_pop, self._ep_pop
        )
        dp_opt_idx = jnp.argmin(dp_costs)
        dp_opt = self._dp_pop[dp_opt_idx]
        dp_opt_cost = dp_costs[dp_opt_idx]

        return dp_opt, self._ep_pop, dp_opt_cost
