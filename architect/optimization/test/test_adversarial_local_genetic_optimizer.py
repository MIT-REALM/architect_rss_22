import jax
import jax.numpy as jnp

from architect.design import (
    BoundedDesignProblem,
    BoundedDesignParameters,
    BoundedExogenousParameters,
)
from architect.optimization import AdversarialLocalGeneticOptimizer


def create_test_design_problem(simple=True):
    """Create a simple DesignProblem instance to use during testing. May be simplified
    so that the exogenous parameters do not affect the cost.

    This problem has a cost minimum at dp[0] = 0 regardless of regularization

    args:
        simple: bool. If True, do not include exogenous parameters in cost. Otherwise,
                add the exogenous parameters to the cost
    """
    # Create simple design and exogenous parameters
    n = 1
    dp = BoundedDesignParameters(n, jnp.array([[-1.0, 1.0]]))
    ep = BoundedExogenousParameters(n, jnp.array([[-1.0, 1.0]]))

    # Create a simple simulator that passes through the design params and a cost
    # function that computes the squared 2-norm of the design params
    def simulator(dp, ep):
        if simple:
            return dp
        else:
            return jnp.concatenate([dp, ep])

    def cost_fn(dp, ep):
        trace = simulator(dp, ep)
        if simple:
            return (trace[0] ** 2).sum()
        else:
            return (trace[0] ** 2).sum() + trace[-1]

    problem = BoundedDesignProblem(dp, ep, cost_fn, simulator)

    return problem


def test_AdversarialLocalGeneticOptimizer_init():
    """Test initialization of VarianceRegularizedOptimizerAD"""
    # Get test problem
    problem = create_test_design_problem()

    # Initialize optimizer
    N_dp = 10
    N_ep = 10
    N_generations = 3
    learning_rate = 1e-2
    learning_steps = 10
    selection_fraction = 0.3
    mutation_rate = 0.3
    opt = AdversarialLocalGeneticOptimizer(
        problem,
        N_dp,
        N_ep,
        N_generations,
        learning_rate,
        learning_steps,
        selection_fraction,
        mutation_rate,
    )
    assert opt is not None


def test_AdversarialLocalGeneticOptimizer_compile():
    """Test compiling cost function of VarianceRegularizedOptimizerAD"""
    # Get test problem
    problem = create_test_design_problem()

    # Initialize optimizer
    N_dp = 10
    N_ep = 10
    N_generations = 3
    learning_rate = 1e-2
    learning_steps = 10
    selection_fraction = 0.3
    mutation_rate = 0.3
    opt = AdversarialLocalGeneticOptimizer(
        problem,
        N_dp,
        N_ep,
        N_generations,
        learning_rate,
        learning_steps,
        selection_fraction,
        mutation_rate,
    )

    # Set a PRNG key to use
    key = jax.random.PRNGKey(0)

    # Compile the cost function
    dp_max_objective_grad, ep_max_objective_grad = opt.compile_cost_grad_fns()

    # Test that both run successfully
    obj, grad = dp_max_objective_grad(
        problem.design_params.get_values().reshape(1, -1),
        problem.exogenous_params.sample(key, N_ep),
    )
    assert obj.size == 1
    assert grad.shape == (1, 1)

    obj, grad = ep_max_objective_grad(
        problem.exogenous_params.sample(key, 1), jnp.ones((N_dp, 1))
    )
    assert obj.size == 1
    assert grad.shape == (1, 1)
