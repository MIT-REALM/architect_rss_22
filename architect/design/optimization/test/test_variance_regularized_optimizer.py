import jax
import jax.numpy as jnp

from architect.design.problem import (
    DesignProblem,
    DesignParameters,
    ExogenousParameters,
)
from architect.design.optimization import VarianceRegularizedOptimizerAD


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
    dp = DesignParameters(n)
    ep = ExogenousParameters(n)

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

    problem = DesignProblem(dp, ep, cost_fn, simulator)

    return problem


def test_VarianceRegularizedOptimizerAD_init():
    """Test initialization of VarianceRegularizedOptimizerAD"""
    # Get test problem
    problem = create_test_design_problem()

    # Initialize optimizer
    variance_weight = 0.1
    sample_size = 10
    vropt = VarianceRegularizedOptimizerAD(problem, variance_weight, sample_size)
    assert vropt is not None


def test_VarianceRegularizedOptimizerAD_compile():
    """Test compiling cost function of VarianceRegularizedOptimizerAD"""
    # Get test problem
    problem = create_test_design_problem()

    # Initialize optimizer
    variance_weight = 0.1
    sample_size = 10
    vropt = VarianceRegularizedOptimizerAD(problem, variance_weight, sample_size)

    # Set a PRNG key to use
    key = jax.random.PRNGKey(0)

    # Compile the cost function
    vr_cost_and_grad, cost_mean_and_variance = vropt.compile_cost_fn(key)

    for i in range(5):
        dp_val = jnp.array([i], dtype=jnp.float32)

        # Test the cost mean and variance. Variance = 0 for the simple problem
        mean, var = cost_mean_and_variance(dp_val)
        assert jnp.allclose(mean, jnp.array([dp_val ** 2]))
        assert jnp.allclose(var, jnp.zeros(1))

        # Test the cost gradient
        cost, grad = vr_cost_and_grad(dp_val)
        assert jnp.allclose(cost, mean)
        assert jnp.allclose(grad, 2 * dp_val)

    # Now try again with a more complicated problem (with stochasticity)
    # Get test problem
    problem = create_test_design_problem(simple=False)

    # Initialize optimizer
    variance_weight = 0.1
    sample_size = 500
    vropt = VarianceRegularizedOptimizerAD(problem, variance_weight, sample_size)

    # Compile the cost function
    vr_cost_and_grad, cost_mean_and_variance = vropt.compile_cost_fn(key)

    for i in range(5):
        dp_val = jnp.array([i], dtype=jnp.float32)

        # Test the cost mean and variance. The mean should not change, but the variance
        # should have increased
        mean, var = cost_mean_and_variance(dp_val)
        assert jnp.allclose(mean, jnp.array([dp_val ** 2]), atol=1e-1)
        assert jnp.allclose(var, jnp.ones(1), atol=1e-1)

        # Test the cost gradient; it should have been unaffected
        cost, grad = vr_cost_and_grad(dp_val)
        assert jnp.allclose(cost, mean + variance_weight * var)
        assert jnp.allclose(grad, 2 * dp_val)


def test_VarianceRegularizedOptimizerAD_optimize():
    """Test whether VarianceRegularizedOptimizerAD can optimize a design problem"""
    # Get test problem. Start simple
    problem = create_test_design_problem()

    # Initialize optimizer
    variance_weight = 0.1
    sample_size = 10
    vropt = VarianceRegularizedOptimizerAD(problem, variance_weight, sample_size)

    # Set a PRNG key to use
    key = jax.random.PRNGKey(0)

    # Optimize!
    success, msg, dp_opt, cost_mean, cost_var = vropt.optimize(key)
    assert success
    assert jnp.allclose(dp_opt, jnp.zeros(1))
    assert jnp.allclose(cost_mean, jnp.zeros(1))
    assert jnp.allclose(cost_var, jnp.zeros(1))

    # Now try with exogenous effects
    problem = create_test_design_problem(simple=False)

    # Initialize optimizer
    variance_weight = 0.1
    sample_size = 500
    vropt = VarianceRegularizedOptimizerAD(problem, variance_weight, sample_size)

    # Optimize!
    success, msg, dp_opt, cost_mean, cost_var = vropt.optimize(key)
    assert success
    assert jnp.allclose(dp_opt, jnp.zeros(1))
    assert jnp.allclose(cost_mean, jnp.zeros(1), atol=1e-1)
    assert jnp.allclose(cost_var, jnp.ones(1), atol=1e-1)
