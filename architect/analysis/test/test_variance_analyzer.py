import jax
import jax.numpy as jnp

from architect.design import DesignProblem, DesignParameters, ExogenousParameters
from architect.analysis import VarianceAnalyzer


def create_test_design_problem():
    """Create a simple DesignProblem instance to use during testing.

    This problem has a cost minimum at dp[0] = 0 regardless of regularization, with
    cost = dp[0] ** 2 + ep[0]

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
        return jnp.concatenate([dp, ep])

    def cost_fn(trace):
        return (trace[0] ** 2).sum() + trace[-1]

    problem = DesignProblem(dp, ep, simulator, cost_fn)

    return problem


def test_VarianceAnalyzer_init():
    """Test initializing a VarianceAnalyzer"""
    # Get the test problem
    problem = create_test_design_problem()

    # Initialize and make sure it worked
    sample_size = 500
    variance_analyzer = VarianceAnalyzer(problem, sample_size)
    assert variance_analyzer is not None


def test_VarianceAnalyzer_analyze():
    """Test using a VarianceAnalyzer to analyze variance"""
    # Get the test problem
    problem = create_test_design_problem()

    # Initialize the analyzer
    sample_size = 500
    variance_analyzer = VarianceAnalyzer(problem, sample_size)

    # Get a PRNG key
    key = jax.random.PRNGKey(0)

    # Run the analysis
    cost_mean, cost_variance = variance_analyzer.analyze(key)
    assert jnp.allclose(cost_mean, jnp.zeros(1), atol=1e-1)
    assert jnp.allclose(cost_variance, jnp.ones(1), atol=1e-1)
