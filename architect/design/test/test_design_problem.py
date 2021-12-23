from architect.design import DesignProblem, DesignParameters, ExogenousParameters


def test_DesignProblem_init():
    """Test initializing a simple DesignProblem"""
    # Create simple design and exogenous parameters
    n = 5
    m = 2
    dp = DesignParameters(n)
    ep = ExogenousParameters(m)

    # Create a cost function that computes the squared 2-norm of the design params

    def cost_fn(dp, ep):
        return (dp ** 2).sum()

    # Create the design problem and make sure it initialized
    problem = DesignProblem(dp, ep, cost_fn, cost_fn)
    assert problem is not None
