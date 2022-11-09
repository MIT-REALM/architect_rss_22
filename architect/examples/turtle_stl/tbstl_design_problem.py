import jax

from architect.design.problem import BoundedDesignProblem
from .tbstl_cost import tbstl_cost
from .tbstl_design_parameters import TBSTLDesignParameters
from .tbstl_exogenous_parameters import TBSTLExogenousParameters
from .tbstl_simulator import tbstl_simulate_dt
from .tbstl_stl_specification import make_tbstl_rendezvous_specification


def make_tbstl_design_problem(
    time_steps: int,
    dt: float,
    mission_1: bool = False,
) -> BoundedDesignProblem:
    """Make an instance of the multi-agent manipulation design problem.
    Uses two turtlebots.

    args:
        specification_weight: how much to weight the STL robustness in the cost
        time_steps: the number of steps to simulate
        dt: the duration of each time step
        mission_1: if true, use the simpler mission spec
    returns:
        a BoundedDesignProblem
    """
    # Define the exogenous parameters
    ep = TBSTLExogenousParameters()

    # Define the design parameters
    dp = TBSTLDesignParameters(time_steps)

    # Make the STL specification
    stl_specification = make_tbstl_rendezvous_specification(mission_1)
    stl_specification_f = lambda trace: stl_specification(trace, smoothing=1e3)
    stl_specification_jit = jax.jit(stl_specification_f)

    # Wrap the cost function
    def cost_fn(design_params, exogenous_sample):
        return tbstl_cost(
            design_params,
            exogenous_sample,
            stl_specification_jit,
            time_steps,
            dt,
        )

    # Wrap the simulator function
    def simulator(design_params, exogenous_sample):
        return tbstl_simulate_dt(
            design_params,
            exogenous_sample,
            time_steps,
            dt,
        )

    # Make a design problem instance
    tbstl_design_problem = BoundedDesignProblem(dp, ep, cost_fn, simulator)
    return tbstl_design_problem
