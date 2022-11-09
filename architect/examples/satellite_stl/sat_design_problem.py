import jax

from architect.design.problem import BoundedDesignProblem
from .sat_cost import sat_cost
from .sat_design_parameters import SatDesignParameters
from .sat_exogenous_parameters import SatExogenousParameters
from .sat_simulator import sat_simulate_dt
from .sat_stl_specification import make_sat_rendezvous_specification


def make_sat_design_problem(
    specification_weight: float,
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
    ep = SatExogenousParameters()

    # Define the design parameters
    dp = SatDesignParameters(time_steps)

    # Make the STL specification
    stl_specification = make_sat_rendezvous_specification(mission_1)
    stl_specification_f = lambda trace: stl_specification(trace, smoothing=1e3)
    stl_specification_jit = jax.jit(stl_specification_f)

    # Wrap the cost function
    def cost_fn(design_params, exogenous_sample):
        return sat_cost(
            design_params,
            exogenous_sample,
            stl_specification_jit,
            specification_weight,
            time_steps,
            dt,
        )

    # Wrap the simulator function
    def simulator(design_params, exogenous_sample):
        return sat_simulate_dt(
            design_params,
            exogenous_sample,
            time_steps,
            dt,
        )

    # Make a design problem instance
    sat_design_problem = BoundedDesignProblem(dp, ep, cost_fn, simulator)
    return sat_design_problem
