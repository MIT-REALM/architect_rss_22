import jax

from architect.design import DesignProblem
from .sat_cost import sat_cost
from .sat_design_parameters import SatDesignParameters
from .sat_exogenous_parameters import SatExogenousParameters
from .sat_simulator import sat_simulate
from .sat_stl_specification import make_sat_rendezvous_specification


def make_sat_design_problem(
    specification_weight: float,
    time_steps: int,
    dt: float,
    substeps: int,
) -> DesignProblem:
    """Make an instance of the multi-agent manipulation design problem.
    Uses two turtlebots.

    args:
        specification_weight: how much to weight the STL robustness in the cost
        time_steps: the number of steps to simulate
        dt: the duration of each time step
        substeps: how many smaller updates to break this interval into
    returns:
        a DesignProblem
    """
    # Define the exogenous parameters
    ep = SatExogenousParameters()

    # Define the design parameters
    dp = SatDesignParameters(time_steps)

    # Make the STL specification
    stl_specification = make_sat_rendezvous_specification()
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
            substeps,
        )

    # Wrap the simulator function
    def simulator(design_params, exogenous_sample):
        return sat_simulate(
            design_params,
            exogenous_sample,
            time_steps,
            dt,
            substeps,
        )

    # Make a design problem instance
    sat_design_problem = DesignProblem(dp, ep, cost_fn, simulator)
    return sat_design_problem
