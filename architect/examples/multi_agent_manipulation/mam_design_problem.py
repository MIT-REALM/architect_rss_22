from typing import List

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import DesignProblem
from .mam_cost import mam_cost_push_two_turtles
from .mam_design_parameters import MAMDesignParameters
from .mam_exogenous_parameters import MAMExogenousParameters
from .mam_simulator import mam_simulate_single_push_two_turtles


def make_mam_design_problem(
    layer_widths: List[int], dt: float, prng_key: PRNGKeyArray
) -> DesignProblem:
    """Make an instance of the multi-agent manipulation design problem.
    Uses two turtlebots.

    args:
        layer_widths: number of units in each layer. First element should be 15. Last
            element should be 6
        dt: timestep
        prng_key: key for pseudo-random number generation for initializing the planning
                  network.
    returns:
        a DesignProblem
    """
    # Define the exogenous parameters
    mu_box_turtle_range = jnp.array([0.05, 0.2])
    mu_turtle_ground_range = jnp.array([0.6, 0.8])
    mu_box_ground_range = jnp.array([0.4, 0.6])
    box_mass_range = jnp.array([0.9, 1.1])
    desired_box_pose_range = jnp.array(
        [
            [0.0, 0.3],
            [0.0, 0.3],
            [-jnp.pi / 4.0, jnp.pi / 4.0],
        ]
    )
    turtlebot_displacement_covariance = (0.1 ** 2) * jnp.eye(3)
    ep = MAMExogenousParameters(
        mu_turtle_ground_range,
        mu_box_ground_range,
        mu_box_turtle_range,
        box_mass_range,
        desired_box_pose_range,
        turtlebot_displacement_covariance,
        2,
    )

    # Define the design parameters
    prng_key, subkey = jax.random.split(prng_key)
    dp = MAMDesignParameters(subkey, layer_widths)

    # Wrap the cost function
    def cost_fn(design_params, exogenous_sample):
        return mam_cost_push_two_turtles(
            design_params,
            exogenous_sample,
            layer_widths,
            dt,
        )

    # Wrap the simulator function
    def simulator(design_params, exogenous_sample):
        return mam_simulate_single_push_two_turtles(
            design_params,
            exogenous_sample,
            layer_widths,
            dt,
        )

    # Make a design problem instance
    agv_design_problem = DesignProblem(dp, ep, cost_fn, simulator)
    return agv_design_problem
