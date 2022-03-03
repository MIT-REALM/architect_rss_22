import jax
import jax.numpy as jnp

import architect.components.specifications.stl as stl


@jax.jit
def softnorm(x):
    """Compute the 2-norm, but if x is too small replace it with the squared 2-norm
    to make sure it's differentiable. This function is continuous and has a derivative
    that is defined everywhere, but its derivative is discontinuous.
    """
    eps = 1e-5
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)


def make_tbstl_rendezvous_specification(mission_1: bool = False) -> stl.STLFormula:
    """Makes the STL specification for the turtlebot rendezvous problem.

    This specification captures the following requirements:
        - If the chaser speed is too high, do not approach the target
        - Eventually enter a ring around the target and stay there for a bit
        - After staying in that ring for a while, approach the target
    """
    ################################################
    #   Make predicates
    ################################################

    # Make a predicate for being docked with the chaser satellite
    docking_threshold = 0.2  # m
    mu_negative_distance = lambda q_t: -softnorm(q_t[:2])  # 1-lipschitz
    p_docked = stl.STLPredicate(mu_negative_distance, -docking_threshold)

    # Make a predicate for being outside the min waiting radius
    min_waiting_radius = 0.5  # m
    mu_distance = lambda q_t: softnorm(q_t[:2])  # 1-lipschitz
    p_outside_min_radius = stl.STLPredicate(mu_distance, min_waiting_radius)

    # Make a predicate for being inside the max waiting radius
    max_waiting_radius = 1.0  # m
    p_inside_max_radius = stl.STLPredicate(mu_negative_distance, -max_waiting_radius)

    ################################################
    #   Define times for the mission
    ################################################
    min_time_in_waiting_zone = 10.0

    ################################################
    #   Construct the STL formula for the mission
    ################################################

    # Reach the target before the end of the trajectory and stay there
    dock_at_target = stl.STLUntimedAlways(p_docked)
    eventually_reach_target = stl.STLUntimedEventually(dock_at_target)

    # Reach the waiting zone stay there for the specified interval
    stay_inside_waiting_zone = stl.STLTimedAlways(
        stl.STLAnd(p_inside_max_radius, p_outside_min_radius, interpolate=True),
        0.0,
        min_time_in_waiting_zone,
    )
    eventually_reach_waiting_zone_and_wait = stl.STLUntimedEventually(
        stay_inside_waiting_zone,
    )

    # Combine to mission requirement
    mission_1_spec = eventually_reach_target
    mission_2_spec = stl.STLAnd(
        mission_1_spec,
        eventually_reach_waiting_zone_and_wait,
        interpolate=True,
    )

    if mission_1:
        return mission_1_spec
    else:
        return mission_2_spec
