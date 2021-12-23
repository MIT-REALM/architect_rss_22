"""Dynamics information for a discrete-time Dubins car"""
import jax.numpy as jnp


def dubins_next_state(
    current_state: jnp.ndarray,
    control_input: jnp.ndarray,
    actuation_noise: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute the next state from the current state and control input.

    To make this function pure, we need to pass in all sources of randomness used.

    args:
        current_state: the (3,) array containing the current state
        control_input: the (2,) array containing the control input at this step
        actuation_noise: the (3,) array containing the actuation noise
        dt: the length of the discrete-time update (scalar)
    returns:
        the new state
    """
    # Update the state
    theta = current_state[2]
    v = control_input[0]
    w = control_input[1]
    next_state = current_state.at[0].add(dt * v * jnp.cos(theta + dt * w / 2))
    next_state = next_state.at[1].add(dt * v * jnp.sin(theta + dt * w / 2))
    next_state = next_state.at[2].add(dt * w)

    next_state = next_state + actuation_noise

    return next_state


def dubins_linearized_dynamics(
    current_state: jnp.ndarray,
    control_input: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Compute the linearization of the system dynamics for use in an EKF

    args:
        current_state: the (3,) array containing the current state
        control_input: the (2,) array containing the control input at this step
        dt: the length of the discrete-time update (scalar)
    returns:
        the (3, 3) state-to-state transfer matrix
    """
    theta = current_state[2]

    v = control_input[0]
    w = control_input[1]

    F = jnp.eye(3)
    F = F.at[0, 2].add(-dt * v * jnp.sin(theta + dt * w / 2))
    F = F.at[1, 2].add(dt * v * jnp.cos(theta + dt * w / 2))

    return F
