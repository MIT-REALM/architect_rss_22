"""Dynamics information for a discrete-time satellite, linearized in the CHW frame"""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# Define constant parameters
MU = 3.986e14  # Earth's gravitational parameter (m^3 / s^2)
A_GEO = 42164e3  # GEO semi-major axis (m)
A_LEO = 353e3  # GEO semi-major axis (m)
M_CHASER = 500  # chaser satellite mass
N = jnp.sqrt(MU / A_LEO ** 3)  # mean-motion parameter


@jax.jit
def linear_satellite_dt_AB(dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Define continuous-time A and B matrices
    A = jnp.zeros((6, 6))
    A = A.at[0, 3].set(1.0)
    A = A.at[1, 4].set(1.0)
    A = A.at[2, 5].set(1.0)
    A = A.at[3, 0].set(3 * N ** 2)
    A = A.at[3, 4].set(2 * N)
    A = A.at[4, 3].set(-2 * N)
    A = A.at[5, 2].set(-(N ** 2))

    B = jnp.zeros((6, 3))
    B = B.at[3, 0].set(1 / M_CHASER)
    B = B.at[4, 1].set(1 / M_CHASER)
    B = B.at[5, 2].set(1 / M_CHASER)

    # Get discrete-time versions using the matrix exponential
    M = jnp.zeros((9, 9))
    M = M.at[:6, :6].set(A)
    M = M.at[:6, 6:].set(B)
    F = jax.scipy.linalg.expm(dt * M)  # type: ignore
    A_dt: jnp.ndarray = F[:6, :6]
    B_dt: jnp.ndarray = F[:6, 6:]

    return A_dt, B_dt


@jax.jit
def linear_satellite_next_state(
    current_state: jnp.ndarray,
    control_input: jnp.ndarray,
    actuation_noise: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute the next state from the current state and control input.

    To make this function pure, we need to pass in all sources of randomness used.

    args:
        current_state: the (3,) array containing the current state
        control_input: the (3,) array containing the control input at this step
        actuation_noise: the (6,) array containing the actuation noise
        dt: the length of the discrete-time update (scalar)
    returns:
        the new state
    """
    # Extract states and controls
    px, _, pz, vx, vy, vz = current_state
    ux, uy, uz = control_input

    # Update the state.
    # Positions just integrate velocities
    next_state = current_state.at[0].add(dt * vx)
    next_state = next_state.at[1].add(dt * vy)
    next_state = next_state.at[2].add(dt * vz)

    # Velocities follow CHW dynamics (See Jewison & Erwin CDC 2016)
    # Note that the dynamics in z are decoupled from those in xy,
    # so we can also consider just the xy projection if we need to.
    dx2_dt2 = 2 * N * vy + 3 * N ** 2 * px + ux / M_CHASER
    dy2_dt2 = -2 * N * vx + uy / M_CHASER
    dz2_dt2 = -(N ** 2) * pz + uz / M_CHASER
    next_state = next_state.at[3].add(dt * dx2_dt2)
    next_state = next_state.at[4].add(dt * dy2_dt2)
    next_state = next_state.at[5].add(dt * dz2_dt2)

    next_state = next_state + actuation_noise

    return next_state


@partial(jax.jit, static_argnames=["substeps"])
def linear_satellite_next_state_substeps(
    current_state: jnp.ndarray,
    control_input: jnp.ndarray,
    actuation_noise: jnp.ndarray,
    dt: float,
    substeps: int = 1,
) -> jnp.ndarray:
    """Compute the next state from the current state and control input.

    To make this function pure, we need to pass in all sources of randomness used.

    args:
        current_state: the (3,) array containing the current state
        control_input: the (3,) array containing the control input at this step
        actuation_noise: the (6,) array containing the actuation noise
        dt: the length of the discrete-time update (scalar)
        substeps: how many smaller updates to break this interval into
    returns:
        the new state
    """
    # Update the dynamics on smaller subdivisions of this interval, using a zero-order
    # hold for control inputs
    state = current_state
    substep_dt = dt / substeps
    for _ in range(substeps):
        state = linear_satellite_next_state(
            state, control_input, 0 * actuation_noise, substep_dt
        )

    # Add the actuation noise at the end
    return state + actuation_noise


if __name__ == "__main__":
    # Test the dynamics by simulating from a point
    start_point = jnp.zeros((6,)) + 1.0

    # Don't apply any control inputs or disturbance, just
    # observe the drift dynamics
    t_sim = 10.0
    dt = 0.2
    T = int(t_sim // dt)
    sim_trace = jnp.zeros((T, 6)).at[0].set(start_point)
    control_input = jnp.zeros(3)
    actuation_noise = jnp.zeros(6)
    for i in range(1, T):
        sim_trace = sim_trace.at[i].set(
            linear_satellite_next_state(
                sim_trace[i - 1], control_input, actuation_noise, dt
            )
        )

    print(sim_trace)

    # Turns out (from observation and conversations with Sydney) that these dynamics are
    # pretty simple. The intuition is that the wackiness of orbital dynamics decreases
    # as you move further out, so in GEO this will be a slightly perturbed double
    # integrator, but in LEO it is weirder. Let's embrace the weirdness and work in LEO.
    ax = plt.axes(projection="3d")
    ax.plot3D(sim_trace[:, 0], sim_trace[:, 1], sim_trace[:, 2])
    plt.show()
