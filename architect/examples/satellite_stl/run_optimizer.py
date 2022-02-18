import sys

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from architect.optimization import (
    VarianceRegularizedOptimizerAD,
)
import architect.components.specifications.stl as stl
from architect.examples.satellite_stl.sat_design_problem import (
    make_sat_design_problem,
)
from architect.examples.satellite_stl.sat_stl_specification import (
    make_sat_rendezvous_specification,
)


def run_optimizer():
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(0)

    # Make the design problem
    t_sim = 60.0
    dt = 0.2
    substeps = 10
    time_steps = int(t_sim // dt)
    specification_weight = 1e5
    prng_key, subkey = jax.random.split(prng_key)
    sat_design_problem = make_sat_design_problem(
        specification_weight, time_steps, dt, substeps
    )

    # Run a simulation for plotting the optimal solution
    exogenous_sample = sat_design_problem.exogenous_params.sample(prng_key)
    dp_init = sat_design_problem.design_params.get_values()
    state_trace, total_effort = sat_design_problem.simulator(dp_init, exogenous_sample)

    # Get the robustness of this solution
    stl_specification = make_sat_rendezvous_specification()
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0])
    signal = jnp.vstack((t.reshape(1, -1), state_trace.T))
    robustness = stl_specification(signal)

    # Plot the results
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Trajectory in state space
    state_ax = fig.add_subplot(2, 2, 1, projection="3d")
    state_ax.plot3D(state_trace[:, 0], state_trace[:, 1], state_trace[:, 2])
    state_ax.plot3D(0.0, 0.0, 0.0, "ko")
    state_ax.set_title(f"Rendezvous. Total effort: {total_effort}")

    # Robustness trace over time
    rob_ax = fig.add_subplot(2, 2, 2)
    rob_ax.plot(robustness[0], robustness[1])
    rob_ax.set_xlabel("t")
    rob_ax.set_ylabel("STL Robustness")

    # Plot distance and speed on the same axis to see if safety constraints are met
    safety_ax = fig.add_subplot(2, 2, 4)
    safety_ax.plot(t, jnp.linalg.norm(state_trace[:, :3], axis=-1), label="Distance")
    safety_ax.plot(t, jnp.linalg.norm(state_trace[:, 3:], axis=-1), label="Speed")
    safety_ax.plot(t, 0 * t, "k-")
    safety_ax.plot(t, 0 * t + 0.1, "k--", label="Docking Limit")
    safety_ax.plot(t, 0 * t + 2.0, "k:", label="Waiting Limit")
    safety_ax.set_ylim([0.0, 10.0])
    safety_ax.legend()
    safety_ax.set_xlabel("t")

    plt.show()

    # Create the optimizer
    variance_weight = 0.0
    sample_size = 128
    vr_opt = VarianceRegularizedOptimizerAD(
        sat_design_problem, variance_weight, sample_size
    )

    # Optimize!
    prng_key, subkey = jax.random.split(prng_key)
    success, msg, dp_opt, cost_mean, cost_var = vr_opt.optimize(subkey, disp=True)
    print("==================================")
    print(f"Success? {success}! Message: {msg}")
    print("----------------------------------")
    jnp.set_printoptions(threshold=sys.maxsize)
    print(f"Optimal design parameters:\n{dp_opt}")
    print(f"Optimal mean cost: {cost_mean}")
    print(f"Optimal cost variance: {cost_var}")

    # Run a simulation for plotting the optimal solution
    exogenous_sample = sat_design_problem.exogenous_params.sample(prng_key)
    state_trace, total_effort = sat_design_problem.simulator(dp_opt, exogenous_sample)

    # Get the robustness of this solution
    stl_specification = make_sat_rendezvous_specification()
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0]).reshape(1, -1)
    signal = jnp.vstack((t, state_trace.T))
    robustness = stl_specification(signal)

    # Plot the results
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Trajectory in state space
    state_ax = fig.add_subplot(2, 2, 1, projection="3d")
    state_ax.plot3D(state_trace[:, 0], state_trace[:, 1], state_trace[:, 2])
    state_ax.plot3D(0.0, 0.0, 0.0, "ko")
    state_ax.set_title(f"Rendezvous. Total effort: {total_effort}")

    # Robustness trace over time
    rob_ax = fig.add_subplot(2, 2, 2)
    rob_ax.plot(robustness[0], robustness[1])
    rob_ax.set_xlabel("t")
    rob_ax.set_ylabel("STL Robustness")

    # Plot distance and speed on the same axis to see if safety constraints are met
    safety_ax = fig.add_subplot(2, 2, 4)
    safety_ax.plot(t, jnp.linalg.norm(state_trace[:, :3], axis=-1), label="Distance")
    safety_ax.plot(t, jnp.linalg.norm(state_trace[:, 3:], axis=-1), label="Speed")
    safety_ax.plot(t, 0 * t, "k-")
    safety_ax.plot(t, 0 * t + 0.1, "k--", label="Docking Limit")
    safety_ax.plot(t, 0 * t + 2.0, "k:", label="Waiting Limit")
    safety_ax.set_ylim([0.0, 10.0])
    safety_ax.legend()
    safety_ax.set_xlabel("t")

    plt.show()


if __name__ == "__main__":
    run_optimizer()
