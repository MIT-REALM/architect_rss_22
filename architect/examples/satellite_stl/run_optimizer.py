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
    dt = 0.1
    substeps = 5
    time_steps = int(t_sim // dt)
    specification_weight = 100.0
    prng_key, subkey = jax.random.split(prng_key)
    sat_design_problem = make_sat_design_problem(
        specification_weight, time_steps, dt, substeps
    )

    # Create the optimizer
    variance_weight = 0.1
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
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0])
    signal = stl.SampledSignal(t, state_trace)
    robustness = stl_specification(signal).x[0]

    # Plot the results
    fig = plt.figure(figsize=plt.figaspect(0.5))
    state_ax = fig.add_subplot(1, 2, 1, projection="3d")
    state_ax.plot3D(state_trace[:, 0], state_trace[:, 1], state_trace[:, 2])
    state_ax.plot3D(0.0, 0.0, 0.0, "ko")
    state_ax.set_title(f"Rendezvous. Total effort: {total_effort}")

    rob_ax = fig.add_subplot(1, 2, 2)
    rob_ax.plt(t, robustness)
    rob_ax.set_xlabel("t")
    rob_ax.set_ylabel("STL Robustness")

    plt.show()


if __name__ == "__main__":
    run_optimizer()
