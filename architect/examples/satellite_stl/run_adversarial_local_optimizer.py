import os
import sys
import time

import pandas as pd
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from architect.optimization import (
    AdversarialLocalOptimizer,
)
from architect.examples.satellite_stl.sat_design_problem import (
    make_sat_design_problem,
)
from architect.examples.satellite_stl.sat_stl_specification import (
    make_sat_rendezvous_specification,
)


def run_optimizer(seed: int = 0):
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(seed)

    # Make the design problem
    t_sim = 200.0
    dt = 2.0
    time_steps = int(t_sim // dt)
    specification_weight = 2e4
    prng_key, subkey = jax.random.split(prng_key)
    sat_design_problem = make_sat_design_problem(specification_weight, time_steps, dt)

    # Create the optimizer
    ad_opt = AdversarialLocalOptimizer(sat_design_problem)

    # Optimize!
    prng_key, subkey = jax.random.split(prng_key)
    # start = time.perf_counter()
    print("==================================")
    dp_opt, ep_opt, cost, cost_gap, opt_time, t_jit, rounds, pop_size = ad_opt.optimize(
        subkey,
        disp=False,
        rounds=1,
        n_init=1,
        stopping_tolerance=0.1,
        maxiter=500,
        jit=True,
    )
    # end = time.perf_counter()
    # jnp.set_printoptions(threshold=sys.maxsize)
    # print(f"Optimal design parameters:\n{dp_opt}")
    # print(f"Optimal exogenous parameters:\n{ep_opt}")
    # print(f"Optimal cost: {cost}")
    # print(f"Optimization took {end - start} s.")

    # Run a simulation for plotting the optimal solution
    state_trace, total_effort = sat_design_problem.simulator(dp_opt, ep_opt)

    # Get the robustness of this solution
    stl_specification = make_sat_rendezvous_specification()
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0])
    signal = jnp.vstack((t.reshape(1, -1), state_trace.T))
    robustness = stl_specification(signal)

    # # Plot the results
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # # Trajectory in state space
    # state_ax = fig.add_subplot(2, 2, 1, projection="3d")
    # state_ax.plot3D(state_trace[:, 0], state_trace[:, 1], state_trace[:, 2])
    # state_ax.plot3D(0.0, 0.0, 0.0, "ko")
    # state_ax.set_title(f"Rendezvous. Total effort: {total_effort}")
    # print(f"Rendezvous. Total effort: {total_effort:.2f}")

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # # Trajectory in state space
    # state_ax = fig.add_subplot(1, 1, 1, projection="3d")
    # state_ax.plot3D(
    #     state_trace[:, 0],
    #     state_trace[:, 1],
    #     state_trace[:, 2],
    #     label="Chaser trajectory",
    # )
    # state_ax.plot3D(0.0, 0.0, 0.0, "ko", label="Target")
    # state_ax.legend()
    # # state_ax.set_title(f"Rendezvous. Total effort: {total_effort}")
    # print(f"Rendezvous. Total effort: {total_effort:.2f}")
    # plt.show()

    # # Robustness trace over time
    # rob_ax = fig.add_subplot(2, 2, 2)
    # rob_ax.plot(robustness[0], robustness[1])
    # rob_ax.set_xlabel("t")
    # rob_ax.set_ylabel("STL Robustness")
    # rob_ax.set_title(f"Robustness at start: {robustness[1, 0]}")
    # print(f"Robustness at start: {robustness[1, 0]:.4f}")

    # # Plot distance and speed on the same axis to see if safety constraints are met
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # safety_ax = fig.add_subplot(1, 1, 1)
    # safety_ax.plot(t, jnp.linalg.norm(state_trace[:, :3], axis=-1), color="blue")
    # safety_ax.plot(t, 0 * t, "k-")
    # safety_ax.plot(t, 0 * t + 2.0, "b:")
    # # safety_ax.set_ylim([0.0, 10.0])
    # safety_ax = safety_ax.twinx()
    # safety_ax.plot(t, jnp.linalg.norm(state_trace[:, 3:], axis=-1), color="red")
    # safety_ax.plot(t, 0 * t + 0.1, "r:")

    # safety_ax.legend()
    # safety_ax.set_xlabel("t")
    # plt.show()

    return [
        {"seed": seed, "measurement": "Cost", "value": cost},
        {"seed": seed, "measurement": "Cost gap", "value": cost_gap},
        {"seed": seed, "measurement": "Optimization time (s)", "value": opt_time},
        {"seed": seed, "measurement": "Compilation time (s)", "value": t_jit},
        {"seed": seed, "measurement": "STL Robustness", "value": robustness[1, 0]},
        {"seed": seed, "measurement": "Total effort (N-s)", "value": total_effort},
        {"seed": seed, "measurement": "Optimization rounds", "value": rounds + 1},
        {"seed": seed, "measurement": "Final population size", "value": pop_size},
    ]


if __name__ == "__main__":
    results_df = pd.DataFrame()

    seed = int(sys.argv[1])
    print(f"Running with seed {seed}")

    results = run_optimizer(seed)
    for packet in results:
        results_df = results_df.append(packet, ignore_index=True)

    # Make sure the given directory exists; create it if it does not
    save_dir = "logs/satellite_stl/all_constraints/comparison"
    os.makedirs(save_dir, exist_ok=True)

    # Save the results
    filename = f"{save_dir}/random_batch_1_{seed}.csv"
    results_df.to_csv(filename, index=False)
