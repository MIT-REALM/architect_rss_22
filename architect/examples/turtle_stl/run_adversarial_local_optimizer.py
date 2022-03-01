import os
import sys

import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np

from architect.optimization import (
    AdversarialLocalOptimizer,
)
from architect.examples.turtle_stl.tbstl_design_problem import (
    make_tbstl_design_problem,
)
from architect.examples.turtle_stl.tbstl_stl_specification import (
    make_tbstl_rendezvous_specification,
)


def run_optimizer(seed: int = 0):
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(seed)

    # Make the design problem
    t_sim = 30.0
    dt = 0.1
    time_steps = int(t_sim // dt)
    prng_key, subkey = jax.random.split(prng_key)
    tbstl_design_problem = make_tbstl_design_problem(time_steps, dt, mission_1=False)

    # Create the optimizer
    ad_opt = AdversarialLocalOptimizer(tbstl_design_problem)

    # Optimize!
    prng_key, subkey = jax.random.split(prng_key)
    # start = time.perf_counter()
    print("==================================")
    dp_opt, ep_opt, cost, cost_gap, opt_time, t_jit, rounds, pop_size = ad_opt.optimize(
        subkey,
        disp=False,
        rounds=10,
        n_init=8,
        stopping_tolerance=0.1,
        maxiter=500,
        jit=True,
    )

    # Run a simulation for plotting the optimal solution
    state_trace = tbstl_design_problem.simulator(dp_opt, ep_opt)

    # Get the robustness of this solution
    stl_specification = make_tbstl_rendezvous_specification(mission_1=False)
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0])
    signal = jnp.vstack((t.reshape(1, -1), state_trace.T))
    robustness = stl_specification(signal)

    return [
        {"seed": seed, "measurement": "Cost", "value": cost},
        {"seed": seed, "measurement": "Cost gap", "value": cost_gap},
        {"seed": seed, "measurement": "Optimization time (s)", "value": opt_time},
        {"seed": seed, "measurement": "Compilation time (s)", "value": t_jit},
        {"seed": seed, "measurement": "STL Robustness", "value": robustness[1, 0]},
        {"seed": seed, "measurement": "Optimization rounds", "value": rounds + 1},
        {"seed": seed, "measurement": "Final population size", "value": pop_size},
    ], dp_opt


if __name__ == "__main__":
    results_df = pd.DataFrame()

    seed = int(sys.argv[1])
    print(f"Running with seed {seed}")

    results, solution = run_optimizer(seed)
    for packet in results:
        results_df = results_df.append(packet, ignore_index=True)

    # Make sure the given directory exists; create it if it does not
    save_dir = "logs/turtle_stl/all_constraints/comparison"
    os.makedirs(save_dir, exist_ok=True)

    # Save the results
    filename = f"{save_dir}/counterexample_guided_{seed}.csv"
    results_df.to_csv(filename, index=False)

    # Save the solution too
    save_dir = "logs/turtle_stl/all_constraints/solutions"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/counterexample_guided_{seed}.csv"
    np.savetxt(filename, np.array(solution), delimiter=",")
