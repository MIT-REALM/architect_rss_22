import time

import jax
import arviz as az
import matplotlib.pyplot as plt

from architect.analysis import WorstCaseCostAnalyzer
from architect.examples.agv_localization.agv_design_problem import (
    make_agv_localization_design_problem_analysis,
)


def run_analysis():
    """Run design optimization and plot the results"""
    # Make the design problem
    T = 30
    dt = 0.5
    agv_design_problem = make_agv_localization_design_problem_analysis(T, dt)

    # # Set it with the optimal parameters found using `run_optimizer`
    # agv_design_problem.design_params.set_values(
    #     jnp.array([2.535058, 0.09306894, -1.6945883, -1.0, 0.0, -0.8280163])
    # )

    # Create the analyzer
    sample_size = 1000
    block_size = 1000
    worst_case_cost_analyzer = WorstCaseCostAnalyzer(
        agv_design_problem, sample_size, block_size
    )

    # Analyze!
    prng_key = jax.random.PRNGKey(0)
    start = time.perf_counter()
    summary, idata = worst_case_cost_analyzer.analyze(prng_key)
    end = time.perf_counter()
    print("==================================")
    print(
        f"Worst-case cost analysis of ({sample_size} blocks"
        f" of size {block_size}) took {end - start} s."
    )
    print("Summary:")
    print(summary)
    print("----------------------------------")

    # Plot the posterior distributions
    az.plot_trace(idata)
    plt.show()


if __name__ == "__main__":
    run_analysis()
