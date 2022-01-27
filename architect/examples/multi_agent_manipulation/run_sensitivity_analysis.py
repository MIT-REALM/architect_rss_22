import time

import jax
import jax.numpy as jnp
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from architect.analysis import SensitivityAnalyzer
from architect.examples.multi_agent_manipulation.mam_design_problem import (
    make_mam_design_problem,
    make_mam_design_problem_friction_only,
)


def run_analysis():
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(0)

    # Make the design problem
    layer_widths = (2 * 3 + 3, 32, 2 * 2)
    dt = 0.01
    prng_key, subkey = jax.random.split(prng_key)
    mam_design_problem = make_mam_design_problem(layer_widths, dt, subkey)

    logfile = (
        "logs/multi_agent_manipulation/real_turtle_dimensions/"
        "design_optimization_512_samples_0p5x0p5xpi_4_target_"
        "9x32x4_network_spline_1e-1_variance_weight_solution.csv"
    )
    mam_design_problem.design_params.set_values(
        jnp.array(
            np.loadtxt(
                logfile,
                delimiter=",",
            )
        )
    )

    # Create the analyzer
    sample_size = 3000
    stride_length = 10
    block_size = 1000
    sensitivity_analyzer = SensitivityAnalyzer(
        mam_design_problem,
        sample_size,
        block_size,
        stride_length=stride_length,
    )

    # Analyze!
    start = time.perf_counter()
    summary, idata = sensitivity_analyzer.analyze(prng_key)
    end = time.perf_counter()
    print("==================================")
    print(
        f"Sensitivity analysis of ({sample_size} blocks"
        f" of size {block_size}) took {end - start} s."
    )
    print("Summary:")
    print(summary)
    print("----------------------------------")

    # Save data
    idata.observed_data.to_pandas().to_csv(
        "logs/multi_agent_manipulation/sensitivity/block_maxes.csv"
    )

    # Plot the posterior distributions
    az.plot_trace(idata)
    plt.show()


def run_analysis_friction_only():
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(0)

    # Make the design problem
    layer_widths = (2 * 3 + 3, 32, 2 * 2)
    dt = 0.01
    prng_key, subkey = jax.random.split(prng_key)
    mam_design_problem = make_mam_design_problem_friction_only(layer_widths, dt, subkey)

    logfile = (
        "logs/multi_agent_manipulation/real_turtle_dimensions/"
        "design_optimization_512_samples_0p5x0p5xpi_4_target_"
        "9x32x4_network_spline_1e-1_variance_weight_solution.csv"
    )
    mam_design_problem.design_params.set_values(
        jnp.array(
            np.loadtxt(
                logfile,
                delimiter=",",
            )
        )
    )

    # Create the analyzer
    sample_size = 3000
    stride_length = 10
    block_size = 1000
    sensitivity_analyzer = SensitivityAnalyzer(
        mam_design_problem,
        sample_size,
        block_size,
        stride_length=stride_length,
    )

    # Analyze!
    start = time.perf_counter()
    summary, idata = sensitivity_analyzer.analyze(prng_key)
    end = time.perf_counter()
    print("==================================")
    print(
        f"Sensitivity analysis of ({sample_size} blocks"
        f" of size {block_size}) took {end - start} s."
    )
    print("Summary:")
    print(summary)
    print("----------------------------------")

    # Save data
    idata.observed_data.to_pandas().to_csv(
        "logs/multi_agent_manipulation/sensitivity/block_maxes.csv"
    )

    # Plot the posterior distributions
    az.plot_trace(idata)
    plt.show()


if __name__ == "__main__":
    # run_analysis()
    run_analysis_friction_only()
