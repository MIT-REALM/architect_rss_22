import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.optimization import VarianceRegularizedOptimizerAD
from architect.examples.multi_agent_manipulation.mam_design_problem import (
    make_mam_design_problem,
)
from architect.examples.multi_agent_manipulation.mam_plotting import (
    plot_box_trajectory,
    plot_turtle_trajectory,
    make_box_patches,
)


def run_optimizer():
    """Run design optimization and plot the results"""
    prng_key = jax.random.PRNGKey(0)

    # Make the design problem
    layer_widths = (2 * 3 + 3, 32, 2 * 2)
    dt = 0.01
    prng_key, subkey = jax.random.split(prng_key)
    mam_design_problem = make_mam_design_problem(layer_widths, dt, subkey)

    # Create the optimizer
    variance_weight = 0.1
    sample_size = 512
    vr_opt = VarianceRegularizedOptimizerAD(
        mam_design_problem, variance_weight, sample_size
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
    exogenous_sample = mam_design_problem.exogenous_params.sample(prng_key)
    (turtle_states, box_states) = mam_design_problem.simulator(dp_opt, exogenous_sample)

    # Plot the results
    plot_box_trajectory(box_states, 0.5, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 0, :], 0.1, 20, plt.gca())
    plot_turtle_trajectory(turtle_states[:, 1, :], 0.1, 20, plt.gca())
    desired_box_pose = exogenous_sample[4:7]
    make_box_patches(desired_box_pose, 1.0, 0.5, plt.gca(), hatch=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    run_optimizer()
