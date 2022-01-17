import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.optimization import (
    VarianceRegularizedOptimizerAD,
    VarianceRegularizedOptimizerCMA,
)
from architect.examples.agv_localization.agv_design_problem import (
    make_agv_localization_design_problem,
)
from architect.examples.agv_localization.agv_simulator import navigation_function


def run_optimizer():
    """Run design optimization and plot the results"""
    # Make the design problem
    T = 30
    dt = 0.5
    agv_design_problem = make_agv_localization_design_problem(T, dt)

    # Create the optimizer
    variance_weight = 0.1
    sample_size = 512
    vr_opt = VarianceRegularizedOptimizerAD(
        agv_design_problem, variance_weight, sample_size
    )

    # Optimize!
    prng_key = jax.random.PRNGKey(0)
    success, msg, dp_opt, cost_mean, cost_var = vr_opt.optimize(prng_key, disp=True)
    print("==================================")
    print(f"Success? {success}! Message: {msg}")
    print("----------------------------------")
    print(f"Optimal design parameters:\n{dp_opt}")
    print(f"Optimal mean cost: {cost_mean}")
    print(f"Optimal cost variance: {cost_var}")

    # Run a simulation for plotting the optimal solution
    (
        true_states,
        state_estimates,
        state_estimate_covariances,
        V_trace,
    ) = agv_design_problem.simulator(
        dp_opt, agv_design_problem.exogenous_params.sample(prng_key)
    )

    # Extract some of the optimal values
    beacon_locations = dp_opt[2:]
    n_beacons = beacon_locations.shape[0] // 2
    beacon_locations = beacon_locations.reshape(n_beacons, 2)

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax = axs[0]
    x = jnp.linspace(-3, 0, 100)
    y = jnp.linspace(-1, 1, 100)
    X, Y = jnp.meshgrid(x, y)
    XY = jnp.stack((X, Y)).reshape(2, 10000).T
    V = jax.vmap(navigation_function, in_axes=0)(XY).reshape(100, 100)
    ax.contourf(X, Y, V, levels=10)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.plot(beacon_locations[:, 0], beacon_locations[:, 1], "r^", label="Beacons")
    ax.plot(
        true_states[:, 0],
        true_states[:, 1],
        "silver",
        marker="x",
        label="True trajectory",
    )
    ax.plot(
        state_estimates[:, 0],
        state_estimates[:, 1],
        "cyan",
        marker="x",
        label="Estimated trajectory",
    )
    _ = ax.legend()

    # Plot the true navigation function value over time
    ax = axs[1]
    ax.plot(jnp.arange(0, T, dt), V_trace)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Navigation function value")

    # Plot the estimation error over time
    ax = axs[2]
    ax.plot(
        jnp.arange(0, T, dt), jnp.linalg.norm(true_states - state_estimates, axis=-1)
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$||x - \hat{x}||$")
    plt.show()


def run_optimizer_cma():
    """Run CMA design optimization and plot the results"""
    # Make the design problem
    T = 30
    dt = 0.5
    agv_design_problem = make_agv_localization_design_problem(T, dt)

    # Create the optimizer
    variance_weight = 0.1
    sample_size = 512
    vr_opt = VarianceRegularizedOptimizerCMA(
        agv_design_problem, variance_weight, sample_size
    )

    # Optimize!
    prng_key = jax.random.PRNGKey(0)
    dp_opt, cost_mean, cost_var = vr_opt.optimize(prng_key, budget=200, verbosity=2)
    print("==================================")
    print("Success? who knows! hard to tell with zero-order methods")
    print("----------------------------------")
    print(f"Optimal design parameters:\n{dp_opt}")
    print(f"Optimal mean cost: {cost_mean}")
    print(f"Optimal cost variance: {cost_var}")


if __name__ == "__main__":
    run_optimizer()
    # run_optimizer_cma()
