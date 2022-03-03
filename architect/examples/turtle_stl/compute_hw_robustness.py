import ast

import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.examples.turtle_stl.tbstl_stl_specification import (
    make_tbstl_rendezvous_specification,
)


def compute_robustness():
    df = pd.read_csv("logs/turtle_stl/all_constraints/hw/tbstl_ros_controller_log.csv")
    mask = df.measurement == "state_est_mean"
    t = jnp.array(df[mask].time - df[mask].time.min())
    t = t - t.min()
    state_estimates = df[mask].value.apply(lambda x: ast.literal_eval(x))
    state_estimates = list(state_estimates)
    state_estimates = jnp.array(state_estimates)
    # Account for offset in map frame
    state_estimates = state_estimates.at[:, 1].add(2.75)

    # plt.plot(state_estimates[:, 0], state_estimates[:, 1])
    # plt.show()
    # plt.plot(t, state_estimates[:, 1])
    # plt.show()

    # Select only the relevant part of the trajectory
    mask = t < 31.0
    t = t[mask]
    state_estimates = state_estimates[mask]

    # Reduce temporal resolution
    t = t[::30]
    state_estimates = state_estimates[::30]

    # Get the robustness of this solution
    stl_specification = make_tbstl_rendezvous_specification(mission_1=False)
    signal = jnp.vstack((t.reshape(1, -1), state_estimates.T))  # type: ignore
    robustness = stl_specification(signal)
    plt.plot(robustness[0], robustness[1])
    plt.show()


if __name__ == "__main__":
    compute_robustness()
