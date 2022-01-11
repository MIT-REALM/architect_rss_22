import ast

import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp

from architect.examples.agv_localization.agv_simulator import navigation_function


if __name__ == "__main__":
    filename = "agv_controller_log.csv"
    df = pd.read_csv(filename)
    state_estimates = df[df.measurement == "state_est_mean"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimates = list(state_estimates)
    state_estimates = jnp.array(state_estimates)

    plt.plot(state_estimates[:, 0], state_estimates[:, 1], "o-")

    # Overlay the navigation function
    x = jnp.linspace(-3, 0, 100)
    y = jnp.linspace(-1, 1, 100)

    X, Y = jnp.meshgrid(x, y)
    XY = jnp.stack((X, Y)).reshape(2, 10000).T

    V = jax.vmap(navigation_function, in_axes=0)(XY).reshape(100, 100)
    contours = plt.contourf(X, Y, V, levels=10)

    plt.show()

    # Plot the observed navigation function over time
    V = df[df.measurement == "V"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    V = jnp.array(V)

    plt.plot(V)
    plt.show()
