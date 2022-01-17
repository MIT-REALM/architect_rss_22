import ast

import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from architect.examples.agv_localization.agv_simulator import navigation_function


def plot_estimated_state_trajectory():
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
    plt.contourf(X, Y, V, levels=10)

    plt.show()

    # Plot the observed navigation function over time
    V = df[df.measurement == "V"].value.apply(lambda x: ast.literal_eval(x))
    V = jnp.array(V)

    plt.plot(V)
    plt.show()


def plot_covariance_both():
    # Load data
    filename = "logs/agv_localization/hardware/initial_dt0p1_jan_14_2022.csv"
    df = pd.read_csv(filename)

    # Extract estimates and error covariance
    state_estimate_mean = df[df.measurement == "state_est_mean"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimate_mean = list(state_estimate_mean)
    state_estimate_mean = jnp.array(state_estimate_mean)

    state_estimate_cov = df[df.measurement == "state_est_cov"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimate_cov = list(state_estimate_cov)
    state_estimate_cov = jnp.array(state_estimate_cov)

    fig, ax1 = plt.subplots()

    # Plot through time
    i_to_show = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    i_to_show += [275, 300, 325]
    for i in i_to_show:
        mean_x, mean_y = state_estimate_mean[i, :2]
        cov = state_estimate_cov[i, :2, :2]  # project onto xy

        confidence_ellipse(mean_x, mean_y, cov, ax1, facecolor="pink", edgecolor="red")
    ax1.plot(
        state_estimate_mean[:, 0], state_estimate_mean[:, 1], c="red", label="Initial"
    )

    # Load data
    filename = "logs/agv_localization/hardware/optimized_dt0p1_jan_14_2022.csv"
    df = pd.read_csv(filename)

    # Extract estimates and error covariance
    state_estimate_mean = df[df.measurement == "state_est_mean"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimate_mean = list(state_estimate_mean)
    state_estimate_mean = jnp.array(state_estimate_mean)

    state_estimate_cov = df[df.measurement == "state_est_cov"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimate_cov = list(state_estimate_cov)
    state_estimate_cov = jnp.array(state_estimate_cov)

    # Plot through time
    for i in i_to_show:
        mean_x, mean_y = state_estimate_mean[i, :2]
        cov = state_estimate_cov[i, :2, :2]  # project onto xy

        confidence_ellipse(
            mean_x, mean_y, cov, ax1, facecolor="lightskyblue", edgecolor="darkblue"
        )
    ax1.plot(
        state_estimate_mean[:, 0],
        state_estimate_mean[:, 1],
        c="darkblue",
        label="Optimized",
    )

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.legend()
    ax1.set_aspect("equal")
    fig.tight_layout()
    plt.show()


def plot_covariance_optimized():
    # Load data
    filename = "logs/agv_localization/hardware/optimized_dt0p1_jan_14_2022.csv"
    df = pd.read_csv(filename)

    # Extract estimates and error covariance
    state_estimate_mean = df[df.measurement == "state_est_mean"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimate_mean = list(state_estimate_mean)
    state_estimate_mean = jnp.array(state_estimate_mean)

    state_estimate_cov = df[df.measurement == "state_est_cov"].value.apply(
        lambda x: ast.literal_eval(x)
    )
    state_estimate_cov = list(state_estimate_cov)
    state_estimate_cov = jnp.array(state_estimate_cov)

    fig, ax = plt.subplots()

    # Plot through time
    i_to_show = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    i_to_show += [275, 300, 325, 350, 375]
    for i in i_to_show:
        mean_x, mean_y = state_estimate_mean[i, :2]
        cov = state_estimate_cov[i, :2, :2]  # project onto xy

        confidence_ellipse(mean_x, mean_y, cov, ax, facecolor="pink", edgecolor="red")
    ax.plot(state_estimate_mean[:, 0], state_estimate_mean[:, 1], c="red")

    plt.show()


def confidence_ellipse(mean_x, mean_y, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Adapted from
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    mean_x, mean_y : floats giving the location of the ellipse center

    cov: covariance matrix to plot

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    plot_covariance_both()
