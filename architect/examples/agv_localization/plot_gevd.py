import matplotlib.pyplot as plt
import numpy as np


def gevd_cdf(z, mu, xi, sigma):
    """Compute the CDF of the GEVD with the given parameters

    (i.e. probability that max <= z)
    """
    return np.exp(-((1 + xi * (z - mu) / sigma) ** (-1 / xi)))


def plot_gevd(
    mu,
    xi,
    sigma,
    min_z: float = 0.0,
    max_z: float = 10.0,
    ax=None,
    color="darkblue",
    label=None,
):
    """Plot the generalized extreme value distribution PDF with the given parameters"""
    resolution = int(1e3)
    z = np.linspace(min_z, max_z, resolution)
    G = gevd_cdf(z, mu, xi, sigma)

    # Get the 97% CI
    z_97 = z[np.where(G <= 0.97)[0].max()]

    # Plot the CDF
    if ax is None:
        fig, ax = plt.subplots()

    if label is not None:
        ax.plot(z, G, linewidth=3, c=color, label=label)
    else:
        ax.plot(z, G, linewidth=3, c=color)

    # Plot the confidence interval
    ax.plot(
        [z_97, z_97],
        [0.0, 0.97],
        ":",
        label=f"97% Confidence = {np.round(z_97, 2)}",
        c=color,
    )

    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$G(z) = \rm{Pr}[$max error $\leq z]$")
    ax.set_ylim([-0.05, 1.1])
    ax.legend(loc="lower right")

    return ax


if __name__ == "__main__":
    ax = plot_gevd(0.811, 0.105, 0.228, 0.1, 2.0, None, "red", "Initial")
    ax = plot_gevd(0.174, 0.059, 0.009, 0.1, 2.0, ax, "darkblue", "Optimized")

    plt.show()
