import matplotlib.pyplot as plt
import seaborn as sns

from architect.examples.agv_localization.plot_gevd import plot_gevd


def plot_mam_gevd():
    sns.set_theme(context="talk", style="white", font_scale=1.5)

    # Upper bound (97%)
    (lb, ub) = (2.0, 60.0)
    (mu, xi, sigma) = (9.959, 0.325, 5.494)
    plot_gevd(
        mu,
        xi,
        sigma,
        lb,
        ub,
        color="darkblue",
        ls="-",
        lw=8,
        label=None,
        plot_97=True,
        metric="sensitivity",
    )

    plt.xlim([lb, ub])
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_mam_gevd()
