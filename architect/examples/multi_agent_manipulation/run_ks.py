import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from architect.examples.agv_localization.plot_gevd import plot_gevd, gevd_cdf


def plot_ks():
    sns.set_theme(context="talk", style="white", font_scale=2)

    # read the data
    df = pd.read_csv("logs/multi_agent_manipulation/sensitivity/block_maxes.csv")
    block_maxes = df["gevd"].to_numpy()

    # Plot the ECDF
    block_maxes = np.sort(block_maxes)
    y = np.arange(len(block_maxes)) / float(len(block_maxes))
    plt.plot(block_maxes, y, color="red", label="Empirical CDF", linewidth=2)

    lb = block_maxes.min()
    ub = block_maxes.max()

    # Lower bound (3%)
    (mu, xi, sigma) = (9.527, 0.253, 5.125)
    plot_gevd(
        mu,
        xi,
        sigma,
        lb,
        ub,
        plt.gca(),
        "skyblue",
        "-",
        r"3% GEVD CDF",
        metric="sensitivity",
    )
    cdf = lambda x: gevd_cdf(x, mu, xi, sigma)
    ks_result = stats.kstest(block_maxes, cdf, alternative="less")
    print(f"Lower bound less-than KS test results: {ks_result}")

    # Upper bound (97%)
    (mu, xi, sigma) = (9.959, 0.325, 5.494)
    plot_gevd(
        mu,
        xi,
        sigma,
        lb,
        ub,
        plt.gca(),
        "darkblue",
        "-",
        r"97% GEVD CDF",
        metric="sensitivity",
    )
    cdf = lambda x: gevd_cdf(x, mu, xi, sigma)
    ks_result = stats.kstest(block_maxes, cdf, alternative="greater")
    print(f"Upper bound greater-than KS test results: {ks_result}")

    plt.xlim([lb, 60])

    plt.show()


if __name__ == "__main__":
    plot_ks()
