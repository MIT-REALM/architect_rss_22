import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from architect.examples.agv_localization.plot_gevd import plot_gevd, gevd_cdf


def plot_ks():
    sns.set_theme(context="talk", style="white", font_scale=2)

    # read the data
    df = pd.read_csv("logs/agv_localization/worst_case_cost/block_maxes.csv")
    block_maxes = df["gevd"].to_numpy()

    # Plot the ECDF
    block_maxes = np.sort(block_maxes)
    y = np.arange(len(block_maxes)) / float(len(block_maxes))
    plt.plot(block_maxes, y, color="red", label="Empirical CDF", linewidth=2)

    lb = block_maxes.min()
    ub = block_maxes.max()

    # Lower bound (3%)
    (mu, xi, sigma) = (0.173, -0.023, 0.008)
    plot_gevd(mu, xi, sigma, lb, ub, plt.gca(), "skyblue", "-", r"3% GEVD CDF")
    cdf = lambda x: gevd_cdf(x, mu, xi, sigma)
    ks_result = stats.kstest(block_maxes, cdf, alternative="less")
    print(f"Lower bound less-than KS test results: {ks_result}")

    # # Mean
    # (mu, xi, sigma) = (0.174, 0.017, 0.009)
    # plot_gevd(mu, xi, sigma, lb, ub, plt.gca(), "royalblue", "--", r"Mean GEVD CDF")
    # cdf = lambda x: gevd_cdf(x, mu, xi, sigma)
    # ks_result = stats.kstest(block_maxes, cdf, alternative="two-sided")
    # print(f"Mean two-sided KS test results: {ks_result}")

    # Upper bound (97%)
    (mu, xi, sigma) = (0.174, 0.059, 0.009)
    plot_gevd(mu, xi, sigma, lb, ub, plt.gca(), "darkblue", "-", r"97% GEVD CDF")
    cdf = lambda x: gevd_cdf(x, mu, xi, sigma)
    ks_result = stats.kstest(block_maxes, cdf, alternative="greater")
    print(f"Upper bound greater-than KS test results: {ks_result}")

    plt.xlim([lb, 0.22])

    plt.show()


if __name__ == "__main__":
    plot_ks()
