import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_comparison():
    # Load the various results, being sure to add a column to represent the method
    results_df = pd.DataFrame()

    save_dir = "logs/satellite_stl/safety_and_goal_only/comparison"

    filename = f"{save_dir}/combined_random_batch_1.csv"
    se_df = pd.read_csv(filename)
    se_df["method"] = "NLopt\n(n=1)"
    results_df = results_df.append(se_df, ignore_index=True)

    filename = f"{save_dir}/combined_random_batch_32.csv"
    b32_df = pd.read_csv(filename)
    b32_df["method"] = "NLopt\n(n=32)"
    results_df = results_df.append(b32_df, ignore_index=True)

    filename = f"{save_dir}/combined_random_batch_64.csv"
    b64_df = pd.read_csv(filename)
    b64_df["method"] = "NLopt\n(n=64)"
    results_df = results_df.append(b64_df, ignore_index=True)

    filename = f"{save_dir}/combined_counterexample_guided.csv"
    cg_df = pd.read_csv(filename)
    cg_df["method"] = "Counter-\n example\n guided"
    mask = cg_df.measurement == "Optimization time (s)"
    print(cg_df[mask].mean())
    results_df = results_df.append(cg_df, ignore_index=True)

    # Plot
    sns.set_theme(context="poster", style="whitegrid")

    # Plot STL robustness
    fig, axs = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    mask = results_df.measurement == "STL Robustness"
    sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=2, ax=axs[0])

    axs[0].set_ylabel("STL Robustness")
    axs[0].set_xlabel("")

    # Plot optimization time
    mask = results_df.measurement == "Optimization time (s)"
    sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=2, ax=axs[1])

    axs[1].set_ylabel("Optimization time (s)")
    axs[1].set_xlabel("")

    # fig = plt.gcf()
    # fig.set_size_inches(12, 8)
    fig.tight_layout()
    plt.show()

    # # Plot optimization cost
    # mask = results_df.measurement == "Cost"
    # sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=2)

    # ax = plt.gca()
    # ax.set_ylabel("Cost")
    # ax.set_xlabel("")

    # fig = plt.gcf()
    # fig.set_size_inches(12, 8)
    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    plot_comparison()
