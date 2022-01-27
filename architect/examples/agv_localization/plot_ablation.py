import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_agv_ablation_ad_vs_df():
    sns.set_theme(context="talk", style="white", font_scale=2)
    # Load in the data from my lab notebook
    df = pd.DataFrame(
        [
            {
                "Method": "AD",
                "Metric": "Final Objective",
                "Value": 3.1450 + 0.1 * 0.1160,
            },
            {"Method": "AD", "Metric": "Expected Cost", "Value": 3.1450},
            {"Method": "AD", "Metric": "Cost Variance", "Value": 0.1160},
            {"Method": "AD", "Metric": "CPU Time (s)", "Value": 213.7},
            ######
            {
                "Method": "FD",
                "Metric": "Final Objective",
                "Value": 3.8996 + 0.1 * 0.3853,
            },
            {"Method": "FD", "Metric": "Expected Cost", "Value": 3.8996},
            {"Method": "FD", "Metric": "Cost Variance", "Value": 0.3853},
            {"Method": "FD", "Metric": "CPU Time (s)", "Value": 315.4},
        ]
    )

    # Plot
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(18, 8))

    # The left axis plots all cost information
    left_plot_mask = df["Metric"] != "CPU Time (s)"
    sns.barplot(x="Metric", y="Value", hue="Method", data=df[left_plot_mask], ax=axs[0])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")

    # The right axis plots time information
    right_plot_mask = df["Metric"] == "CPU Time (s)"
    sns.barplot(
        x="Metric", y="Value", hue="Method", data=df[right_plot_mask], ax=axs[1]
    )
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")
    axs[1].get_legend().remove()

    fig.tight_layout()

    plt.show()


def plot_agv_ablation_vr():
    sns.set_theme(context="talk", style="white", font_scale=2)
    # Load in the data from my lab notebook
    df = pd.DataFrame(
        [
            {
                "Method": "Variance-regularized",
                "Metric": "Expected Cost",
                "Value": 3.1450,
            },
            {
                "Method": "Variance-regularized",
                "Metric": "Cost Variance",
                "Value": 0.1160,
            },
            {
                "Method": "Variance-regularized",
                "Metric": "CPU Time (s)",
                "Value": 213.7,
            },
            ######
            {"Method": "Expectation only", "Metric": "Expected Cost", "Value": 3.8738},
            {"Method": "Expectation only", "Metric": "Cost Variance", "Value": 0.3866},
            {"Method": "Expectation only", "Metric": "CPU Time (s)", "Value": 117.8},
        ]
    )

    with sns.color_palette(sns.color_palette("pastel")):
        # Plot
        fig, axs = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(18, 8)
        )

        # The left axis plots all cost information
        left_plot_mask = df["Metric"] != "CPU Time (s)"
        sns.barplot(
            x="Metric", y="Value", hue="Method", data=df[left_plot_mask], ax=axs[0]
        )
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")

        # The right axis plots time information
        right_plot_mask = df["Metric"] == "CPU Time (s)"
        sns.barplot(
            x="Metric", y="Value", hue="Method", data=df[right_plot_mask], ax=axs[1]
        )
        axs[1].set_xlabel("")
        axs[1].set_ylabel("")
        axs[1].get_legend().remove()

        fig.tight_layout()

        plt.show()


if __name__ == "__main__":
    plot_agv_ablation_ad_vs_df()
    plot_agv_ablation_vr()
