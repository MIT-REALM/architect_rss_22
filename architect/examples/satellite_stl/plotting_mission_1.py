import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns


def plot_comparison_mission_1():
    # Load the various results, being sure to add a column to represent the method
    results_df = pd.DataFrame()

    save_dir = "logs/satellite_stl/safety_and_goal_only/comparison"

    filename = f"{save_dir}/combined_milp.csv"
    mip_df = pd.read_csv(filename)
    mip_df["method"] = "MIP"
    results_df = results_df.append(mip_df, ignore_index=True)
    num_feasible = mip_df[mip_df.measurement == "Feasible"].value.sum()
    print(f"{num_feasible} out of 50 MIPs were feasible")

    filename = f"{save_dir}/combined_random_batch_1.csv"
    se_df = pd.read_csv(filename)
    se_df["method"] = "NLopt\n(n=1)"
    results_df = results_df.append(se_df, ignore_index=True)

    filename = f"{save_dir}/combined_random_batch_32.csv"
    b32_df = pd.read_csv(filename)
    b32_df["method"] = "NLopt\n(n=32)"
    mask = b32_df.measurement == "STL Robustness"
    print((b32_df[mask].value > 0).sum())
    results_df = results_df.append(b32_df, ignore_index=True)

    filename = f"{save_dir}/combined_random_batch_64.csv"
    b64_df = pd.read_csv(filename)
    b64_df["method"] = "NLopt\n(n=64)"
    results_df = results_df.append(b64_df, ignore_index=True)

    filename = f"{save_dir}/combined_counterexample_guided.csv"
    cg_df = pd.read_csv(filename)
    cg_df["method"] = "CG\n(ours)"
    mask = cg_df.measurement == "Final population size"
    print(
        (
            f"CG: min pop: {cg_df[mask].value.min()}, "
            f"median: {cg_df[mask].value.median()}, "
            f"max: {cg_df[mask].value.max()}"
        )
    )
    results_df = results_df.append(cg_df, ignore_index=True)

    # Plot
    sns.set_theme(context="poster", style="whitegrid")

    # Plot STL robustness
    fig = plt.figure(tight_layout=True, figsize=plt.figaspect(0.5))
    gs = gridspec.GridSpec(3, 2)

    rob_ax1 = fig.add_subplot(gs[:2, 0])
    rob_ax2 = fig.add_subplot(gs[2, 0])
    mask = results_df.measurement == "STL Robustness"
    sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=1, ax=rob_ax1)
    sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=1, ax=rob_ax2)

    rob_ax1.set_ylabel("STL Robustness")
    rob_ax1.set_xlabel("")
    rob_ax1.set_ylim([-5, 1.0])
    rob_ax2.set_ylabel("")
    rob_ax2.set_xlabel("")
    rob_ax2.set_ylim([-20, -8])

    # hide the spines between ax1 and ax2
    rob_ax1.spines.bottom.set_visible(False)
    rob_ax1.axes.get_xaxis().set_visible(False)
    rob_ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    rob_ax2.spines.top.set_visible(False)
    rob_ax2.xaxis.tick_bottom()

    # Make broken axis marks
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=24,
        linestyle="none",
        color="silver",
        mec="silver",
        mew=2,
        clip_on=False,
    )
    rob_ax1.plot([0, 1], [0, 0], transform=rob_ax1.transAxes, **kwargs)
    rob_ax2.plot([0, 1], [1, 1], transform=rob_ax2.transAxes, **kwargs)

    # Plot optimization time
    time_ax1 = fig.add_subplot(gs[0, 1])
    time_ax2 = fig.add_subplot(gs[1:, 1])
    fig.subplots_adjust(hspace=0.05)
    mask = results_df.measurement == "Optimization time (s)"
    sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=1, ax=time_ax1)
    sns.boxplot(x="method", y="value", data=results_df[mask], linewidth=1, ax=time_ax2)

    time_ax1.set_ylabel("Optimization time (s)")
    time_ax1.set_xlabel("")
    time_ax1.set_ylim([475, 550])
    time_ax2.set_title("")
    time_ax2.set_ylabel("")
    time_ax2.set_xlabel("")
    time_ax2.set_ylim([0, 150])

    # hide the spines between ax1 and ax2
    time_ax1.spines.bottom.set_visible(False)
    time_ax1.axes.get_xaxis().set_visible(False)
    time_ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    time_ax2.spines.top.set_visible(False)
    time_ax2.xaxis.tick_bottom()

    # Make broken axis marks
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=24,
        linestyle="none",
        color="silver",
        mec="silver",
        mew=2,
        clip_on=False,
    )
    time_ax1.plot([0, 1], [0, 0], transform=time_ax1.transAxes, **kwargs)
    time_ax2.plot([0, 1], [1, 1], transform=time_ax2.transAxes, **kwargs)

    plt.show()


if __name__ == "__main__":
    plot_comparison_mission_1()
