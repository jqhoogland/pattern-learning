from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, colors, gridspec
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from patterns.sweep import exp_filter, extract_run, extract_slice, get_pivot

BLUE, RED = sns.color_palette()[0], sns.color_palette()[3]
BLUES = sns.dark_palette("#79C", as_cmap=True)
REDS = sns.dark_palette((20, 60, 50), input="husl", as_cmap=True)


# Heatmap
def create_heatmap(
    x,
    y,
    z,
    ax,
    smooth: Union[bool, float] = False,
    cmap="inferno",
    log_x: bool = True,
    log_y: bool = True,
    log_z: bool = True,
    metric_label: str = "",
    title: str = "",
):
    X, Y = np.meshgrid(x, y)

    if smooth:
        z = exp_filter(z, sigma=smooth)

    if log_z:
        mesh = ax.pcolormesh(X, Y, z, cmap=cmap, norm=LogNorm())
    else:
        mesh = ax.pcolormesh(X, Y, z, cmap=cmap)

    if log_y:
        ax.set_yscale("log")

    if log_x:
        ax.set_xscale("log")

    ax.set_title(metric_label)
    ax.set_xlabel(title)
    ax.set_ylabel("Steps")
    ax.set_ylim(y.max(), y.min())
    yticks = [10**i for i in range(0, int(np.floor(np.log10(y.max()))))]

    if y.max() not in yticks:
        yticks.append(y.max())

    ax.set_yticks(yticks)
    ax.set_xlim(x[0], x[-1])
    ax.invert_yaxis()

    return mesh

def plot(
    df: pd.DataFrame,
    smooth: Union[bool, float] = False,
    log_loss=True,
    cmap="inferno",
    titles: Optional[dict] = None,
    unique_col: str = "weight_decay",
    log_x: bool = True,
    log_y: bool = True,
    title: str = "",
    columns=["train/loss", "test/loss", "train/acc", "test/acc"],
):
    # Set up the subplot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    pivot_table = get_pivot(
        df, unique_col, reindex=True, interpolate=True, columns=columns
    )
    unique_vals = sorted(df[unique_col].unique())

    # Loop through each column and plot the heatmap
    for i, column in enumerate(columns):
        row = i // 2
        col = i % 2
        ax = axes[row][col]

        # Create a meshgrid for the x and y edges
        X, Y = np.meshgrid(unique_vals, pivot_table[column].index)

        # Apply a Gaussian filter
        data = pivot_table[column].values

        if smooth:
            data = gaussian_filter(pivot_table[column].values, sigma=float(smooth))

        if "loss" in column and log_loss:
            mesh = ax.pcolormesh(X, Y, data, cmap=cmap, norm=LogNorm())
        else:
            mesh = ax.pcolormesh(X, Y, data, cmap=cmap)

        subtitle = titles.get(column, column) if titles else column
        ax.set_title(subtitle)
        ax.set_xlabel(title)
        ax.set_ylabel("Steps")

        if log_y:
            ax.set_yscale("log")

        if log_x:
            ax.set_xscale("log")

        ax.set_ylim(df._step.max(), df._step.min())
        ax.set_yticks([1, 10, 100, 1000, 10000, df._step.max()])
        ax.set_xlim(unique_vals[0], unique_vals[-1])
        ax.invert_yaxis()

        # Add a colorbar to each subplot
        fig.colorbar(mesh, ax=ax)

    # Adjust the layout of the subplots
    fig.tight_layout()


def plot_slice(
    df: pd.DataFrame,
    step: int,
    smooth: Union[bool, float] = False,
    log_loss=True,
    cmap="inferno",
    titles: Optional[dict] = None,
    unique_col: str = "weight_decay",
    log_x: bool = True,
    log_y: bool = True,
    title: str = "",
    columns=["train/loss", "test/loss", "train/acc", "test/acc"],
):
    df.sort_values(by=unique_col, inplace=True)

    unique_vals, slice_ = extract_slice(df, step, unique_col)
    # Two plots (loss & accuracy, combine test & train on each subfigure)

    # Set up the subplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Loop through each column and plot the slice as a line plot
    for i, column in enumerate(columns):
        ax = axes[i // 2]

        ax.plot(unique_vals, slice_[column])
        subtitle = titles.get(column, column) if titles else column
        ax.set_title(subtitle)

        ax.set_xlabel(title)

        if log_x:
            ax.set_xscale("log")

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")

    if log_y:
        axes[0].set_yscale("log")

    return fig, axes


def plot_curves_2x2(
    df: pd.DataFrame, title: str, unique_col: str, log_color: bool = True
):
    # Set up the subplot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Define the columns to plot
    columns = [
        ("_step", "train/loss"),
        ("_step", "test/loss"),
        ("_step", "train/acc"),
        ("_step", "test/acc"),
    ]
    subtitles = [
        "Training Loss",
        "Validation Loss",
        "Training Accuracy",
        "Validation Accuracy",
    ]

    n_unique = df[unique_col].nunique()

    # Loop through each column and plot the learning curve
    for i, (column, subtitle) in enumerate(zip(columns, subtitles)):
        row = i // 2
        col = i % 2
        ax = axes[row][col]

        if log_color:
            sns.lineplot(
                data=df,
                x=column[0],
                y=column[1],
                hue=unique_col,
                hue_norm=LogNorm(),
                ax=ax,
            )
        else:
            sns.lineplot(data=df, x=column[0], y=column[1], hue=unique_col, ax=ax)

        ax.set_title(subtitle)
        ax.set_xscale("log")
        ax.set_xlabel("Step")
        # ax.set_ylabel(column[1])

        if "loss" in column[1]:
            ax.set_yscale("log")

    # Add a legend to the last subplot (in the form of a heatmap)
    handles, labels = axes[1][1].get_legend_handles_labels()
    axes[1][1].legend(
        handles=handles[1:],
        labels=labels[1:],
        title=title,
        loc="center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=n_unique,
    )

    # Adjust the layout of the subplots
    fig.tight_layout()



def plot_details(
    df: pd.DataFrame,
    unique_col: str = "weight_decay",
    smooth: Union[bool, float] = False,
    log_loss=True,
    cmap="inferno",
    log_x: bool = True,
    log_y: bool = True,
    title: str = "",
    metric: str = "test/acc",
    metric_label: str = "Accuracy",
    step: int = 10000,
    run_val: float = 0.0,
    plot_extra: bool = False,
):
    # Figure:
    # - Top Row: Heatmap stretching across
    # - Bottom Row: Two line plots.

    metric_label_short = (
        metric_label.split(" ")[1] if " " in metric_label else metric_label
    )

    # create a figure with a 2x2 grid of subplots
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.59, 6, 6, 1.59])

    pivot_table = get_pivot(
        df, unique_col, reindex=True, interpolate=True, columns=[metric]
    )
    unique_vals = sorted(df[unique_col].unique())

    ax1 = plt.subplot(gs[0, 1:])
    mesh = create_heatmap(
        x=unique_vals,
        y=pivot_table[metric].index,
        z=pivot_table[metric].values,
        ax=ax1,
        smooth=smooth,
        cmap=cmap,
        log_x=log_x,
        log_y=log_y,
        log_z=log_loss and "loss" in metric,
        title=title,
    )

    fig.colorbar(mesh, ax=ax1)

    # Plot a vertical line at unique_col = run_val
    ax1.axvline(x=run_val, color=BLUE, linestyle="--", linewidth=1)

    # Plot a horizontal line at step = step
    ax1.axhline(y=step, color=RED, linestyle="--", linewidth=1)

    # Line plots

    # Slice
    unique_vals, slice_ = extract_slice(df, step, unique_col)
    ax2 = plt.subplot(gs[1, 1])

    if plot_extra:
        num_steps = len(df._step.unique())
        # Plot one slice every 100 steps
        slices_table = pivot_table.loc[pivot_table.index % 10 == 0, :].T.reset_index()
        slices = pd.melt(
            slices_table,
            id_vars=[unique_col, "level_0"],
            var_name="_step",
            value_name=metric,
        )
        print(slices)

        sns.lineplot(
            data=slices,
            x=unique_col,
            y=metric,
            hue="_step",
            ax=ax2,
            alpha=100.0 / num_steps,
            palette=REDS,
            legend=False,
        )
        slice_norm = colors.Normalize(vmin=0, vmax=df._step.max())
        slice_colorbar = cm.ScalarMappable(norm=slice_norm, cmap=REDS)
        fig.colorbar(slice_colorbar, ax=ax2, label="Steps")

    ax2.plot(unique_vals, slice_[metric], label=title, color=RED)
    ax2.set_xlabel(title)

    # Run example
    ax3 = plt.subplot(gs[1, 2])
    kwargs = {unique_col: run_val}
    steps, run = extract_run(df, **kwargs)

    if plot_extra:
        num_vals = len(unique_vals)
        sns.lineplot(
            data=df,
            x="_step",
            y=metric,
            hue=unique_col,
            ax=ax3,
            alpha=10.0 / num_vals,
            palette=BLUES,
            legend=False,
        )
        run_norm = colors.Normalize(vmin=min(unique_vals), vmax=max(unique_vals))
        run_colorbar = cm.ScalarMappable(norm=run_norm, cmap=BLUES)
        fig.colorbar(run_colorbar, ax=ax3, label=title)

    ax3.plot(steps, run[metric], label=title, color=BLUE)
    ax3.set_xlabel("Steps")

    for ax in [ax2, ax3]:
        if "Accuracy" in metric_label:
            ax.set_ylim(0.0, 1.05)
        else:
            min_loss, max_loss = df[metric].min(), df[metric].max()
            ax.set_ylim(min_loss - 0.5 * min_loss, max_loss + 0.5 * max_loss)
            ax.set_yscale("log")

        ax.set_ylabel(metric_label_short)
        ax.set_xscale("log")

    # Adjust the layout of the subplots
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_all_details(
    df,
    title,
    unique_col,
    run_val,
    log_x=True,
    log_y=True,
    plot_extra=False,
    cmap="viridis",
    metrics_and_labels=[
        ("train/acc", "Train Accuracy"),
        ("test/acc", "Test Accuracy"),
        ("train/loss", "Train Loss"),
        ("test/loss", "Test Loss"),
    ],
    format="png",
):
    for metric, label in metrics_and_labels:
        fig = plot_details(
            df,
            metric_label=label,
            metric=metric,
            title=title,
            unique_col=unique_col,
            run_val=run_val,
            cmap=cmap,
            log_x=log_x,
            log_y=log_y,
            plot_extra=plot_extra,
        )
        slug = metric.replace("/", "_")
        fig.savefig(f"../../figures/{unique_col}_{slug}.{format}", dpi=300)
        plt.show()

