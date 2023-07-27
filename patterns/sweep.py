import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import wandb

OutlierStrategy = Literal["remove", "replace", "keep"]
METRICS = [
    "train/acc",
    "test/acc",
    "train/loss",
    "test/loss",
    "corrupted/acc",
    "uncorrupted/acc",
]

ENTITY = os.environ.get("WANDB_ENTITY", None)

if ENTITY is None:
    raise ValueError("WANDB_ENTITY environment variable not set")


def generate_coarse_to_fine_grid_sweep(
    min_, max_, total_steps, step_sizes=[10, 5, 3, 1], type_="log"
):
    if type_ == "log":
        # Generate the logscale range
        grid = np.logspace(np.log10(min_), np.log10(max_), total_steps)
    elif type_ == "linear":
        grid = np.linspace(min_, max_, total_steps)
    else:
        grid = np.arange(min_, max_, int((max_ - min_) / total_steps))

    # Initialize an empty list to store the rearranged elements
    rearranged_grid = []

    # Iterate over the step sizes and merge the sublists
    for step in step_sizes:
        for i in range(0, len(grid), step):
            if grid[i] not in rearranged_grid:
                rearranged_grid.append(grid[i])

    return rearranged_grid


def rearrange_coarse_to_fine(grid: List, step_sizes=[10, 5, 3, 1]):
    # Initialize an empty list to store the rearranged elements
    rearranged_grid = []

    # Iterate over the step sizes and merge the sublists
    for step in step_sizes:
        for i in range(0, len(grid), step):
            if grid[i] not in rearranged_grid:
                rearranged_grid.append(grid[i])

    return rearranged_grid


def get_history(
    *sweep_ids,
    unique_cols: Union[List[str], str] = "weight_decay",
    entity: str = ENTITY,
    project: str = "grokking",
    allow_duplicates=False,
    combine_seeds=False,
    metrics=METRICS,
):
    """
    Gathers all the runs from a series of sweeps and combines them into a single dataframe.

    `unique_col` is used to identify duplicate runs. By default, `"_step"` is added.
    If there are duplicates, the run from the last sweep is kept.
    """
    api = wandb.Api()
    unique_cols = unique_cols if isinstance(unique_cols, list) else [unique_cols]

    def _get_history(sweep_id):
        """Get a dataframe for a single sweep."""
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs

        def create_run_df(history, config):
            for k, v in config.items():
                if k == "momentum" and isinstance(v, list):
                    v = [tuple(v)] * len(history)
                history[k] = v

            return history

        return pd.concat([create_run_df(run.history(), run.config) for run in runs])

    histories = pd.concat([_get_history(sweep_id) for sweep_id in sweep_ids])

    if not allow_duplicates:
        histories = histories.drop_duplicates(["_step", *unique_cols], keep="last")

    # Change step 0 to 1 to avoid issues with log plots
    histories.loc[histories._step == 0, "_step"] = 1

    # Fix types
    histories.applymap(lambda x: x.item() if isinstance(x, np.generic) else x)
    non_numeric_columns = histories.select_dtypes(
        exclude=["int", "float", "int64", "float64"]
    ).columns
    histories = histories.drop(columns=non_numeric_columns)

    # Sort
    histories = histories.sort_values(by=[*unique_cols, "_step"])

    if combine_seeds:
        assert (
            len(unique_cols) == 1
        ), "Can only combine seeds if there is a single unique column"

        unique_col = unique_cols[0]
        unique_vals = histories[unique_col].unique()

        for val in unique_vals:
            runs = histories[histories[unique_col] == val]
            seeds = runs.seed.unique()

            if len(seeds) > 1:
                # Define the metrics that need to be averaged
                for metric in metrics:
                    # Calculate the mean value for each metric and _step
                    means_groups = runs.groupby("_step")[metric]

                    means = means_groups.apply(
                        lambda x: x.ffill().bfill().mean()
                        if x.isna().any()
                        else x.mean()
                    )

                    # Update the histories dataframe
                    for _step, mean_value in means.items():
                        mask = (histories[unique_col] == val) & (
                            histories._step == _step
                        )
                        histories.loc[mask, metric] = mean_value

        # Remove duplicate rows
        histories = histories.drop_duplicates(subset=[*unique_cols, "_step"])

    return histories


def handle_outliers(
    df,
    unique_cols: List[str],
    loss_cols: List[str],
    threshold: float,
    late_epochs_ratio: float = 0.1,
    oscillation_window: int = 100,
    action: OutlierStrategy = "keep",
):
    """Some runs display large oscillations/instabilities late in training.
    (e.g., because of an excessively large weight decay).
    """
    if action not in ["keep", "remove", "replace"]:
        raise ValueError(
            "Invalid action. Supported actions: 'keep', 'remove', 'replace'"
        )

    # Group the DataFrame by unique_cols
    grouped = df.copy().groupby(unique_cols)

    # Initialize an empty list to store the processed data
    processed_runs = []

    # Iterate over the groups
    for name, group in grouped:
        # Sort the group by steps
        group = group.sort_values(by="_step")
        late_epochs = int(late_epochs_ratio * len(group))

        # Loop through the specified loss columns
        for loss_col in loss_cols:
            # Calculate the absolute difference in loss across oscillation_window epochs
            oscillation_windows = (
                group[loss_col]
                .rolling(oscillation_window)
                .apply(lambda x: x.std())
                .fillna(0)
            )

            # Calculate if any of the last epochs have an oscillation_measure above the threshold
            oscillation_measure = oscillation_windows.iloc[-late_epochs:].max()

            if oscillation_measure > threshold:
                if action == "remove":
                    break  # Skip this group, effectively removing it from the result
                elif action == "replace":
                    raise NotImplementedError("Replace not implemented yet")

                    # Find the first index within threshold of the minimum of that entire run
                    # min_value = group[loss_col].min()
                    # min_index = (group[loss_col].sub(min_value).abs() < 1e-9).idxmax()

                    # # Replace any windows exceeding the threshold with the mean of that window after the minimum
                    # group[loss_col, min_index:] = np.where(oscillation_windows > threshold, oscillation_windows.mean(), group[loss_col])[min_index:]

        else:  # This 'else' block only executes if the inner loop completes without encountering a 'break' statement
            processed_runs.append(group)

    return pd.concat(processed_runs)


def get_pivot(
    df: pd.DataFrame,
    unique_col: str,
    columns=["train/loss", "test/loss", "train/acc", "test/acc"],
    reindex: bool = False,
    interpolate: bool = False,
):
    # Create a pivot table with the data
    pivot_table = pd.pivot_table(df, values=columns, index="_step", columns=unique_col)

    if reindex:
        # Fill in the missing values using linear interpolation and gaussian smoothing
        pivot_table = pivot_table.reindex(np.arange(df._step.min(), df._step.max() + 1))

    if interpolate:
        # This will be used to fill in the missing values for the first few steps
        pivot_table = pivot_table.interpolate(method="linear").fillna(method="bfill")

    return pivot_table


def extract_slice(df: pd.DataFrame, step: int, unique_col: str):
    df.sort_values(by=unique_col, inplace=True)
    pivot_table = get_pivot(df, unique_col, reindex=True, interpolate=True)
    unique_vals = sorted(df[unique_col].unique())
    slice_ = pivot_table.loc[step]

    return unique_vals, slice_


def extract_run(df: pd.DataFrame, **kwargs):
    # Generate the 1x4 grid of Epoch-wise, Model-wise, Sample-wise, and Regularization-wise plots
    # Epoch-wise
    df.sort_values(by="_step", inplace=True)
    run = df
    for key, value in kwargs.items():
        run = run.loc[run[key] == value]

    run = run.set_index("_step")
    steps = run.index.values

    return steps, run


def extract_slice_from_pivot(
    pivot_table, step, metric, unique_col, smooth: Union[bool, float] = False
):
    _pivot_table = pivot_table.copy()

    if smooth:
        _pivot_table[metric] = gaussian_filter(pivot_table[metric].values, sigma=smooth)

    slice_ = _pivot_table.loc[_pivot_table.index == step, :].T.reset_index()
    slice_ = pd.melt(
        slice_,
        id_vars=[unique_col, "level_0"],
        var_name="_step",
        value_name=metric,
    )

    return slice_


def exp_filter(z, sigma):
    z = gaussian_filter(z, sigma=sigma, mode="nearest")

    return z


def extract_run_from_pivot(
    pivot_table, run_val, smooth: Union[bool, float] = False, metrics=METRICS
):
    _pivot_table = pivot_table.copy()

    if smooth:
        for metric in metrics:
            _pivot_table[metric] = exp_filter(
                pivot_table[metric].values, sigma=(smooth)
            )

    run = _pivot_table[[(m, run_val) for m in metrics]]
    run = run.reset_index()

    return run
