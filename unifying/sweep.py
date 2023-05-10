from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

import wandb

OutlierStrategy = Literal["remove", "replace", "keep"]


def get_history(
    *sweep_ids,
    unique_cols: Union[List[str], str] = "weight_decay",
    entity: str = "jqhoogland",
    project: str = "grokking",
    allow_duplicates=False,
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

    # Remove any runs that didn't have any steps after 1000
    for unique_col in unique_cols:
        valid_runs = histories.groupby(unique_col).apply(
            lambda x: x["_step"].max() > 1000
        )
        histories = histories[histories[unique_col].isin(valid_runs[valid_runs].index)]

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
