"""Draw fixed-cohort score and MSE scaling comparisons.

Input is a summary table with model, x-axis value, and test score/MSE
mean and SEM columns. An optional reference table has one row per model
with the same test metric columns. Output is a PDF at the absolute path
supplied by the caller. The caller owns artifact lifecycle messages.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import plots
from analysis.diagnostics.figure_style import (
    LABEL_FONT, TICK_FONT, save_diagnostic,
)

FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 9.5
LINE_WIDTH = 2.8
REFERENCE_COLOR = '#666666'
REFERENCE_BAND_ALPHA = 0.10
X_MARGIN = 0.08
Y_MARGIN = 0.08
MODEL_STYLES = {'POCO': '-', 'NLinear': '--'}
METRICS = (
    ('test_score', 'Test score relative to last-frame copy'),
    ('test_loss', 'Test MSE (PC activity²)'),
)


def plot_scaling_panels(
    summary: pd.DataFrame, x_field: str, x_label: str,
    title: str, path: Path, log2_x: bool = False,
    reference: pd.DataFrame | None = None,
) -> None:
    """Save score and MSE panels with optional singleton controls.

    Parameters
    ----------
    summary : pandas.DataFrame
        One row per model and x value with finite test means and SEMs.
    x_field : str
        Column containing positive x-axis values.
    x_label : str
        Axis label including units where applicable.
    title : str
        Figure title; the shared saver wraps it.
    path : pathlib.Path
        Absolute PDF output path.
    log2_x : bool, optional
        Use a base-two logarithmic x-axis, default False.
    reference : pandas.DataFrame or None, optional
        One fixed-subject singleton summary per model, default None.
    """
    if not path.is_absolute():
        raise ValueError(f'Expected absolute figure path, got {path}.')
    models = list(MODEL_STYLES)
    if set(summary['model']) != set(models):
        raise ValueError(f'Expected models {models} in {path}.')
    if reference is not None:
        if (len(reference) != len(models) or
                set(reference['model']) != set(models)):
            raise ValueError(
                f'Expected one singleton reference per model in {path}.'
            )
        reference = reference.set_index('model')
    sns.set_theme(style='ticks', context='talk')
    colors = dict(zip(models, plots.get_model_colors(models)))
    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )
    for ax, (metric, ylabel) in zip(axes, METRICS):
        bounds = []
        for model in models:
            rows = summary.loc[summary['model'] == model].sort_values(
                x_field
            )
            x = rows[x_field].to_numpy(dtype=float)
            mean = rows[f'{metric}_mean'].to_numpy(dtype=float)
            sem = rows[f'{metric}_sem'].to_numpy(dtype=float)
            if (not np.isfinite((x, mean, sem)).all() or
                    np.any(x <= 0) or np.any(sem < 0)):
                raise ValueError(
                    f'Invalid {metric} scaling values for {model}.'
                )
            sns.lineplot(
                x=x, y=mean, ax=ax, color=colors[model],
                linewidth=LINE_WIDTH, marker='o',
                linestyle=MODEL_STYLES[model],
                label=f'{model} multi-session',
            )
            ax.fill_between(
                x, mean - sem, mean + sem,
                color=colors[model], alpha=0.18,
            )
            bounds.extend((mean - sem).tolist())
            bounds.extend((mean + sem).tolist())
        if reference is not None:
            for model in models:
                mean = float(reference.loc[
                    model, f'{metric}_mean'
                ])
                sem = float(reference.loc[
                    model, f'{metric}_sem'
                ])
                if (not np.isfinite((mean, sem)).all()
                        or sem < 0):
                    raise ValueError(
                        f'Invalid singleton {metric} for {model}.'
                    )
                ax.axhspan(
                    mean - sem, mean + sem,
                    color=REFERENCE_COLOR,
                    alpha=REFERENCE_BAND_ALPHA,
                )
                ax.axhline(
                    mean, color=REFERENCE_COLOR,
                    linestyle=MODEL_STYLES[model],
                    linewidth=LINE_WIDTH,
                    label=f'{model} single-session',
                )
                bounds.extend((mean - sem, mean + sem))
        lower = min(bounds)
        upper = max(bounds)
        span = upper - lower
        if span <= 0:
            raise ValueError(f'No {metric} range for {path}.')
        ax.set_ylim(
            lower - Y_MARGIN * span,
            upper + Y_MARGIN * span,
        )
        ticks = np.sort(summary[x_field].unique())
        if log2_x:
            ax.set_xscale('log', base=2)
        ax.set_xlim(
            ticks[0] * (1 - X_MARGIN),
            ticks[-1] * (1 + X_MARGIN),
        )
        ax.set_xticks(ticks, labels=[str(value) for value in ticks])
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT)
        ax.tick_params(axis='both', labelsize=TICK_FONT)
        ax.legend().remove()
        sns.despine(ax=ax)
    axes[-1].set_xlabel(x_label, fontsize=LABEL_FONT)
    axes[0].set_xlabel('')
    handles, labels = axes[0].get_legend_handles_labels()
    save_diagnostic(
        fig, path, title, handles, labels,
        legend_columns=2 if reference is not None else None,
    )
