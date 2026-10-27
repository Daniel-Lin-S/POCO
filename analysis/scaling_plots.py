"""Draw score-first paired scaling figures with shared model styling.

Input is a summary table with ``model``, an x-axis field, and
``test_score_mean``, ``test_score_sem``, ``test_loss_mean``, and
``test_loss_sem`` columns. Output is one PDF at an absolute path supplied
by the caller. The caller owns result and folder lifecycle messages.
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from analysis import plots

FIGURE_WIDTH = 7.2
FIGURE_HEIGHT = 6.0
LINE_WIDTH = 2.0
TITLE_WIDTH = 55
X_MARGIN = 0.08
Y_MARGIN = 0.08
MODEL_STYLES = {'POCO': '-', 'NLinear': '--'}
METRICS = (
    ('test_score', 'Test score relative to last-frame copy'),
    ('test_loss', 'Test loss (MSE)'),
)


def plot_scaling_panels(
    summary: pd.DataFrame, x_field: str, x_label: str,
    title: str, path: Path, log2_x: bool = False,
) -> None:
    """Save score and loss panels with model colors and line styles.

    Parameters
    ----------
    summary : pandas.DataFrame
        Rows by model and x value with finite means and standard errors.
    x_field : str
        Column containing positive x-axis values.
    x_label : str
        Axis label including units where applicable.
    title : str
        Figure title wrapped to fit the canvas.
    path : Path
        Absolute output PDF path.
    log2_x : bool, optional
        Use a base-two logarithmic x-axis, default False.
    """
    if not path.is_absolute():
        raise ValueError(f'Expected absolute figure path, got {path}.')
    models = list(MODEL_STYLES)
    if set(summary['model']) != set(models):
        raise ValueError(f'Expected models {models} in {path}.')
    sns.set_theme(style='ticks', context='paper')
    colors = dict(zip(models, plots.get_model_colors(models)))
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for model in models:
            rows = summary.loc[summary['model'] == model].sort_values(
                x_field
            )
            x = rows[x_field].to_numpy(dtype=float)
            mean = rows[f'{metric}_mean'].to_numpy(dtype=float)
            sem = rows[f'{metric}_sem'].to_numpy(dtype=float)
            if not np.isfinite((x, mean, sem)).all():
                raise ValueError(f'Non-finite {metric} figure values.')
            sns.lineplot(
                x=x, y=mean, ax=ax, color=colors[model],
                linewidth=LINE_WIDTH, marker='o', label=model,
                linestyle=MODEL_STYLES[model],
            )
            ax.fill_between(
                x, mean - sem, mean + sem,
                color=colors[model], alpha=0.18,
            )
        means = summary[f'{metric}_mean'].to_numpy(dtype=float)
        errors = summary[f'{metric}_sem'].to_numpy(dtype=float)
        lower = np.min(means - errors)
        upper = np.max(means + errors)
        span = upper - lower
        if span <= 0:
            raise ValueError(f'No {metric} range for {path}.')
        ax.set_ylim(lower - Y_MARGIN * span, upper + Y_MARGIN * span)
        ticks = np.sort(summary[x_field].unique())
        if log2_x:
            ax.set_xscale('log', base=2)
        ax.set_xlim(ticks[0] * (1 - X_MARGIN),
                    ticks[-1] * (1 + X_MARGIN))
        ax.set_xticks(ticks, labels=[str(value) for value in ticks])
        ax.set_ylabel(ylabel)
        ax.legend().remove()
        sns.despine(ax=ax)
    axes[-1].set_xlabel(x_label)
    axes[0].set_xlabel('')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
        ncol=len(models), frameon=False,
    )
    fig.suptitle(fill(title, TITLE_WIDTH), y=0.98)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
