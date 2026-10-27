"""Refresh Ahrens fit PDFs in place from complete fit artifacts.

Input is a hash-named fit folder with ``config.json``, ``progress.txt``
(Epoch, BatchNum, TrainLoss, ValLoss, and split MAE columns), and
``train_best_info.npy``, ``val_best_info.npy``, ``test_best_info.npy``
(each containing ``sample_trials`` as prediction, target, input arrays
with shape ``(time, batch, PCs)``). Output is ``training_dynamics.pdf``
and four ``best_dim*.pdf`` files in each existing split prediction folder.
``fit_figures_complete.json`` records figure hashes for safe backfill.
Scientific metrics, checkpoints, and ``complete.json`` are read only.
"""

import argparse
import json
import os
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import plots
from configs.configs import NeuralPredictionConfig
from configs.experiments import (
    ahrens_length_lowpass_four, ahrens_session_lowpass_six,
)
from utils.artifacts import file_digest, fit_id, validate_complete
from utils.campaigns.claims import fit_claim

FIGURE_VERSION = 1
PREDICTION_DIMS = (0, 20, 50, -1)
EXAMPLE_COUNT = 10
FIGURE_NAMES = (
    'training_dynamics.pdf',
    *(f'{phase}_pred_vs_target/best_dim{pc}.pdf'
      for phase in ('train', 'val', 'test')
      for pc in PREDICTION_DIMS),
)
CAMPAIGN_FACTORIES = {
    'ahrens_length_lowpass_four': ahrens_length_lowpass_four,
    'ahrens_session_lowpass_six': ahrens_session_lowpass_six,
}


def fit_title(config: NeuralPredictionConfig) -> str:
    """Describe a fit using its model and campaign-relevant condition.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit configuration.

    Returns
    -------
    str
        Short title without paths or unrelated architecture settings.
    """
    if len(config.dataset_label) != 1:
        raise ValueError(
            f'Expected one dataset label, got {config.dataset_label}.'
        )
    label = config.dataset_label[0]
    if label.startswith('zebrafishahrens_pc-'):
        subjects = label.partition('-')[2].split('+')
        numbers = [subject.removeprefix('subject_')
                   for subject in subjects]
        noun = 'subject' if len(numbers) == 1 else 'subjects'
        cohort = f'{noun} {", ".join(numbers)}'
        dataset_name = 'Ahrens'
    else:
        cohort = label
        dataset_name = 'Neural data'
    dataset = config.dataset_config[label]
    condition = str(config.dataset_filter).replace('lowpass', 'low-pass')
    nominal = getattr(config, 'train_data_length', None)
    if nominal is None:
        exposure = 'full chronological 60% train'
    else:
        train_frames = int(nominal * dataset.train_split)
        exposure = (
            f'{train_frames} train frames '
            f'(nominal {nominal})'
        )
    return (
        f'{config.model_label} | {dataset_name} {condition}, '
        f'{dataset.pc_dim} PCs | {cohort} | {exposure} | '
        f'seed {config.seed}'
    )


def pc_label(config: NeuralPredictionConfig, dimension: int) -> str:
    """Name a zero-based PC index, resolving -1 to the last PC."""
    label = config.dataset_label[0]
    count = config.dataset_config[label].pc_dim
    index = count - 1 if dimension == -1 else dimension
    if index < 0 or index >= count:
        raise ValueError(
            f'PC index {dimension} is outside 0..{count - 1}.'
        )
    suffix = ' (last)' if dimension == -1 else ''
    return f'PC index {index}{suffix}'


def sample_row_label(
    config: NeuralPredictionConfig, session_index: int,
    sample_index: int, session_count: int,
) -> str:
    """Name a saved prediction window without implying recording time."""
    label = config.dataset_label[0]
    selected = config.dataset_config[label].session_ids
    if isinstance(selected, list) and len(selected) == session_count:
        recording = selected[session_index].replace('_', ' ')
    else:
        recording = f'recording {session_index + 1}'
    return f'{recording} | saved batch slot {sample_index + 1}'


def _require_dynamics(config: NeuralPredictionConfig) -> pd.DataFrame:
    """Read finite epoch-indexed train and validation metrics."""
    path = Path(config.save_path) / 'progress.txt'
    table = pd.read_table(path)
    label = config.dataset_label[0]
    required = (
        'Epoch', 'BatchNum', 'TrainLoss', 'ValLoss',
        f'{label}_train_mae', f'{label}_val_mae',
    )
    missing = set(required) - set(table.columns)
    if table.empty or missing:
        raise ValueError(
            f'Expected nonempty progress with {sorted(required)} '
            f'in {path}; missing {sorted(missing)}.'
        )
    if not np.isfinite(table[list(required)].to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite training dynamics in {path}.')
    if (table['Epoch'].iloc[0] < 0 or
            (table['Epoch'].diff().dropna() < 0).any() or
            (table['BatchNum'].diff().dropna() <= 0).any()):
        raise ValueError(f'Nonmonotonic epochs or steps in {path}.')
    return table


def _save_training_dynamics(
    config: NeuralPredictionConfig, table: pd.DataFrame,
) -> None:
    """Save loss and MAE curves with both splits in each panel."""
    sns.set_theme(style='ticks', context='paper')
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0))
    label = config.dataset_label[0]
    x = table['Epoch'].to_numpy(dtype=float) + 1
    train_color = plots.get_model_colors([config.model_label])[0]
    panels = (
        ('TrainLoss', 'ValLoss', 'Loss (MSE)'),
        (f'{label}_train_mae', f'{label}_val_mae',
         'Mean absolute error'),
    )
    for ax, (train_column, val_column, ylabel) in zip(axes, panels):
        sns.lineplot(
            x=x, y=table[train_column].to_numpy(dtype=float),
            ax=ax, label='Train', color=train_color,
            linewidth=2.0, linestyle='-', estimator=None,
        )
        sns.lineplot(
            x=x, y=table[val_column].to_numpy(dtype=float),
            ax=ax, label='Validation', color='#3C4147',
            linewidth=2.0, linestyle='--', estimator=None,
        )
        ax.set_ylabel(ylabel)
        ax.legend().remove()
        sns.despine(ax=ax)
    axes[0].set_xlabel('')
    axes[-1].set_xlabel('Epoch (one-based)')
    for ax in axes:
        if x[0] == x[-1]:
            ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
        else:
            ax.set_xlim(max(1, x[0]), x[-1])
    title = fill(f'{fit_title(config)} | Training dynamics', width=62)
    lines = title.count('\n') + 1
    fig.suptitle(title, y=0.99, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94 - 0.045 * (lines - 1)))
    output = Path(config.save_path) / 'training_dynamics.pdf'
    pending = output.with_name(f'.{output.name}.{os.getpid()}.pending')
    try:
        fig.savefig(pending, format='pdf')
        os.replace(pending, output)
    finally:
        plt.close(fig)
        pending.unlink(missing_ok=True)


def _prediction_examples(
    config: NeuralPredictionConfig, samples: list,
    dimension: int, loss_mode: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int | None,
           list[str]]:
    """Select deterministic examples from saved session samples.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit condition used to name sampled recordings.
    samples : list
        Saved prediction, target, input triples of shape
        ``(time, batch, PCs)`` for each sampled session.
    dimension : int
        PC index, including -1 for the last PC.
    loss_mode : str
        Fit prediction mode.

    Returns
    -------
    tuple
        Ten prediction/target 1D pairs, optional forecast boundary,
        and one recording/window label per row.
    """
    if not samples:
        raise ValueError('No saved prediction samples are available.')
    examples = []
    row_labels = []
    boundary = None
    for index in range(EXAMPLE_COUNT):
        session_index = index % len(samples)
        prediction, target, history = samples[session_index]
        if (prediction.ndim != 3 or target.ndim != 3 or
                history.ndim != 3 or
                prediction.shape != target.shape or
                prediction.shape[1] < 1 or
                prediction.shape[2] <= dimension):
            raise ValueError(
                f'Expected matching nonempty (time, batch, PCs) '
                f'prediction arrays, got {prediction.shape} and '
                f'{target.shape}.'
            )
        sample_index = (index // len(samples)) % prediction.shape[1]
        predicted = prediction[:, sample_index, dimension]
        expected = target[:, sample_index, dimension]
        if loss_mode == 'prediction':
            context = history[:, sample_index, dimension]
            predicted = np.concatenate((context, predicted))
            expected = np.concatenate((context, expected))
            boundary = context.size
        examples.append((predicted, expected))
        row_labels.append(sample_row_label(
            config, session_index, sample_index, len(samples)
        ))
    return examples, boundary, row_labels


def _save_prediction_figures(config: NeuralPredictionConfig) -> None:
    """Redraw each split's saved examples at the original PDF paths."""
    root = Path(config.save_path)
    color = plots.get_model_colors([config.model_label])[0]
    for phase in ('train', 'val', 'test'):
        info_path = root / f'{phase}_best_info.npy'
        info = np.load(info_path, allow_pickle=True).item()
        samples = info['sample_trials']
        for dimension in PREDICTION_DIMS:
            examples, boundary, row_labels = _prediction_examples(
                config, samples, dimension, config.loss_mode
            )
            title = (
                f'{fit_title(config)} | {phase.title()} '
                f'predictions | {pc_label(config, dimension)}'
            )
            plots.pred_vs_target_plot(
                examples, str(root / f'{phase}_pred_vs_target'),
                f'best_dim{dimension}', boundary,
                title=title, prediction_color=color,
                row_labels=row_labels,
            )


def figures_complete(config: NeuralPredictionConfig) -> bool:
    """Validate the independent completion record for fit PDFs."""
    root = Path(config.save_path)
    marker = root / 'fit_figures_complete.json'
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
        if (record['version'] != FIGURE_VERSION or
                record['fit_id'] != fit_id(config) or
                set(record['files']) != set(FIGURE_NAMES)):
            return False
        return all(
            (root / name).is_file() and
            file_digest(root / name) == digest
            for name, digest in record['files'].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def render_completed_fit_figures(
    config: NeuralPredictionConfig,
) -> bool:
    """Refresh in-place PDFs for one validated complete fit.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Complete fit with saved metrics and sample trials.

    Returns
    -------
    bool
        True when figures were refreshed, False when already valid.
    """
    if not validate_complete(config):
        raise ValueError(f'Cannot plot incomplete fit {config.save_path}.')
    if figures_complete(config):
        return False
    table = _require_dynamics(config)
    _save_training_dynamics(config, table)
    _save_prediction_figures(config)
    root = Path(config.save_path)
    record = {
        'version': FIGURE_VERSION,
        'fit_id': fit_id(config),
        'files': {
            name: file_digest(root / name) for name in FIGURE_NAMES
        },
    }
    marker = root / 'fit_figures_complete.json'
    pending = root / f'.fit_figures_complete.{os.getpid()}.pending'
    try:
        pending.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(pending, marker)
    finally:
        pending.unlink(missing_ok=True)
    if not figures_complete(config):
        raise RuntimeError(
            f'Fit figure completion check failed: {marker}'
        )
    print(f'Saved fit diagnostic figures: {root.resolve()}')
    return True


def main() -> None:
    """Backfill complete Ahrens fits without rerunning model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--campaign', choices=sorted(CAMPAIGN_FACTORIES),
        action='append', required=True,
    )
    args = parser.parse_args()
    rendered = 0
    skipped = 0
    failures = 0
    for name in args.campaign:
        configs = CAMPAIGN_FACTORIES[name]()
        for seed_configs in configs.values():
            for config in seed_configs:
                if not (Path(config.save_path) / 'complete.json').is_file():
                    skipped += 1
                    continue
                try:
                    with fit_claim(config):
                        rendered += render_completed_fit_figures(config)
                except Exception as error:
                    failures += 1
                    print(
                        f'Figure refresh failed for '
                        f'{Path(config.save_path).resolve()}: '
                        f'{type(error).__name__}: {error}'
                    )
    print(
        f'Figure backfill: refreshed={rendered}, '
        f'not_ready_or_current={skipped}, failed={failures}'
    )
    if failures:
        raise RuntimeError(f'{failures} fit figure refreshes failed.')


if __name__ == '__main__':
    main()
