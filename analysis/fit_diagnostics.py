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
import time
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
    ahrens_session_lowpass_extended,
    ahrens_head_sharing_factorial,
)
from utils.artifacts import file_digest, fit_id, validate_complete
from utils.campaigns.claims import fit_claim
from utils.campaigns.lifecycle import CampaignLog

FIGURE_VERSION = 2
PREDICTION_DIMS = (0, 20, 50, -1)
EXAMPLE_COUNT = 10
WATCH_POLL_SECONDS = 30
FIGURE_NAMES = (
    'training_dynamics.pdf',
    *(f'{phase}_pred_vs_target/best_dim{pc}.pdf'
      for phase in ('train', 'val', 'test')
      for pc in PREDICTION_DIMS),
)
CAMPAIGN_FACTORIES = {
    'ahrens_length_lowpass_four': ahrens_length_lowpass_four,
    'ahrens_session_lowpass_six': ahrens_session_lowpass_six,
    'ahrens_session_lowpass_extended': (
        ahrens_session_lowpass_extended
    ),
    'ahrens_head_sharing_factorial': (
        ahrens_head_sharing_factorial
    ),
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
    if config.experiment_name == 'ahrens_head_sharing_factorial':
        if not label.startswith('zebrafishahrens_pc-'):
            raise ValueError(
                f'Expected Ahrens subject group, got {label}.'
            )
        cohort = f'{len(subjects)} subjects; reference 8, 09, 11'
    dataset = config.dataset_config[label]
    condition = str(config.dataset_filter).replace('lowpass', 'low-pass')
    details = [dataset_name]
    if condition != 'none':
        details.append(condition)
    if dataset.pc_dim is not None:
        details.append(f'{dataset.pc_dim} PCs')
    nominal = getattr(config, 'train_data_length', None)
    if nominal is None:
        exposure = 'full chronological 60% train'
    else:
        train_frames = int(nominal * dataset.train_split)
        exposure = (
            f'{train_frames} train frames '
            f'(nominal {nominal})'
        )
    policy = ''
    if config.experiment_name == 'ahrens_head_sharing_factorial':
        names = (
            ('W_o', config.out_weight_scope),
            ('input', config.in_proj_scope),
            ('W_alpha', config.alpha_weight_scope),
        )
        policy = ' | ' + ', '.join(
            f'{name} {scope}' for name, scope in names
        )
    return (
        f'{config.model_label} | {" ".join(details)} | '
        f'{cohort} | {exposure}{policy} | '
        f'seed {config.seed}'
    )


def pc_label(
    config: NeuralPredictionConfig, dimension: int,
    component_count: int | None = None,
) -> str:
    """Name a zero-based PC index, resolving -1 to the last PC.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit whose processed PC count sets the last index.
    dimension : int
        Zero-based component index or -1 for the last component.
    component_count : int or None, optional
        Actual number of recorded dimensions, default processed PC count.

    Returns
    -------
    str
        Explicit PC or unit index for the figure title.
    """
    label = config.dataset_label[0]
    dataset = config.dataset_config[label]
    count = (component_count if component_count is not None
             else dataset.pc_dim)
    if count is None or count < 1:
        raise ValueError(f'No positive component count for {label}.')
    index = count - 1 if dimension == -1 else dimension
    if index < 0 or index >= count:
        raise ValueError(
            f'PC index {dimension} is outside 0..{count - 1}.'
        )
    suffix = ' (last)' if dimension == -1 else ''
    component = 'PC' if dataset.pc_dim is not None else 'Unit'
    return f'{component} index {index}{suffix}'


def sample_row_label(
    config: NeuralPredictionConfig, session_index: int,
    sample_index: int, session_count: int,
) -> str:
    """Name a saved prediction window without implying recording time.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit whose selected recordings define sample order.
    session_index : int
        Position of the sampled recording in the fit.
    sample_index : int
        Position of the window within the saved batch.
    session_count : int
        Number of sampled recordings in the saved information.

    Returns
    -------
    str
        Subject and batch-slot label for one plotted row.
    """
    label = config.dataset_label[0]
    selected = config.dataset_config[label].session_ids
    if (isinstance(selected, list) and
            len(selected) == session_count and
            isinstance(selected[session_index], str)):
        recording = selected[session_index].replace('_', ' ')
    else:
        recording = f'recording {session_index + 1}'
    return f'{recording} | saved batch slot {sample_index + 1}'


def _require_dynamics(config: NeuralPredictionConfig) -> pd.DataFrame:
    """Read finite epoch-indexed train and validation metrics.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Complete fit containing ``progress.txt``.

    Returns
    -------
    pandas.DataFrame
        Logged training rows with loss and MAE columns.
    """
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
    """Save loss and MAE curves with both splits in each panel.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit whose title and output directory are used.
    table : pandas.DataFrame
        Training rows with Epoch, TrainLoss, ValLoss, and split MAE.
    """
    sns.set_theme(style='ticks', context='talk', font_scale=0.9)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5))
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
            linewidth=2.5, linestyle='-', estimator=None,
        )
        sns.lineplot(
            x=x, y=table[val_column].to_numpy(dtype=float),
            ax=ax, label='Validation', color='#3C4147',
            linewidth=2.5, linestyle='--', estimator=None,
        )
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend().remove()
        sns.despine(ax=ax)
    axes[0].set_xlabel('')
    axes[-1].set_xlabel('Epoch (one-based)', fontsize=14)
    for ax in axes:
        if x[0] == x[-1]:
            ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
        else:
            ax.set_xlim(max(1, x[0]), x[-1])
    title = fill(
        f'{fit_title(config)} | Training dynamics', width=68
    )
    lines = title.count('\n') + 1
    fig.suptitle(title, y=0.99, fontsize=17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96 - 0.025 * (lines - 1)))
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
    """Redraw each split's saved examples at the original PDF paths.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Complete fit with sample-trial arrays in ``*_best_info.npy``.
    """
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
    """Validate the independent completion record for fit PDFs.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit whose figure hashes are checked.

    Returns
    -------
    bool
        True when all current-version PDFs match the figure marker.
    """
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


def _refresh_one(
    config: NeuralPredictionConfig, campaign: CampaignLog,
) -> bool:
    """Record one figure repair or isolated failure in a lifecycle.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit to refresh when scientific artifacts are complete.
    campaign : CampaignLog
        Figure-repair lifecycle containing this fit ID.

    Returns
    -------
    bool
        True when current-version figures were newly written.
    """
    job_id = fit_id(config)
    try:
        with fit_claim(
            config, on_wait=lambda: campaign.mark(job_id, 'busy')
        ):
            campaign.mark(job_id, 'running')
            changed = render_completed_fit_figures(config)
            campaign.mark(job_id, 'finished' if changed else 'reused')
            return changed
    except Exception as error:
        campaign.mark(
            job_id, 'failed', f'{type(error).__name__}: {error}'
        )
        print(
            f'Figure refresh failed for '
            f'{Path(config.save_path).resolve()}: '
            f'{type(error).__name__}: {error}'
        )
        return False


def _run_once(configs: list[NeuralPredictionConfig]) -> None:
    """Refresh every fit complete when this invocation begins.

    Parameters
    ----------
    configs : list[NeuralPredictionConfig]
        Resolved fits from selected Ahrens campaigns.
    """
    ready = [
        config for config in configs
        if (Path(config.save_path) / 'complete.json').is_file()
    ]
    if not ready:
        raise ValueError(
            'No completed Ahrens fits are ready for figure repair.'
        )
    campaign = CampaignLog.create(
        'ahrens_fit_figure_repair', [fit_id(config) for config in ready]
    )
    print(f'Figure repair status: {campaign.path / "status.json"}')
    rendered = sum(_refresh_one(config, campaign) for config in ready)
    status = campaign.read_status()
    print(
        f'Figure backfill: refreshed={rendered}, '
        f'not_yet_complete={len(configs) - len(ready)}, '
        f'failed={status["failed"]}'
    )
    if status['failed']:
        raise RuntimeError(
            f'{status["failed"]} fit figure refreshes failed.'
        )


def _source_jobs(
    names: list[str], paths: list[Path],
    configs: list[NeuralPredictionConfig],
) -> dict[str, CampaignLog]:
    """Validate model-campaign logs for a long-running figure watch.

    Parameters
    ----------
    names : list[str]
        Selected experiment names in source-path order.
    paths : list[Path]
        Absolute lifecycle folders for the training campaigns.
    configs : list[NeuralPredictionConfig]
        All resolved fits expected in the two logs.

    Returns
    -------
    dict[str, CampaignLog]
        Validated training lifecycle readers by experiment name.
    """
    if len(names) != len(paths) or len(set(names)) != len(names):
        raise ValueError(
            'Watch requires one distinct source status per campaign.'
        )
    readers = {}
    for name, path in zip(names, paths):
        if not path.is_absolute():
            raise ValueError(f'Expected absolute source status: {path}.')
        reader = CampaignLog(path)
        status = reader.read_status()
        expected = {
            fit_id(config) for config in configs
            if config.experiment_name == name
        }
        if (status['campaign'] != f'{name}_training' or
                set(status['jobs']) != expected):
            raise ValueError(
                f'Training lifecycle does not match {name}: {path}.'
            )
        readers[name] = reader
    return readers


def _watch(
    configs: list[NeuralPredictionConfig],
    sources: dict[str, CampaignLog],
) -> None:
    """Repair newly complete fits until both source campaigns end.

    Parameters
    ----------
    configs : list[NeuralPredictionConfig]
        Fixed scientific fits to monitor.
    sources : dict[str, CampaignLog]
        Training lifecycle readers keyed by experiment name.
    """
    campaign = CampaignLog.create(
        'ahrens_fit_figure_watch', [fit_id(config) for config in configs]
    )
    print(f'Figure watch status: {campaign.path / "status.json"}')
    pending = list(configs)
    while pending:
        states = {
            name: source.read_status()['jobs']
            for name, source in sources.items()
        }
        next_pending = []
        for config in pending:
            job_id = fit_id(config)
            training_state = states[config.experiment_name][job_id]
            marker = Path(config.save_path) / 'complete.json'
            if marker.is_file():
                _refresh_one(config, campaign)
            elif training_state in ('failed', 'finished', 'reused'):
                campaign.mark(
                    job_id, 'failed',
                    f'Training ended as {training_state} without a '
                    'valid completion marker.',
                )
            else:
                next_pending.append(config)
        pending = next_pending
        if pending:
            time.sleep(WATCH_POLL_SECONDS)
    status = campaign.read_status()
    print(
        f'Figure watch ended: finished={status["finished"]}, '
        f'failed={status["failed"]}; '
        f'status={campaign.path / "status.json"}'
    )


def main() -> None:
    """Backfill once or follow both Ahrens training campaigns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--campaign', choices=sorted(CAMPAIGN_FACTORIES),
        action='append', required=True,
    )
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--source-status', action='append', type=Path)
    args = parser.parse_args()
    configs = [
        config
        for name in args.campaign
        for seed_configs in CAMPAIGN_FACTORIES[name]().values()
        for config in seed_configs
    ]
    if args.watch:
        if not args.source_status:
            parser.error('--watch requires --source-status entries.')
        sources = _source_jobs(
            args.campaign, args.source_status, configs
        )
        _watch(configs, sources)
    else:
        if args.source_status:
            parser.error('--source-status requires --watch.')
        _run_once(configs)


if __name__ == '__main__':
    main()
