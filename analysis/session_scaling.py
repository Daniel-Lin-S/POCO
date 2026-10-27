"""Analyze the six-subject Ahrens session-scaling pilot.

Input is each complete hash-named fit's ``test.txt`` (TestLoss and
``<label>_test_score``), ``progress.txt`` (TrainExamplesSeen), and
``flops.json`` (forward/backward FLOPs and trainable parameters).
Output is a unique artifact figure folder with ``summary.csv`` (one
row per model and session count, seed means and standard errors plus
per-fit and cohort training-compute summaries) and
``session_scaling.pdf``. Source fit
metrics and checkpoints remain in their own immutable directories.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.scaling_plots import plot_scaling_panels
from configs.campaigns.ahrens_session import load_ahrens_session_campaign
from configs.config_global import FIG_DIR
from configs.experiments import ahrens_session_lowpass_six
from utils.artifacts import validate_complete

METRICS = ('test_score', 'test_loss')


def read_group_results() -> pd.DataFrame:
    """Read finite test and training-compute metrics from all pilot fits.

    Returns
    -------
    pandas.DataFrame
        One row per completed fit, keyed by seed, model, and group.
    """
    settings = load_ahrens_session_campaign()
    records = []
    for seed, configs in ahrens_session_lowpass_six().items():
        for config in configs:
            if not validate_complete(config):
                raise ValueError(f'Incomplete session fit: {config.save_path}')
            label = config.dataset_label[0]
            root = Path(config.save_path)
            test = pd.read_table(root / 'test.txt')
            progress = pd.read_table(root / 'progress.txt')
            flops = json.loads((root / 'flops.json').read_text())
            score_column = f'{label}_test_score'
            if len(test) != 1 or score_column not in test:
                raise ValueError(
                    f'Expected one test row with {score_column} '
                    f'in {root / "test.txt"}.'
                )
            record = {
                'seed': seed,
                'model': config.model_label,
                'group': label,
                'sessions_per_fit': len(
                    config.dataset_config[label].session_ids
                ),
                'test_score': float(test[score_column].iloc[0]),
                'test_loss': float(test['TestLoss'].iloc[0]),
                'train_step_flops': float(
                    flops['forward_backward_flops_per_step']
                ),
                'trainable_params': float(
                    flops['trainable_parameters']
                ),
                'train_examples_per_fit': float(
                    progress['TrainExamplesSeen'].iloc[-1]
                ),
                'training_flops_per_fit_estimate': float(
                    flops['forward_backward_flops_per_step'] *
                    config.max_batch
                ),
            }
            numeric = [value for key, value in record.items()
                       if key not in ('seed', 'model', 'group')]
            if not np.isfinite(numeric).all():
                raise ValueError(f'Non-finite metrics in {root}.')
            records.append(record)
    expected = (
        sum(len(settings['subjects']) // size for size in
            settings['sessions_per_fit']) *
        len(settings['models']) * settings['seeds']
    )
    if len(records) != expected:
        raise ValueError(
            f'Expected {expected} complete fits, got {len(records)}.'
        )
    return pd.DataFrame.from_records(records)


def summarize_groups(results: pd.DataFrame) -> pd.DataFrame:
    """Average equal-size groups within each seed, then seeds.

    Parameters
    ----------
    results : pandas.DataFrame
        Complete per-fit metrics from ``read_group_results``.

    Returns
    -------
    pandas.DataFrame
        One row per model and sessions-per-fit count.
    """
    settings = load_ahrens_session_campaign()
    keys = ['seed', 'model', 'sessions_per_fit']
    by_seed = results.groupby(keys).agg(
        test_score=('test_score', 'mean'),
        test_loss=('test_loss', 'mean'),
        train_step_flops=('train_step_flops', 'mean'),
        trainable_params=('trainable_params', 'mean'),
        train_examples_per_fit=('train_examples_per_fit', 'mean'),
        cohort_train_examples=('train_examples_per_fit', 'sum'),
        training_flops_per_fit_estimate=(
            'training_flops_per_fit_estimate', 'mean'
        ),
        cohort_training_flops_estimate=(
            'training_flops_per_fit_estimate', 'sum'
        ),
        group_count=('group', 'size'),
    ).reset_index()
    for size in settings['sessions_per_fit']:
        rows = by_seed.loc[by_seed['sessions_per_fit'] == size]
        expected_groups = len(settings['subjects']) // size
        if len(rows) != settings['seeds'] * len(settings['models']):
            raise ValueError(f'Missing seed/model rows for size {size}.')
        if not (rows['group_count'] == expected_groups).all():
            raise ValueError(
                f'Expected {expected_groups} groups at size {size}.'
            )
    summaries = []
    for (model, size), rows in by_seed.groupby(
        ['model', 'sessions_per_fit']
    ):
        if len(rows) != settings['seeds']:
            raise ValueError(
                f'Expected {settings["seeds"]} seeds for '
                f'{model} at size {size}.'
            )
        summary = {
            'model': model,
            'sessions_per_fit': size,
            'group_count': len(settings['subjects']) // size,
            'seed_count': len(rows),
        }
        for metric in METRICS:
            values = rows[metric].to_numpy(dtype=float)
            summary[f'{metric}_mean'] = values.mean()
            summary[f'{metric}_sem'] = (
                values.std(ddof=1) / np.sqrt(len(values))
            )
        for field in (
            'train_step_flops', 'trainable_params',
            'train_examples_per_fit', 'cohort_train_examples',
            'training_flops_per_fit_estimate',
            'cohort_training_flops_estimate',
        ):
            summary[f'{field}_mean'] = rows[field].mean()
        summaries.append(summary)
    table = pd.DataFrame.from_records(summaries)
    if table.empty or not np.isfinite(
        table.select_dtypes(include='number')
    ).all().all():
        raise ValueError('Session scaling summary is empty or non-finite.')
    return table.sort_values(['model', 'sessions_per_fit'])


def analyze_ahrens_sessions() -> Path:
    """Save score/loss figure and derived compute summary.

    Returns
    -------
    Path
        Absolute output directory under the configured figure root.
    """
    summary = summarize_groups(read_group_results())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = Path(FIG_DIR) / 'ahrens_session_lowpass_six' / stamp
    root.mkdir(parents=True)
    summary.to_csv(root / 'summary.csv', index=False)
    plot_scaling_panels(
        summary, 'sessions_per_fit', 'Training sessions per fit',
        'Ahrens low-pass 512-PC session scaling (six subjects)',
        root / 'session_scaling.pdf',
    )
    print(f'Saved Ahrens session figures: {root.resolve()}')
    return root
