"""Analyze original and fixed-reference Ahrens session scaling.

Input is completed hash-named fits with test.txt, progress.txt, and
per-subject val_best_info.npy/test_best_info.npy error arrays. Output
under the artifact figure root is two comparison PDFs and CSV tables
for group results, fixed-reference results, and subject membership.
Source fit artifacts are read only.
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.scaling_plots import plot_scaling_panels
from analysis.session_scaling import read_group_results, summarize_groups
from configs.campaigns.ahrens_session_extended import (
    load_ahrens_session_extended_campaign,
)
from configs.config_global import FIG_DIR
from configs.configs import NeuralPredictionConfig
from configs.experiments import ahrens_session_lowpass_extended
from utils.artifacts import fit_id, validate_complete
from utils.config_utils import configs_dict_unpack

FIRST_FIXED_SIZE = 3
BASELINE_TOLERANCE = 1e-6
BASELINE_FIT_COUNT = 4
METRICS = ('test_score', 'test_loss', 'val_score', 'val_loss')


def subject_errors(
    config: NeuralPredictionConfig, phase: str,
) -> dict[str, tuple[float, float]]:
    """Return per-subject squared-error sums and prediction counts.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Complete fit with one dataset and ordered subject names.
    phase : str
        Validation or test split.

    Returns
    -------
    dict[str, tuple[float, float]]
        Subject to squared-error sum and prediction-element count.
    """
    if phase not in ('val', 'test'):
        raise ValueError(f'Expected val or test, got {phase!r}.')
    label = config.dataset_label[0]
    subjects = config.dataset_config[label].session_ids
    path = Path(config.save_path) / f'{phase}_best_info.npy'
    info = np.load(path, allow_pickle=True).item()
    if (not isinstance(info, dict) or
            any(key not in info for key in ('mse', 'pred_num')) or
            len(info['mse']) != len(subjects) or
            len(info['pred_num']) != len(subjects)):
        raise ValueError(
            f'Expected {len(subjects)} subject error arrays in {path}.'
        )
    results = {}
    for subject, error, count in zip(
        subjects, info['mse'], info['pred_num']
    ):
        if (error.ndim != 2 or count.ndim != 1 or
                error.shape[1] != count.size):
            raise ValueError(
                f'Expected (horizon, PCs) errors and (PCs,) counts '
                f'for {subject} in {path}.'
            )
        squared_error = float(error.sum())
        elements = float(count.sum() * error.shape[0])
        if (not np.isfinite((squared_error, elements)).all() or
                squared_error < 0 or elements <= 0):
            raise ValueError(
                f'Invalid {phase} error or count for {subject} in {path}.'
            )
        results[subject] = (squared_error, elements)
    return results


def reported_metrics(
    config: NeuralPredictionConfig, phase: str,
) -> tuple[float, float]:
    """Read score and loss at the fit's selected checkpoint.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Complete fit with progress and test tables.
    phase : str
        Validation or test split.

    Returns
    -------
    tuple[float, float]
        Split score and MSE loss.
    """
    label = config.dataset_label[0]
    root = Path(config.save_path)
    if phase == 'test':
        table = pd.read_table(root / 'test.txt')
        loss_column = 'TestLoss'
    elif phase == 'val':
        table = pd.read_table(root / 'progress.txt')
        loss_column = 'ValLoss'
        losses = table[loss_column].to_numpy(dtype=float)
        if not np.isfinite(losses).all():
            raise ValueError(
                f'Non-finite validation losses in {root}.'
            )
        table = table.iloc[[
            np.flatnonzero(losses == losses.min())[-1]
        ]]
    else:
        raise ValueError(f'Expected val or test, got {phase!r}.')
    score_column = f'{label}_{phase}_score'
    if len(table) != 1 or score_column not in table:
        raise ValueError(
            f'Expected one {phase} score row in {root}.'
        )
    values = (
        float(table[score_column].iloc[0]),
        float(table[loss_column].iloc[0]),
    )
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite {phase} metrics in {root}.')
    return values


def reference_baselines(
    configs: list[NeuralPredictionConfig],
    reference: list[str], phase: str,
) -> dict[str, float]:
    """Recover fixed-subject copy baselines from completed singletons.

    Parameters
    ----------
    configs : list[NeuralPredictionConfig]
        Campaign fits including reference-subject singletons.
    reference : list[str]
        Fixed subject names.
    phase : str
        Validation or test split.

    Returns
    -------
    dict[str, float]
        One validated copy-MSE baseline per reference subject.
    """
    estimates = defaultdict(list)
    for config in configs:
        label = config.dataset_label[0]
        subjects = config.dataset_config[label].session_ids
        if len(subjects) != 1 or subjects[0] not in reference:
            continue
        subject = subjects[0]
        squared_error, elements = subject_errors(
            config, phase
        )[subject]
        score, loss = reported_metrics(config, phase)
        mse = squared_error / elements
        if not np.isclose(mse, loss, rtol=BASELINE_TOLERANCE):
            raise ValueError(
                f'Saved {phase} loss disagrees with errors in '
                f'{config.save_path}.'
            )
        baseline = mse / (1 - score)
        if (score >= 1 or not np.isfinite(baseline) or
                baseline <= 0):
            raise ValueError(
                f'Invalid {phase} copy baseline for {subject}.'
            )
        estimates[subject].append(baseline)
    result = {}
    for subject in reference:
        values = estimates[subject]
        if len(values) != BASELINE_FIT_COUNT or not np.allclose(
            values, values[0], rtol=BASELINE_TOLERANCE,
        ):
            raise ValueError(
                f'Expected four matching {phase} baselines for '
                f'{subject}, got {values}.'
            )
        result[subject] = float(np.mean(values))
    return result


def fixed_metrics(
    errors: dict[str, tuple[float, float]],
    reference: list[str], baselines: dict[str, float],
) -> tuple[float, float]:
    """Return equal-subject score and pooled MSE for fixed subjects.

    Parameters
    ----------
    errors : dict[str, tuple[float, float]]
        Subject to squared-error sum and evaluated element count.
    reference : list[str]
        Subjects scored in every eligible fit.
    baselines : dict[str, float]
        Split-specific copy-MSE baseline by subject.

    Returns
    -------
    tuple[float, float]
        Equal-subject score and pooled prediction MSE.
    """
    if not set(reference) <= set(errors):
        raise ValueError(
            f'Fixed subjects {reference} absent from {sorted(errors)}.'
        )
    scores = []
    total_error = 0.0
    total_count = 0.0
    for subject in reference:
        squared_error, elements = errors[subject]
        total_error += squared_error
        total_count += elements
        scores.append(
            1 - squared_error / elements / baselines[subject]
        )
    values = float(np.mean(scores)), total_error / total_count
    if not np.isfinite(values).all():
        raise ValueError('Fixed-reference metrics are non-finite.')
    return values


def read_fixed_results(
    configs: list[NeuralPredictionConfig], reference: list[str],
) -> pd.DataFrame:
    """Read fixed validation and test metrics from eligible fit files.

    Parameters
    ----------
    configs : list[NeuralPredictionConfig]
        All resolved fits, including complete singletons.
    reference : list[str]
        Fixed three-subject validation and test cohort.

    Returns
    -------
    pandas.DataFrame
        One row per eligible fit with fixed-set metrics.
    """
    baselines = {
        phase: reference_baselines(configs, reference, phase)
        for phase in ('val', 'test')
    }
    records = []
    for config in configs:
        label = config.dataset_label[0]
        subjects = config.dataset_config[label].session_ids
        size = len(subjects)
        if size < FIRST_FIXED_SIZE:
            continue
        if not set(reference) <= set(subjects):
            if size == FIRST_FIXED_SIZE:
                continue
            raise ValueError(
                f'Fit {config.save_path} omits fixed subjects.'
            )
        if not validate_complete(config):
            raise ValueError(
                f'Incomplete fixed-reference fit: {config.save_path}.'
            )
        record = {
            'fit_id': fit_id(config),
            'seed': config.seed,
            'model': config.model_label,
            'group': label,
            'sessions_per_fit': size,
        }
        for phase in ('val', 'test'):
            score, loss = fixed_metrics(
                subject_errors(config, phase),
                reference, baselines[phase],
            )
            record[f'{phase}_score'] = score
            record[f'{phase}_loss'] = loss
            if size == FIRST_FIXED_SIZE and subjects == reference:
                reported = reported_metrics(config, phase)
                if not np.allclose(
                    (score, loss), reported,
                    rtol=BASELINE_TOLERANCE,
                ):
                    raise ValueError(
                        f'Fixed {phase} metrics disagree with '
                        f'{config.save_path}.'
                    )
        records.append(record)
    if not records:
        raise ValueError('No fixed-reference fits were found.')
    return pd.DataFrame.from_records(records)


def summarize_fixed(results: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Average overlapping groups within each seed, then seeds.

    Parameters
    ----------
    results : pandas.DataFrame
        Per-fit fixed-reference validation and test metrics.
    settings : dict
        Campaign groups, models, and seed count.

    Returns
    -------
    pandas.DataFrame
        One model/size row with seed means and standard errors.
    """
    keys = ['seed', 'model', 'sessions_per_fit']
    by_seed = results.groupby(keys)[list(METRICS)].mean().reset_index()
    rows = []
    for (model, size), group in by_seed.groupby(
        ['model', 'sessions_per_fit']
    ):
        source = results.loc[
            (results['model'] == model) &
            (results['sessions_per_fit'] == size)
        ]
        reference = set(settings['reference_subjects'])
        expected = sum(
            reference <= set(subjects)
            for subjects in settings['groups'][str(size)]
        )
        if (len(group) != settings['seeds'] or
                len(source) != expected * settings['seeds']):
            raise ValueError(
                f'Expected {expected} groups per seed for '
                f'{model} at size {size}.'
            )
        row = {
            'model': model,
            'sessions_per_fit': size,
            'group_count': expected,
            'seed_count': settings['seeds'],
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = values.mean()
            row[f'{metric}_sem'] = (
                values.std(ddof=1) / np.sqrt(values.size)
            )
        rows.append(row)
    summary = pd.DataFrame.from_records(rows)
    expected_rows = sum(
        int(size) >= FIRST_FIXED_SIZE
        for size in settings['groups']
    ) * len(settings['models'])
    if (len(summary) != expected_rows or
            not np.isfinite(
                summary.select_dtypes(include='number')
            ).all().all()):
        raise ValueError(
            'Fixed-reference summary is incomplete or non-finite.'
        )
    return summary.sort_values(
        ['model', 'sessions_per_fit']
    )


def membership_table(settings: dict) -> pd.DataFrame:
    """Record group membership and repeat count once per subject.

    Parameters
    ----------
    settings : dict
        Explicit subject groups at each training-session count.

    Returns
    -------
    pandas.DataFrame
        Subject, session count, group selectors, and repeat count.
    """
    rows = []
    for size, selectors in settings['groups'].items():
        memberships = defaultdict(list)
        for group in selectors:
            label = '+'.join(group)
            for subject in group:
                memberships[subject].append(label)
        for subject, groups in memberships.items():
            rows.append({
                'sessions_per_fit': int(size),
                'subject': subject,
                'subject_group_occurrences': len(groups),
                'groups': '|'.join(groups),
            })
    if not rows:
        raise ValueError('No campaign subject memberships were found.')
    return pd.DataFrame.from_records(rows).sort_values(
        ['sessions_per_fit', 'subject']
    )


def analyze_ahrens_session_extended() -> Path:
    """Save group and fixed-reference comparisons in one folder.

    Returns
    -------
    pathlib.Path
        Absolute comparison artifact folder under the figure root.
    """
    settings = load_ahrens_session_extended_campaign()
    factory = ahrens_session_lowpass_extended
    configs = configs_dict_unpack(factory())
    original = summarize_groups(
        read_group_results(settings, factory), settings
    )
    fixed_fits = read_fixed_results(
        configs, settings['reference_subjects']
    )
    fixed = summarize_fixed(fixed_fits, settings)
    membership = membership_table(settings)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root = Path(FIG_DIR) / 'ahrens_session_lowpass_extended' / stamp
    root.mkdir(parents=True)
    original.to_csv(root / 'original_summary.csv', index=False)
    fixed_fits.to_csv(root / 'fixed_reference_per_fit.csv', index=False)
    fixed.to_csv(root / 'fixed_reference_summary.csv', index=False)
    membership.to_csv(root / 'group_membership.csv', index=False)
    plot_scaling_panels(
        original, 'sessions_per_fit', 'Training sessions per fit',
        'Ahrens low-pass 512-PC scaling: group tests',
        root / 'original_group_tests.pdf',
    )
    plot_scaling_panels(
        fixed, 'sessions_per_fit', 'Training sessions per fit',
        'Ahrens low-pass 512-PC scaling: fixed subjects 8, 09, 11',
        root / 'fixed_reference_tests.pdf',
    )
    print(f'Saved Ahrens session comparisons: {root.resolve()}')
    return root
