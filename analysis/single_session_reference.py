"""Add fitted singleton controls to the saved Ahrens comparison.

Input is complete POCO and NLinear singleton fits for subjects 8, 09,
and 11, plus the complete extended campaign and its fixed-reference
summary. The target snapshot is selected by TARGET_SNAPSHOT below.
Output in that snapshot replaces fixed_reference_tests.pdf and adds two
tables. single_session_per_seed.csv has model, seed, the three source fit
IDs, and fixed-reference validation/test score and MSE. The summary CSV
has one row per model with seed count and metric means and SEMs.
Completed fit artifacts and per-fit diagnostic figures are read only.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.scaling_plots import plot_scaling_panels
from analysis.session_scaling_extended import (
    fixed_metrics, read_fixed_results, reference_baselines,
    require_matching_subject_protocol, subject_errors,
    summarize_fixed,
)
from configs.campaigns.ahrens_session_extended import (
    load_ahrens_session_extended_campaign,
)
from configs.config_global import FIG_DIR
from configs.configs import NeuralPredictionConfig
from configs.experiments import (
    ahrens_session_lowpass_extended,
    ahrens_single_subject_reference,
)
from utils.artifacts import fit_id, validate_complete
from utils.config_utils import configs_dict_unpack

TARGET_SNAPSHOT = '20261027T130941.832565Z'
REFERENCE_SIZE = 3
EXPECTED_MODELS = ('POCO', 'NLinear')
SEEDS = (0, 1)
PHASES = ('val', 'test')
METRICS = ('val_score', 'val_loss', 'test_score', 'test_loss')
SUMMARY_COLUMNS = ('model', 'sessions_per_fit')
VALUE_TOLERANCE = 1e-6


def singleton_metrics(
    configs: list[NeuralPredictionConfig],
    reference: list[str],
    all_configs: list[NeuralPredictionConfig],
) -> pd.DataFrame:
    """Aggregate three independently fitted subjects within each seed.

    Parameters
    ----------
    configs : list
        Resolved singleton control fits with complete artifacts.
    reference : list[str]
        Fixed subject names in canonical evaluation order.
    all_configs : list
        Extended campaign fits used to recover persistence baselines.

    Returns
    -------
    pandas.DataFrame
        One model/seed row with validation and test score and MSE.
    """
    expected = {
        (model, seed, subject)
        for model in EXPECTED_MODELS
        for seed in SEEDS
        for subject in reference
    }
    selected = {}
    for config in configs:
        subject = config.dataset_config[
            config.dataset_label[0]
        ].session_ids[0]
        key = (config.model_label, config.seed, subject)
        if key in selected:
            raise ValueError(f'Duplicate singleton control {key}.')
        if not validate_complete(config):
            raise ValueError(
                f'Incomplete singleton control: {config.save_path}.'
            )
        selected[key] = config
    if set(selected) != expected:
        raise ValueError(
            f'Singleton controls differ from the expected cells: '
            f'{set(selected) ^ expected}.'
        )
    copy_baselines = {
        phase: reference_baselines(
            all_configs, reference, phase,
        )
        for phase in PHASES
    }
    fixed_group = {}
    for config in all_configs:
        subjects = config.dataset_config[
            config.dataset_label[0]
        ].session_ids
        if subjects == reference:
            fixed_group[(config.model_label, config.seed)] = config
    if set(fixed_group) != {
        (model, seed) for model in EXPECTED_MODELS for seed in SEEDS
    }:
        raise ValueError(
            'Expected one fixed triple fit per model and seed.'
        )
    rows = []
    for model in EXPECTED_MODELS:
        for seed in SEEDS:
            row = {
                'model': model,
                'seed': seed,
                'source_fit_ids': json.dumps([
                    fit_id(selected[(model, seed, subject)])
                    for subject in reference
                ]),
            }
            group = fixed_group[(model, seed)]
            for phase in PHASES:
                group_errors = subject_errors(group, phase)
                errors = {}
                for subject in reference:
                    require_matching_subject_protocol(
                        selected[(model, seed, subject)],
                        group, subject,
                    )
                    error = subject_errors(
                        selected[(model, seed, subject)], phase,
                    )[subject]
                    if error[1] != group_errors[subject][1]:
                        raise ValueError(
                            f'{phase} prediction count differs for '
                            f'{model}, seed {seed}, {subject}.'
                        )
                    errors[subject] = error
                score, loss = fixed_metrics(
                    errors, reference,
                    copy_baselines[phase],
                )
                row[f'{phase}_score'] = score
                row[f'{phase}_loss'] = loss
            rows.append(row)
    result = pd.DataFrame(rows)
    if not np.isfinite(result[list(METRICS)]).all().all():
        raise ValueError(
            'Singleton fixed-reference metrics are non-finite.'
        )
    return result


def summarize_singletons(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize fitted singleton controls across two seeds.

    Parameters
    ----------
    per_seed : pandas.DataFrame
        Fixed-subject validation and test metrics by model and seed.

    Returns
    -------
    pandas.DataFrame
        One row per model with metric means and seed SEMs.
    """
    rows = []
    for model, group in per_seed.groupby('model'):
        if len(group) != 2 or set(group['seed']) != set(SEEDS):
            raise ValueError(
                f'Expected seeds {SEEDS} for {model}.'
            )
        row = {
            'model': model,
            'seed_count': len(group),
            'subjects_per_seed': REFERENCE_SIZE,
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_sem'] = float(
                values.std(ddof=1) / np.sqrt(values.size)
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    if (set(result['model']) != set(EXPECTED_MODELS) or
            not np.isfinite(
                result.select_dtypes(include='number')
            ).all().all()):
        raise ValueError(
            'Singleton summary is incomplete or non-finite.'
        )
    return result.sort_values('model')


def validated_existing_summary(
    all_configs: list[NeuralPredictionConfig],
    settings: dict, root: Path,
) -> pd.DataFrame:
    """Check the saved multi-session summary against complete fits.

    Parameters
    ----------
    all_configs : list
        Resolved extended campaign fits.
    settings : dict
        Validated campaign settings.
    root : pathlib.Path
        Existing fixed-reference comparison snapshot.

    Returns
    -------
    pandas.DataFrame
        Saved multi-session summary, verified against fit artifacts.
    """
    path = root / 'fixed_reference_summary.csv'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing fixed-reference summary: {path}.'
        )
    saved = pd.read_csv(path).sort_values(
        list(SUMMARY_COLUMNS)
    ).reset_index(drop=True)
    actual = summarize_fixed(
        read_fixed_results(
            all_configs, settings['reference_subjects'],
        ),
        settings,
    ).sort_values(
        list(SUMMARY_COLUMNS)
    ).reset_index(drop=True)
    if (list(saved.columns) != list(actual.columns) or
            len(saved) != len(actual) or
            not saved[list(SUMMARY_COLUMNS)].equals(
                actual[list(SUMMARY_COLUMNS)]
            )):
        raise ValueError(
            f'Saved fixed-reference table differs from '
            f'complete-fit membership: {path}.'
        )
    numeric = saved.select_dtypes(include='number').columns
    if not np.allclose(
        saved[numeric], actual[numeric],
        rtol=VALUE_TOLERANCE, atol=VALUE_TOLERANCE,
    ):
        raise ValueError(
            f'Saved fixed-reference values differ from '
            f'complete-fit metrics: {path}.'
        )
    return saved


def add_singleton_reference() -> Path:
    """Replace the comparison PDF and save singleton control tables.

    Returns
    -------
    pathlib.Path
        Absolute existing fixed-reference snapshot folder.
    """
    settings = load_ahrens_session_extended_campaign()
    root = (
        Path(FIG_DIR) / 'ahrens_session_lowpass_extended' /
        'fixed_reference' / TARGET_SNAPSHOT
    )
    figure = root / 'fixed_reference_tests.pdf'
    if not root.is_dir() or not figure.is_file():
        raise FileNotFoundError(
            f'Missing target fixed-reference figure: {figure}.'
        )
    all_configs = configs_dict_unpack(
        ahrens_session_lowpass_extended()
    )
    controls = configs_dict_unpack(
        ahrens_single_subject_reference()
    )
    per_seed = singleton_metrics(
        controls, settings['reference_subjects'],
        all_configs,
    )
    reference = summarize_singletons(per_seed)
    summary = validated_existing_summary(
        all_configs, settings, root,
    )
    pending_figure = root / '.fixed_reference_tests.pending.pdf'
    pending_tables = {
        root / 'single_session_per_seed.csv':
            root / '.single_session_per_seed.pending.csv',
        root / 'single_session_summary.csv':
            root / '.single_session_summary.pending.csv',
    }
    try:
        plot_scaling_panels(
            summary, 'sessions_per_fit',
            'Training sessions per fit',
            'Ahrens fixed subjects 8, 09, 11: multi-session fits '
            'and fitted single-subject controls',
            pending_figure, reference=reference,
        )
        per_seed.to_csv(
            pending_tables[
                root / 'single_session_per_seed.csv'
            ], index=False,
        )
        reference.to_csv(
            pending_tables[
                root / 'single_session_summary.csv'
            ], index=False,
        )
        for destination, pending in pending_tables.items():
            os.replace(pending, destination)
        os.replace(pending_figure, figure)
    finally:
        pending_figure.unlink(missing_ok=True)
        for pending in pending_tables.values():
            pending.unlink(missing_ok=True)
    print(
        'Saved fixed-reference comparison: '
        f'{root.resolve()}'
    )
    return root
