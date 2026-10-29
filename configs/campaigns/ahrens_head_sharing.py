"""Validate Ahrens POCO sharing campaign settings.

Input is one dedicated JSON per sharing campaign. Output is validated
scientific settings with no machine paths. Both campaigns use the same
subject-group and fit-resolution rules.
"""

import json
from pathlib import Path

from configs.campaigns.ahrens_session_extended import (
    load_ahrens_session_extended_campaign,
)
from utils.ahrens_sessions import sorted_subject_names

CAMPAIGN_DIR = Path(__file__).resolve().parent
FACTORIAL_PATH = CAMPAIGN_DIR / 'ahrens_head_sharing.json'
INPUT_PROJECTION_PATH = (
    CAMPAIGN_DIR / 'ahrens_input_projection_scaling.json'
)
EXPECTED_SIZES = (3, 6, 9, 18)
COMPOSITION_COUNTS = {3: 1, 6: 3, 9: 3, 18: 1}
EXPECTED_SEEDS = 3
EXPECTED_STEPS = 10000
SCOPES = ['shared', 'session']
COMPARISON_POLICIES = ['full', 'input_only']


def _load_base(
    path: Path, required: set[str],
) -> tuple[dict, dict]:
    """Read one sharing campaign and validate common settings.

    Parameters
    ----------
    path : pathlib.Path
        Campaign-specific JSON file.
    required : set[str]
        Expected scientific fields.

    Returns
    -------
    tuple[dict, dict]
        Campaign settings and established Ahrens cohort settings.
    """
    with path.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f'Expected fields {sorted(required)} in {path.name}.'
        )
    original = load_ahrens_session_extended_campaign()
    if (settings['filter'] != original['filter'] or
            settings['model'] != 'POCO' or
            settings['seeds'] != EXPECTED_SEEDS or
            settings['max_batch'] != EXPECTED_STEPS):
        raise ValueError(
            f'{path.name} requires low-pass POCO, three seeds, '
            'and 10000 training steps.'
        )
    return settings, original


def _validate_group(
    group: list[str], size: int, candidates: list[list[str]],
    reference: set[str],
) -> None:
    """Require one canonical group from the established cohort.

    Parameters
    ----------
    group : list[str]
        Subject names in numeric order.
    size : int
        Required subject count.
    candidates : list[list[str]]
        Established groups at this size.
    reference : set[str]
        Subjects required in every evaluated fit.
    """
    if (not isinstance(group, list) or len(group) != size or
            group != sorted_subject_names(group) or
            group not in candidates or not reference <= set(group)):
        raise ValueError(
            f'Invalid size-{size} Ahrens group: {group!r}.'
        )


def _validate_group_sizes(groups: dict) -> None:
    """Require the established four training-session sizes.

    Parameters
    ----------
    groups : dict
        Subject selections keyed by session count.
    """
    if not isinstance(groups, dict) or set(groups) != {
        str(size) for size in EXPECTED_SIZES
    }:
        raise ValueError(
            f'Expected group sizes {EXPECTED_SIZES}.'
        )


def load_ahrens_head_sharing_campaign() -> dict:
    """Validate the nested eight-policy factorial configuration.

    Returns
    -------
    dict
        Factorial scientific settings.
    """
    required = {
        'filter', 'model', 'seeds', 'max_batch',
        'groups', 'sharing_scopes',
    }
    settings, original = _load_base(FACTORIAL_PATH, required)
    if settings['sharing_scopes'] != SCOPES:
        raise ValueError('Expected shared/session head scopes.')
    groups = settings['groups']
    _validate_group_sizes(groups)
    reference = set(original['reference_subjects'])
    previous = set()
    for size in EXPECTED_SIZES:
        group = groups[str(size)]
        _validate_group(
            group, size, original['groups'][str(size)], reference,
        )
        if group != original['groups'][str(size)][0]:
            raise ValueError(
                f'Factorial size-{size} group must match '
                'the established first group.'
            )
        if not previous <= set(group):
            raise ValueError(
                f'Factorial size-{size} group loses earlier subjects.'
            )
        previous = set(group)
    return settings


def load_ahrens_input_projection_campaign() -> dict:
    """Validate the three composition paths and policy settings.

    Returns
    -------
    dict
        Input-only comparison scientific settings.
    """
    required = {
        'filter', 'model', 'seeds', 'max_batch',
        'reference_subjects', 'groups', 'policies',
    }
    settings, original = _load_base(
        INPUT_PROJECTION_PATH, required,
    )
    if (settings['reference_subjects'] !=
            original['reference_subjects'] or
            settings['policies'] != COMPARISON_POLICIES):
        raise ValueError(
            'Expected the established reference subjects and '
            'full/input-only sharing policies.'
        )
    groups = settings['groups']
    _validate_group_sizes(groups)
    reference = set(settings['reference_subjects'])
    for size in EXPECTED_SIZES:
        chosen = groups[str(size)]
        if (not isinstance(chosen, list) or
                len(chosen) != COMPOSITION_COUNTS[size]):
            raise ValueError(
                f'Expected {COMPOSITION_COUNTS[size]} '
                f'size-{size} groups.'
            )
        for group in chosen:
            _validate_group(
                group, size, original['groups'][str(size)],
                reference,
            )
        if len({tuple(group) for group in chosen}) != len(chosen):
            raise ValueError(f'Duplicate size-{size} groups.')
    for index in range(COMPOSITION_COUNTS[6]):
        if not set(groups['6'][index]) <= set(groups['9'][index]):
            raise ValueError(
                f'Size-9 path {index} loses size-6 subjects.'
            )
    return settings
