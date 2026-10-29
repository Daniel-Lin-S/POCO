"""Validate Ahrens sharing campaign settings.

Input is one dedicated JSON per sharing campaign. Output is validated
scientific settings with no machine paths. The campaigns use the same
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
LOCAL_MLP_PATH = (
    CAMPAIGN_DIR / 'ahrens_local_mlp_scaling.json'
)
EXPECTED_SIZES = (3, 6, 9, 18)
COMPOSITION_COUNTS = {3: 1, 6: 3, 9: 3, 18: 1}
EXPECTED_SEEDS = 3
EXPECTED_STEPS = 10000
SCOPES = ['shared', 'session']
COMPARISON_POLICIES = ['full', 'input_only']


def _load_base(
    path: Path, required: set[str],
    expected_model: str = 'POCO',
) -> tuple[dict, dict]:
    """Read one sharing campaign and validate common settings.

    Parameters
    ----------
    path : pathlib.Path
        Campaign-specific JSON file.
    required : set[str]
        Expected scientific fields.
    expected_model : str, optional
        Required model label, default 'POCO'.

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
            settings['model'] != expected_model or
            settings['seeds'] != EXPECTED_SEEDS or
            settings['max_batch'] != EXPECTED_STEPS):
        raise ValueError(
            f'{path.name} requires low-pass {expected_model}, three seeds, '
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


def _validate_composition_groups(
    groups: dict, original: dict, reference: set[str],
) -> None:
    """Validate the eight shared Ahrens composition groups.

    Parameters
    ----------
    groups : dict
        Campaign groups keyed by session count.
    original : dict
        Established extended campaign settings.
    reference : set[str]
        Fixed evaluation subjects.
    """
    _validate_group_sizes(groups)
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
    _validate_composition_groups(
        settings['groups'], original,
        set(settings['reference_subjects']),
    )
    return settings


def load_ahrens_local_mlp_campaign() -> dict:
    """Validate the local-MLP campaign's separate scientific JSON.

    Returns
    -------
    dict
        Local-MLP groups, subject controls, and training settings.
    """
    required = {
        'filter', 'model', 'seeds', 'max_batch',
        'reference_subjects', 'groups',
        'output_projection_scope',
    }
    settings, original = _load_base(
        LOCAL_MLP_PATH, required, expected_model='MLP',
    )
    comparison = load_ahrens_input_projection_campaign()
    if (settings['reference_subjects'] !=
            comparison['reference_subjects'] or
            settings['groups'] != comparison['groups'] or
            settings['output_projection_scope'] != 'session'):
        raise ValueError(
            'Local-MLP groups must match the input-sharing '
            'comparison with private recording output heads.'
        )
    _validate_composition_groups(
        settings['groups'], original,
        set(settings['reference_subjects']),
    )
    return settings


def load_ahrens_five_triple_campaign() -> dict:
    """Read fixed triples and three nested composition paths per triple.

    Returns
    -------
    dict
        Scientific settings, triple membership, and nested group codes.
    """
    path = CAMPAIGN_DIR / 'ahrens_five_triple_scaling.json'
    settings = json.loads(path.read_text())
    original = load_ahrens_session_extended_campaign()
    triples = settings['triples']
    subjects = [s for group in triples.values() for s in group]
    if (set(triples) != set('RABCDE') or
            any(len(g) != 3 for g in triples.values()) or
            len(set(subjects)) != len(subjects) or
            set(subjects) != set(original['subjects']) or
            triples['R'] != original['reference_subjects']):
        raise ValueError('Expected six disjoint established triples.')
    if (settings['filter'] != 'lowpass' or
            settings['seeds'] != EXPECTED_SEEDS or
            settings['max_batch'] != EXPECTED_STEPS or
            settings['models'] != ['POCO', 'MLP'] or
            set(settings['paths']) != set('ABCDE')):
        raise ValueError('Invalid fixed-triple scientific settings.')
    for reference, paths in settings['paths'].items():
        if (len(paths) != 3 or
                len({p[0] for p in paths}) != 3 or
                len({p[1] for p in paths}) != 3):
            raise ValueError('Expected three distinct composition paths.')
        for small, large in paths:
            if (len(set(small)) != 2 or len(set(large)) != 3 or
                    not {reference} <= set(small) < set(large) or
                    not set(large) <= set(triples)):
                raise ValueError(f'Invalid nested path: {small}, {large}.')
    return settings


def five_triple_groups(settings: dict, reference: str) -> list[list[str]]:
    """Return canonical 3/6/9/18-session groups for one fixed triple.

    Parameters
    ----------
    settings : dict
        Validated five-triple campaign configuration.
    reference : str
        Evaluation triple code, A through E.

    Returns
    -------
    list[list[str]]
        Eight canonical subject groups in increasing size order.
    """
    paths = settings['paths'][reference]
    codes = [reference] + [p[0] for p in paths]
    codes += [p[1] for p in paths] + ['RABCDE']
    return [sorted_subject_names([
        subject for code in group
        for subject in settings['triples'][code]
    ]) for group in codes]
