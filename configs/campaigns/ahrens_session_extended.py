"""Validate the 18-subject Ahrens session campaign.

Input is the sibling JSON file with explicit subject-group labels,
reference subjects, model labels, and training steps. Output is a
validated scientific settings mapping. No machine paths are stored.
"""

import json
from collections import Counter
from pathlib import Path

from configs.campaigns.ahrens_session import (
    load_ahrens_session_campaign,
)
from utils.ahrens_sessions import (
    processed_subject_names, sorted_subject_names,
)

CAMPAIGN_PATH = (
    Path(__file__).resolve().parent /
    'ahrens_session_lowpass_extended.json'
)
EXPECTED_SIZES = (1, 2, 3, 6, 9, 18)
GROUP_LIMIT = 6
PAIR_SIZE = 2
REFERENCE_SIZE = 3
EXPECTED_SUBJECTS = 18
OVERLAP_FREQUENCY = {6: (1, 2), 9: (2, 3)}


def _validate_groups(settings: dict, pilot: dict) -> None:
    """Require canonical groups and exact pilot reuse.

    Parameters
    ----------
    settings : dict
        Extended campaign settings with groups of subject names.
    pilot : dict
        Completed six-subject pilot settings.
    """
    subjects = settings['subjects']
    reference = settings['reference_subjects']
    groups = settings['groups']
    if set(groups) != {str(size) for size in EXPECTED_SIZES}:
        raise ValueError(
            f'Expected Ahrens group sizes {EXPECTED_SIZES}.'
        )
    pilot_subjects = pilot['subjects']
    pilot_groups = {
        1: [[subject] for subject in pilot_subjects],
        2: [pilot_subjects[i:i + PAIR_SIZE]
            for i in range(0, len(pilot_subjects), PAIR_SIZE)],
        3: [pilot_subjects[i:i + REFERENCE_SIZE]
            for i in range(0, len(pilot_subjects), REFERENCE_SIZE)],
        6: [pilot_subjects],
    }
    for size in EXPECTED_SIZES:
        selected = groups[str(size)]
        expected_count = 1 if size == EXPECTED_SUBJECTS else GROUP_LIMIT
        if not isinstance(selected, list) or len(selected) != expected_count:
            raise ValueError(
                f'Expected {expected_count} groups of size {size}.'
            )
        identities = []
        occurrences = Counter()
        for group in selected:
            if not isinstance(group, list):
                raise ValueError(
                    f'Expected subject group, got {group!r}.'
                )
            if (len(group) != size or
                    group != sorted_subject_names(group) or
                    not set(group) <= set(subjects)):
                raise ValueError(
                    f'Invalid size-{size} Ahrens group {group!r}.'
                )
            if size > REFERENCE_SIZE and not set(reference) <= set(group):
                raise ValueError(
                    f'Size-{size} group omits reference subjects: '
                    f'{group!r}.'
                )
            identities.append(tuple(group))
            occurrences.update(group)
        if len(set(identities)) != len(identities):
            raise ValueError(f'Duplicate size-{size} Ahrens groups.')
        if size >= REFERENCE_SIZE and set(occurrences) != set(subjects):
            raise ValueError(
                f'Size-{size} groups do not cover all subjects.'
            )
        if size in OVERLAP_FREQUENCY:
            lower, upper = OVERLAP_FREQUENCY[size]
            counts = [occurrences[subject] for subject in subjects
                      if subject not in reference]
            if (any(occurrences[subject] != GROUP_LIMIT
                    for subject in reference) or
                    min(counts) != lower or max(counts) != upper):
                raise ValueError(
                    f'Size-{size} groups have unbalanced subject '
                    f'frequencies: {dict(occurrences)}.'
                )
        for group in pilot_groups.get(size, []):
            if group not in selected:
                raise ValueError(
                    f'Size-{size} group loses pilot fit {group!r}.'
                )


def load_ahrens_session_extended_campaign() -> dict:
    """Return validated settings for 18 subjects and six group sizes.

    Returns
    -------
    dict
        Scientific settings with subject-group selectors by size.
    """
    with CAMPAIGN_PATH.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    pilot = load_ahrens_session_campaign()
    required = {
        'subjects', 'reference_subjects', 'models', 'seeds',
        'filter', 'max_batch', 'groups',
    }
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f'Expected extended campaign fields {sorted(required)}.'
        )
    subjects = settings['subjects']
    reference = settings['reference_subjects']
    if (not isinstance(subjects, list) or
            len(subjects) != EXPECTED_SUBJECTS or
            subjects != sorted_subject_names(subjects) or
            subjects != processed_subject_names(settings['filter'])):
        raise ValueError(
            'Expected all 18 processed low-pass Ahrens subjects '
            'in numeric order.'
        )
    if (not isinstance(reference, list) or
            len(reference) != REFERENCE_SIZE or
            reference != pilot['subjects'][:REFERENCE_SIZE]):
        raise ValueError(
            'Expected the first completed pilot triple as reference.'
        )
    for field in ('models', 'seeds', 'filter', 'max_batch'):
        if settings[field] != pilot[field]:
            raise ValueError(
                f'Extended {field} must match the reusable pilot.'
            )
    _validate_groups(settings, pilot)
    return settings
