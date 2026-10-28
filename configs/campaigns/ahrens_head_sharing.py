"""Load the nested Ahrens POCO sharing factorial.

Input is the adjacent JSON with four explicit, nested subject groups,
three seeds, original fit budget, and the two sharing scopes. Output is
validated scientific campaign settings with no machine paths.
"""

import json
from pathlib import Path

from configs.campaigns.ahrens_session_extended import (
    load_ahrens_session_extended_campaign,
)
from utils.ahrens_sessions import sorted_subject_names

CAMPAIGN_PATH = (
    Path(__file__).resolve().parent / 'ahrens_head_sharing.json'
)
EXPECTED_SIZES = (3, 6, 9, 18)
EXPECTED_SEEDS = 3
EXPECTED_STEPS = 10000
SCOPES = ['shared', 'session']


def load_ahrens_head_sharing_campaign() -> dict:
    """Validate the fixed nested cohort and scientific settings.

    Returns
    -------
    dict
        Explicit subject groups and three-seed sharing policy settings.
    """
    with CAMPAIGN_PATH.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    required = {
        'filter', 'model', 'seeds', 'max_batch',
        'groups', 'sharing_scopes',
    }
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f'Expected campaign fields {sorted(required)}.'
        )
    original = load_ahrens_session_extended_campaign()
    if (settings['filter'] != original['filter'] or
            settings['model'] != 'POCO' or
            settings['seeds'] != EXPECTED_SEEDS or
            settings['max_batch'] != EXPECTED_STEPS or
            settings['sharing_scopes'] != SCOPES):
        raise ValueError(
            'Expected POCO low-pass, three seeds, 10000 steps, '
            'and shared/session policies.'
        )
    groups = settings['groups']
    if not isinstance(groups, dict) or set(groups) != {
        str(size) for size in EXPECTED_SIZES
    }:
        raise ValueError(
            f'Expected groups for sizes {EXPECTED_SIZES}.'
        )
    previous = set()
    for size in EXPECTED_SIZES:
        group = groups[str(size)]
        candidates = original['groups'][str(size)]
        if (not isinstance(group, list) or len(group) != size or
                group != sorted_subject_names(group) or
                group != candidates[0]):
            raise ValueError(
                f'Expected original first size-{size} group, '
                f'got {group!r}.'
            )
        if not previous <= set(group):
            raise ValueError(
                f'Size-{size} group loses earlier subjects.'
            )
        previous = set(group)
    return settings
