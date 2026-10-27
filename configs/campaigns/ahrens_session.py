"""Read the six-subject Ahrens session-scaling campaign definition.

Input is ``configs/campaigns/ahrens_session_lowpass_six.json`` with
explicit subject names, sessions per fit, models, seed count, filter,
and training steps. Output is a validated scientific settings mapping.
No machine paths are stored in the JSON file.
"""

import json
from pathlib import Path

from utils.ahrens_sessions import (
    processed_subject_names, sorted_subject_names,
)

CAMPAIGN_PATH = (
    Path(__file__).resolve().parent /
    'ahrens_session_lowpass_six.json'
)
EXPECTED_SUBJECTS = 6
EXPECTED_SCALES = [1, 2, 3, 6]
EXPECTED_MODELS = {'POCO', 'NLinear'}
EXPECTED_FILTER = 'lowpass'


def load_ahrens_session_campaign() -> dict:
    """Validate the fixed cohort and equal-size scale grid.

    Returns
    -------
    dict
        Scientific campaign settings with processed subject names.
    """
    with CAMPAIGN_PATH.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    required = {
        'subjects', 'sessions_per_fit', 'models', 'seeds',
        'filter', 'max_batch',
    }
    if not isinstance(settings, dict) or required - settings.keys():
        raise ValueError(
            f'Session campaign requires fields {sorted(required)}.'
        )
    subjects = settings['subjects']
    if (not isinstance(subjects, list) or
            len(subjects) != EXPECTED_SUBJECTS or
            subjects != sorted_subject_names(subjects)):
        raise ValueError('Expected six unique subjects in numeric order.')
    missing = sorted(set(subjects) - set(
        processed_subject_names(settings['filter'])
    ))
    if missing:
        raise ValueError(f'Unprocessed Ahrens subjects: {missing}.')
    if settings['filter'] != EXPECTED_FILTER:
        raise ValueError('Session pilot requires low-pass activity.')
    if settings['sessions_per_fit'] != EXPECTED_SCALES:
        raise ValueError(
            f'Expected sessions-per-fit grid {EXPECTED_SCALES}.'
        )
    models = settings['models']
    if not isinstance(models, list) or set(models) != EXPECTED_MODELS:
        raise ValueError('Expected POCO and NLinear model labels.')
    for field in ('seeds', 'max_batch'):
        value = settings[field]
        if type(value) is not int or value < 1:
            raise ValueError(f'Expected positive {field}, got {value!r}.')
    return settings
