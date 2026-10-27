"""Read the four-subject Ahrens recording-length campaign definition.

Input is ``configs/campaigns/ahrens_length_lowpass_four.json``. It holds
subject names, nominal recording lengths in frames, model labels, seed
count, filtering condition, and training step budget. Output is a
validated settings mapping for fit configuration and analysis.
"""

import json
from pathlib import Path
from zipfile import ZipFile

from numpy.lib.format import (
    read_array_header_1_0, read_array_header_2_0, read_magic,
)

from configs.configs import DatasetConfig
from utils.ahrens_sessions import (
    processed_directory, processed_subject_names,
)

CAMPAIGN_PATH = (
    Path(__file__).resolve().parent /
    'ahrens_length_lowpass_four.json'
)
EXPECTED_SUBJECTS = 4
EXPECTED_LENGTHS = 6
EXPECTED_MODELS = {'POCO', 'NLinear'}
EXPECTED_FILTER = 'lowpass'
EXPECTED_PC_DIM = 512


def processed_pc_frames(subject: str, filter_type: str) -> int:
    """Read PC array dimensions from an NPZ header without loading data.

    Parameters
    ----------
    subject : str
        Processed Ahrens subject name.
    filter_type : str
        Required processing condition.

    Returns
    -------
    int
        Number of recorded frames in the PC array.
    """
    path = processed_directory(filter_type) / f'{subject}.npz'
    with ZipFile(path) as archive:
        with archive.open('PC.npy') as stream:
            version = read_magic(stream)
            reader = (
                read_array_header_1_0 if version == (1, 0)
                else read_array_header_2_0
            )
            shape, _, _ = reader(stream)
    if len(shape) != 2 or shape[0] != EXPECTED_PC_DIM:
        raise ValueError(
            f'Expected PC shape ({EXPECTED_PC_DIM}, time) in {path}, '
            f'got {shape}.'
        )
    return shape[1]


def load_ahrens_length_campaign() -> dict:
    """Return validated scientific settings for the Ahrens pilot.

    Returns
    -------
    dict
        Campaign fields from the JSON file, with all selected subjects
        available in the processed low-pass cohort.
    """
    with CAMPAIGN_PATH.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    required = {
        'subjects', 'nominal_lengths', 'models', 'seeds',
        'filter', 'max_batch',
    }
    if not isinstance(settings, dict) or required - settings.keys():
        raise ValueError(
            f'Campaign JSON requires fields {sorted(required)}.'
        )
    subjects = settings['subjects']
    lengths = settings['nominal_lengths']
    models = settings['models']
    if not all(isinstance(value, list) for value in
               (subjects, lengths, models)):
        raise ValueError('Subjects, nominal lengths, and models need lists.')
    if settings['filter'] != EXPECTED_FILTER:
        raise ValueError(
            f'Expected filter {EXPECTED_FILTER}, '
            f'got {settings["filter"]!r}.'
        )
    if len(subjects) != EXPECTED_SUBJECTS or len(set(subjects)) != len(
        subjects
    ):
        raise ValueError('Expected four distinct Ahrens subjects.')
    available = set(processed_subject_names(settings['filter']))
    missing = sorted(set(subjects) - available)
    if missing:
        raise ValueError(f'Unprocessed Ahrens subjects: {missing}.')
    if (len(lengths) != EXPECTED_LENGTHS or
            lengths != sorted(set(lengths)) or
            any(type(value) is not int or value < 1 for value in lengths)):
        raise ValueError('Expected six increasing positive length caps.')
    if set(models) != EXPECTED_MODELS or len(models) != len(
        EXPECTED_MODELS
    ):
        raise ValueError('Expected POCO and NLinear once each.')
    for field in ('seeds', 'max_batch'):
        value = settings[field]
        if type(value) is not int or value < 1:
            raise ValueError(f'Expected positive {field}, got {value!r}.')
    train_fraction = DatasetConfig().train_split
    required_train_frames = int(max(lengths) * train_fraction)
    for subject in subjects:
        frames = processed_pc_frames(subject, settings['filter'])
        available_train_frames = int(frames * train_fraction)
        if available_train_frames < required_train_frames:
            raise ValueError(
                f'{subject} has {available_train_frames} train frames; '
                f'{required_train_frames} are required for the largest cap.'
            )
    return settings
