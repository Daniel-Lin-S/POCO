"""Convert raw zebrafish recordings into processed neural archives.

Input for Ahrens is a raw root containing ``subject_<id>/TimeSeries.h5``.
Output is ``subject_<id>.npz`` in the configured processed directory,
containing M, PC, valid_indices, source_subject, and split-fit metadata.
The other zebrafish dataset retains its existing raw HDF5 layout.
"""

import h5py
import os
import tempfile
from pathlib import Path
import numpy as np

from configs.config_global import \
    ZEBRAFISH_RAW_DIR, EXP_TYPES, ZEBRAFISH_PROCESSED_DIR, RAW_DATA_SUFFIX
from utils.data_utils import get_exp_names
from .utils import process_data_matrix
from utils.ahrens_sessions import (
    processed_directory, raw_subject_directories,
    validate_processed_subject,
)

def process_zebrafish_activity():
    exp_names = get_exp_names()
    os.makedirs(ZEBRAFISH_PROCESSED_DIR, exist_ok=True)
    normalize_mode = 'zscore'

    brain_areas = [ 
        'in_l_LHb', 'in_l_MHb', 'in_l_ctel', 'in_l_dthal', 'in_l_gc', 'in_l_raphe', 'in_l_tel', 'in_l_vent', 'in_l_vthal',
        'in_r_LHb', 'in_r_MHb', 'in_r_ctel', 'in_r_dthal', 'in_r_gc', 'in_r_raphe', 'in_r_tel', 'in_r_vent', 'in_r_vthal'
    ]

    curves = []

    for exp_type in EXP_TYPES:
        for exp_name in exp_names[exp_type]:

            filename = os.path.join(ZEBRAFISH_RAW_DIR, exp_name + RAW_DATA_SUFFIX)
            out_filename = os.path.join(ZEBRAFISH_PROCESSED_DIR, exp_name)

            with h5py.File(filename, "r") as f:

                data_dict = {}
                for key in f.keys():
                    data_dict[key] = np.array(f[key])

                roi_indices = np.zeros(data_dict['M'].shape[0], dtype=bool)
                for area in brain_areas:
                    assert area in data_dict, f"Area {area} not found in data"
                    roi_indices |= data_dict[area]

                return_dict = process_data_matrix(
                    data_dict['M'], 
                    fig_dir='preprocess/spontaneous_zebrafish',
                    roi_indices=None,
                    divide_baseline=True,
                    normalize_mode=normalize_mode,
                    n_clusers=[],
                    exp_name=exp_name
                )
                data_dict.update(return_dict)
                
                np.savez(out_filename, **data_dict)

AHRENS_PREPROCESS_PC_DIM = 2048


def save_processed_subject(
    output: Path, processed: dict[str, object],
) -> None:
    """Publish one complete subject archive without replacing another.

    Parameters
    ----------
    output : Path
        Absolute path to a new ``subject_<id>.npz`` archive.
    processed : dict[str, object]
        M, PC, source_subject, and split-fit metadata arrays.
    """
    descriptor, pending = tempfile.mkstemp(
        prefix=f'.{output.stem}.', suffix='.pending', dir=output.parent
    )
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez(stream, **processed)
        os.link(pending, output)
    finally:
        os.unlink(pending)


def process_zebrafish_ahrens_activity(
    filter_mode: str = 'none',
) -> None:
    """Process each named Ahrens subject without renumbering sessions.

    Parameters
    ----------
    filter_mode : str, optional
        Neural activity filter, default 'none'.

    Raises
    ------
    FileNotFoundError
        If a discovered subject lacks its TimeSeries.h5 file.
    ValueError
        If an existing subject archive is incomplete or invalid.
    """
    subjects = raw_subject_directories()
    output_dir = processed_directory(filter_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_dir in subjects:
        source = subject_dir / 'TimeSeries.h5'
        if not source.is_file():
            raise FileNotFoundError(
                f'Ahrens subject is missing TimeSeries.h5: {source}'
            )
        output = output_dir / f'{subject_dir.name}.npz'
        if output.exists():
            validate_processed_subject(output, filter_mode)
            continue
        with h5py.File(source, 'r') as recording:
            if 'CellResp' not in recording:
                raise KeyError(f'Missing CellResp in {source}.')
            activity = np.asarray(recording['CellResp']).T
        if activity.ndim != 2 or min(activity.shape) < 1:
            raise ValueError(
                f'Expected nonempty 2D CellResp in {source}, '
                f'got {activity.shape}.'
            )
        processed = process_data_matrix(
            activity,
            f'preprocess/{filter_mode}/zebrafish_ahrens',
            pc_dim=AHRENS_PREPROCESS_PC_DIM,
            exp_name=subject_dir.name,
            filter_mode=filter_mode,
        )
        processed['source_subject'] = subject_dir.name
        processed['filter_mode'] = filter_mode
        save_processed_subject(output, processed)
