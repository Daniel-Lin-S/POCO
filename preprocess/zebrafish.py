"""Convert raw zebrafish recordings into processed neural archives.

Input for Ahrens is a raw root containing ``subject_<id>/TimeSeries.h5``.
Output is compressed ``subject_<id>.npz`` in the configured processed
directory, containing float32 M and PC, valid_indices, source_subject,
split-fit metadata, and PCA settings. Lifecycle events go to an
ignored JSONL file in ``logs`` with completed, remaining, failed,
and total counts.
The other zebrafish dataset retains its existing raw HDF5 layout.
"""

import h5py
import fcntl
import json
import os
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
import numpy as np

from configs.config_global import (
    ZEBRAFISH_RAW_DIR, EXP_TYPES, ZEBRAFISH_PROCESSED_DIR,
    RAW_DATA_SUFFIX, ROOT_DIR,
)
from utils.data_utils import get_exp_names
from configs.ahrens_preprocess import AhrensPreprocessConfig
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

AHRENS_LOG_NAME = 'ahrens_preprocess.jsonl'


def save_processed_subject(
    output: Path, processed: dict[str, object],
) -> None:
    """Publish one complete compressed archive without replacing another.

    Parameters
    ----------
    output : Path
        Absolute path to a new ``subject_<id>.npz`` archive.
    processed : dict[str, object]
        M and PC float32 arrays plus subject and split metadata.
    """
    descriptor, pending = tempfile.mkstemp(
        prefix=f'.{output.stem}.', suffix='.pending', dir=output.parent
    )
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **processed)
        os.link(pending, output)
    finally:
        os.unlink(pending)


def log_ahrens_progress(
    stream: TextIO, filter_mode: str, pca_solver: str,
    subject: str | None, status: str,
    completed: int, total: int, failures: int,
    detail: str | None = None,
) -> None:
    """Append one lifecycle event for the current filter condition.

    Parameters
    ----------
    stream : TextIO
        Open JSONL lifecycle log under the repository logs directory.
    filter_mode : str
        Preprocessing filter condition.
    pca_solver : str
        PCA solver used for this run.
    subject : str or None
        Subject name for recording events, otherwise None.
    status : str
        Start, processing, completed, skipped, failed, or finished.
    completed : int
        Number of validated subject archives in this invocation.
    total : int
        Number of discovered raw subject directories.
    failures : int
        Number of subject failures in this invocation.
    detail : str or None, optional
        Error text for a failed recording, default None.
    """
    event = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'filter_mode': filter_mode,
        'pca_solver': pca_solver,
        'subject': subject,
        'status': status,
        'completed': completed,
        'remaining': total - completed,
        'failed': failures,
        'total': total,
    }
    if detail is not None:
        event['detail'] = detail
    stream.write(json.dumps(event) + '\n')
    stream.flush()


def process_ahrens_subject(
    subject_dir: Path, output_dir: Path, filter_mode: str,
    settings: AhrensPreprocessConfig,
) -> str:
    """Process or validate one subject, returning its lifecycle status.

    Parameters
    ----------
    subject_dir : Path
        Raw ``subject_<id>`` directory with TimeSeries.h5.
    output_dir : Path
        Directory for the filter-specific NPZ archive.
    filter_mode : str
        Neural filter condition.
    settings : AhrensPreprocessConfig
        Validated PCA and storage settings.

    Returns
    -------
    str
        ``completed`` for a new archive or ``skipped`` for a valid one.
    """
    source = subject_dir / 'TimeSeries.h5'
    if not source.is_file():
        raise FileNotFoundError(
            f'Ahrens subject is missing TimeSeries.h5: {source}'
        )
    output = output_dir / f'{subject_dir.name}.npz'
    if output.exists():
        try:
            validate_processed_subject(output, filter_mode)
        except ValueError:
            quarantine = output.with_name(
                f'{output.name}.invalid.'
                f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
            )
            output.rename(quarantine)
        else:
            validate_processed_subject(
                output, filter_mode, settings.pc_dim,
                settings.pca_solver, settings.pca_seed,
            )
            return 'skipped'
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
        pc_dim=settings.pc_dim,
        exp_name=subject_dir.name,
        filter_mode=filter_mode,
        plot_diagnostics=False,
        pca_solver=settings.pca_solver,
        pca_random_state=settings.pca_seed,
    )
    for key in ('M', 'PC'):
        processed[key] = np.asarray(
            processed[key], dtype=settings.storage_dtype
        )
    processed['source_subject'] = subject_dir.name
    processed['filter_mode'] = filter_mode
    processed['pc_dim'] = settings.pc_dim
    processed['pca_solver'] = settings.pca_solver
    processed['pca_seed'] = settings.pca_seed
    save_processed_subject(output, processed)
    return 'completed'


def process_zebrafish_ahrens_activity(
    filter_mode: str,
    settings: AhrensPreprocessConfig,
) -> None:
    """Process all raw Ahrens subjects with resumable lifecycle logging.

    Parameters
    ----------
    filter_mode : str
        Neural activity filter from the local campaign settings.
    settings : AhrensPreprocessConfig
        Validated PCA and storage settings.

    Raises
    ------
    RuntimeError
        If one or more subjects failed after remaining subjects ran.
    """
    if filter_mode not in settings.filters:
        raise ValueError(
            f'Filter {filter_mode!r} is absent from configured '
            f'filters {settings.filters!r}.'
        )
    subjects = raw_subject_directories()
    output_dir = processed_directory(filter_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(ROOT_DIR) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    log_path = log_dir / AHRENS_LOG_NAME
    lock_path = log_dir / f'ahrens_preprocess_{filter_mode}.lock'
    with ExitStack() as stack:
        lock = stack.enter_context(lock_path.open('a'))
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f'Ahrens {filter_mode} preprocessing is already running.'
            ) from error
        pending = list(output_dir.glob('.*.pending'))
        for path in pending:
            path.unlink()
        stream = stack.enter_context(
            log_path.open('a', encoding='utf-8', buffering=1)
        )
        log_ahrens_progress(
            stream, filter_mode, settings.pca_solver, None, 'started',
            completed, len(subjects), failures,
            f'Removed {len(pending)} stale pending archives.',
        )
        for subject_dir in subjects:
            log_ahrens_progress(
                stream, filter_mode, settings.pca_solver,
                subject_dir.name, 'processing',
                completed, len(subjects), failures,
            )
            try:
                status = process_ahrens_subject(
                    subject_dir, output_dir, filter_mode, settings
                )
            except Exception as error:
                failures += 1
                log_ahrens_progress(
                    stream, filter_mode, settings.pca_solver,
                    subject_dir.name, 'failed',
                    completed, len(subjects), failures,
                    f'{type(error).__name__}: {error}',
                )
                continue
            completed += 1
            log_ahrens_progress(
                stream, filter_mode, settings.pca_solver,
                subject_dir.name, status,
                completed, len(subjects), failures,
            )
        log_ahrens_progress(
            stream, filter_mode, settings.pca_solver, None, 'finished',
            completed, len(subjects), failures,
        )
    print(f'Ahrens preprocessing lifecycle log: {log_path}')
    if failures:
        raise RuntimeError(
            f'Ahrens {filter_mode} preprocessing failed for '
            f'{failures} of {len(subjects)} subjects; see {log_path}.'
        )
