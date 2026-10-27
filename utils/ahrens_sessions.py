"""Resolve Ahrens recording names across raw and processed directories.

Raw input contains ``subject_<integer>/TimeSeries.h5`` directories.
Processed input contains ``subject_<integer>.npz`` archives with ``M``,
``PC``, split-fit metadata, and PCA settings. Outputs are subject names
sorted by numeric
identifier, retaining zero padding from the source directory name.
"""

from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np

from configs.config_global import (
    ZEBRAFISH_AHRENS_PROCESSED_DIR, ZEBRAFISH_AHRENS_RAW_DIR,
)

SUBJECT_PATTERN = re.compile(r'^subject_([0-9]+)$')


def subject_number(name: str) -> int:
    """Extract a subject number from a ``subject_<integer>`` name.

    Parameters
    ----------
    name : str
        Raw directory or processed archive stem.

    Returns
    -------
    int
        Positive subject number.
    """
    match = SUBJECT_PATTERN.fullmatch(name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(
            f'Expected subject_<positive integer>, got {name!r}.'
        )
    return int(match.group(1))


def sorted_subject_names(names: list[str]) -> list[str]:
    """Sort names and reject ambiguous zero-padded duplicates.

    Parameters
    ----------
    names : list[str]
        Subject directory names or archive stems.

    Returns
    -------
    list[str]
        Names in numeric subject order.
    """
    by_number = {}
    for name in names:
        number = subject_number(name)
        if number in by_number:
            raise ValueError(
                f'Subject {number} appears as both '
                f'{by_number[number]!r} and {name!r}.'
            )
        by_number[number] = name
    return [by_number[number] for number in sorted(by_number)]


def raw_subject_directories() -> list[Path]:
    """List raw Ahrens directories with subject names.

    Returns
    -------
    list[Path]
        Absolute directories in numeric subject order.
    """
    root = Path(ZEBRAFISH_AHRENS_RAW_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f'Ahrens raw directory is missing: {root}')
    directories = {
        path.name: path for path in root.iterdir()
        if path.is_dir() and SUBJECT_PATTERN.fullmatch(path.name)
    }
    if not directories:
        raise ValueError(f'No subject directories found in {root}.')
    return [directories[name] for name in sorted_subject_names(
        list(directories)
    )]


def processed_directory(filter_type: str = 'none') -> Path:
    """Locate Ahrens archives for one filtering condition.

    Parameters
    ----------
    filter_type : str, optional
        Filter suffix, default 'none'.

    Returns
    -------
    Path
        Absolute processed directory.
    """
    base = Path(ZEBRAFISH_AHRENS_PROCESSED_DIR)
    return base if filter_type == 'none' else Path(
        f'{base}_{filter_type}'
    )


def validate_processed_subject(
    path: Path, filter_type: str,
    pc_dim: int | None = None, pca_solver: str | None = None,
    pca_seed: int | None = None,
) -> None:
    """Check the identity and split metadata of one complete archive.

    Parameters
    ----------
    path : Path
        Subject-named NPZ archive to inspect.
    filter_type : str
        Expected filter condition for this archive.
    pc_dim : int or None, optional
        Expected saved component count, default None.
    pca_solver : str or None, optional
        Expected PCA solver, default None.
    pca_seed : int or None, optional
        Expected PCA seed, default None.

    Raises
    ------
    ValueError
        If the archive is unreadable or does not match its identity.
    """
    required = {
        'M', 'PC', 'valid_indices', 'source_subject', 'filter_mode',
        'fit_split', 'fit_train_split', 'fit_val_split',
        'fit_filter_isolated', 'pc_dim', 'pca_solver', 'pca_seed',
    }
    try:
        with np.load(path) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f'Missing keys {sorted(missing)}.')
            actual = archive['source_subject'].item()
            actual_filter = archive['filter_mode'].item()
            split = archive['fit_split'].item()
            isolated = archive['fit_filter_isolated'].item()
            saved_pc_dim = archive['pc_dim'].item()
            saved_solver = archive['pca_solver'].item()
            saved_seed = archive['pca_seed'].item()
            if actual != path.stem or actual_filter != filter_type:
                raise ValueError(
                    f'Expected subject {path.stem} with filter '
                    f'{filter_type}, got {actual} with {actual_filter}.'
                )
            if split != 'chronological_train_val' or isolated is not True:
                raise ValueError(
                    'Expected chronological train/validation fit with '
                    'split-local filtering.'
                )
            if not isinstance(saved_pc_dim, int) or saved_pc_dim < 1:
                raise ValueError(
                    f'Expected positive pc_dim, got {saved_pc_dim!r}.'
                )
            if saved_solver not in ('full', 'randomized'):
                raise ValueError(
                    f'Unexpected PCA solver {saved_solver!r}.'
                )
            if not isinstance(saved_seed, int):
                raise ValueError(
                    f'Expected integer pca_seed, got {saved_seed!r}.'
                )
            expected = (pc_dim, pca_solver, pca_seed)
            actual = (saved_pc_dim, saved_solver, saved_seed)
            for wanted, observed, key in zip(
                expected, actual, ('pc_dim', 'pca_solver', 'pca_seed')
            ):
                if wanted is not None and wanted != observed:
                    raise ValueError(
                        f'Expected {key}={wanted!r}, got {observed!r}.'
                    )
    except (OSError, ValueError, KeyError, BadZipFile, EOFError) as error:
        raise ValueError(
            f'Invalid processed Ahrens subject archive {path}: {error}'
        ) from error


def processed_subject_names(filter_type: str = 'none') -> list[str]:
    """List subject archives available for one filtering condition.

    Parameters
    ----------
    filter_type : str, optional
        Filter suffix, default 'none'.

    Returns
    -------
    list[str]
        Archive stems in numeric subject order.
    """
    root = processed_directory(filter_type)
    if not root.is_dir():
        raise FileNotFoundError(
            f'Ahrens processed directory is missing: {root}'
        )
    names = [path.stem for path in root.glob('*.npz')]
    if not names:
        raise ValueError(f'No Ahrens subject archives found in {root}.')
    subjects = sorted_subject_names(names)
    for name in subjects:
        validate_processed_subject(root / f'{name}.npz', filter_type)
    return subjects


def resolve_subject_selector(
    selector: str | None, available: list[str],
) -> list[str]:
    """Resolve explicit subject names or legacy positional labels.

    Parameters
    ----------
    selector : str or None
        Plus-separated subject names, a legacy zero-based index/range,
        or None for every available subject.
    available : list[str]
        Names returned by ``processed_subject_names``.

    Returns
    -------
    list[str]
        Actual processed archive stems in numeric subject order.
    """
    if not available:
        raise ValueError('Cannot select from an empty Ahrens cohort.')
    if selector is None:
        return list(available)
    if selector.startswith('subject_'):
        numbers = [subject_number(item) for item in selector.split('+')]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f'Duplicate subjects in {selector!r}.')
        names = {subject_number(name): name for name in available}
        missing = sorted(set(numbers) - set(names))
        if missing:
            raise ValueError(
                f'Ahrens subject numbers {missing} are not processed.'
            )
        return [names[number] for number in numbers]
    pieces = selector.split('-')
    if len(pieces) not in (1, 2) or any(
        not piece.isdecimal() for piece in pieces
    ):
        raise ValueError(f'Invalid Ahrens subject selector: {selector!r}.')
    start = int(pieces[0])
    end = start + 1 if len(pieces) == 1 else int(pieces[1])
    if start < 0 or end <= start or end > len(available):
        raise ValueError(
            f'Ahrens positional selection {selector!r} is outside '
            f'0..{len(available) - 1}.'
        )
    return available[start:end]


def require_matching_subjects(filter_types: tuple[str, ...]) -> list[str]:
    """Require the same subject inventory in each filter condition.

    Parameters
    ----------
    filter_types : tuple[str, ...]
        Processing variants to compare.

    Returns
    -------
    list[str]
        Shared subject names in numeric order.
    """
    if not filter_types:
        raise ValueError('At least one Ahrens filter is required.')
    reference = processed_subject_names(filter_types[0])
    for filter_type in filter_types[1:]:
        actual = processed_subject_names(filter_type)
        if actual != reference:
            raise ValueError(
                f'Ahrens {filter_type} subjects {actual} differ from '
                f'{filter_types[0]} subjects {reference}.'
            )
    return reference


def equal_group_counts(subject_count: int) -> list[int]:
    """Return group counts that divide a subject cohort exactly.

    Parameters
    ----------
    subject_count : int
        Number of available named subjects.

    Returns
    -------
    list[int]
        All exact divisors from one group to singleton groups.
    """
    if subject_count < 2:
        raise ValueError(
            f'Session scaling requires at least 2 subjects, got '
            f'{subject_count}.'
        )
    return [count for count in range(1, subject_count + 1)
            if subject_count % count == 0]


def equal_subject_group_labels(
    subjects: list[str], dataset_label: str,
) -> dict[int, list[str]]:
    """Partition a fixed cohort into every exact group count.

    Parameters
    ----------
    subjects : list[str]
        Stable subject names in numeric order.
    dataset_label : str
        Dataset label prefix such as ``zebrafishahrens_pc``.

    Returns
    -------
    dict[int, list[str]]
        Group count to plus-separated explicit subject labels.
    """
    if subjects != sorted_subject_names(subjects):
        raise ValueError('Ahrens subjects must be in numeric order.')
    labels = {}
    for group_count in equal_group_counts(len(subjects)):
        group_size = len(subjects) // group_count
        labels[group_count] = [
            f"{dataset_label}-{'+'.join(subjects[start:start + group_size])}"
            for start in range(0, len(subjects), group_size)
        ]
    return labels
