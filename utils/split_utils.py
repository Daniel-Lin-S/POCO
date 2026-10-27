"""Define one chronological train, validation, and test split.

Input is a recording length and 60/20/20 fractions. Output is the two
exclusive boundaries and a mask covering train plus validation samples.
"""

import numpy as np


def chronological_bounds(
    length: int,
    train_split: float,
    val_split: float,
    test_split: float,
) -> tuple[int, int]:
    """Return chronological split boundaries for one recording.

    Parameters
    ----------
    length : int
        Number of time samples in the recording.
    train_split : float
        Training fraction.
    val_split : float
        Validation fraction.
    test_split : float
        Test fraction.

    Returns
    -------
    tuple[int, int]
        Exclusive train and validation endpoints.
    """
    if length < 3:
        raise ValueError(f'Expected at least 3 samples, got {length}.')
    fractions = (train_split, val_split, test_split)
    if any(value <= 0 for value in fractions):
        raise ValueError(f'Split fractions must be positive: {fractions}.')
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f'Split fractions must sum to 1: {fractions}.')
    train_end = int(length * train_split)
    val_end = int(length * (train_split + val_split))
    if not 0 < train_end < val_end < length:
        raise ValueError(
            f'Expected three nonempty splits for length {length}, '
            f'got boundaries {train_end} and {val_end}.'
        )
    return train_end, val_end


def train_validation_mask(
    length: int,
    train_split: float,
    val_split: float,
    test_split: float,
) -> np.ndarray:
    """Select training and validation samples, excluding test samples.

    Parameters
    ----------
    length : int
        Number of time samples in the recording.
    train_split : float
        Training fraction.
    val_split : float
        Validation fraction.
    test_split : float
        Test fraction.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(length,)``.
    """
    _, val_end = chronological_bounds(
        length, train_split, val_split, test_split
    )
    mask = np.zeros(length, dtype=bool)
    mask[:val_end] = True
    return mask
