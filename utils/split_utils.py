"""Select time points used to fit recording preprocessing.

Input is a recording length and the patchwise train/validation fractions.
Output is a Boolean mask over the recording's time axis. The mask includes
train and validation samples and excludes test samples in every patch.
"""

import numpy as np


def train_validation_mask(
    length: int,
    patch_length: int,
    train_split: float,
    val_split: float,
) -> np.ndarray:
    """Return train and validation positions for one recording.

    Parameters
    ----------
    length : int
        Number of time samples in the recording.
    patch_length : int
        Nominal number of samples per patch.
    train_split : float
        Fraction assigned to training within each patch.
    val_split : float
        Fraction assigned to validation within each patch.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(length,)``.

    Raises
    ------
    ValueError
        If the split is invalid or contains no fit samples.
    """
    if length <= 0 or patch_length <= 0:
        raise ValueError(
            f"Expected positive length and patch_length, got "
            f"{length} and {patch_length}."
        )
    if not 0 < train_split < train_split + val_split < 1:
        raise ValueError(
            "Expected 0 < train_split < train_split + val_split < 1, "
            f"got {train_split} and {val_split}."
        )

    mask = np.zeros(length, dtype=bool)
    segments = max(1, length // patch_length)
    for patch in range(segments):
        start = patch * patch_length
        end = (patch + 1) * patch_length
        if patch == segments - 1:
            end = length
        fit_end = start + int((end - start) * (train_split + val_split))
        mask[start:fit_end] = True

    if not mask.any():
        raise ValueError("No train or validation samples available for fit.")
    return mask
