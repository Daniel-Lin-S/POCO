"""Coordinate ownership of hash-named fits across campaign launchers.

Input is a resolved fit configuration. Output is an advisory claim at
``artifact_root/claims/<fit_id>`` held from preparation through worker
exit. Claim files contain no model data and persist for future launches.
"""

import fcntl
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from pathlib import Path

from configs.config_global import ARTIFACT_ROOT
from configs.configs import NeuralPredictionConfig
from utils.artifacts import fit_id

CLAIM_ROOT = Path(ARTIFACT_ROOT) / 'claims'


def claim_path(config: NeuralPredictionConfig) -> Path:
    """Return the shared advisory claim path for one scientific fit."""
    CLAIM_ROOT.mkdir(parents=True, exist_ok=True)
    return CLAIM_ROOT / fit_id(config)


def is_claimed(config: NeuralPredictionConfig) -> bool:
    """Report whether another process holds this fit's claim."""
    with claim_path(config).open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


@contextmanager
def fit_claim(
    config: NeuralPredictionConfig,
    on_wait: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Own a fit across preparation, training, and final validation.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit whose scientific ID determines the claim path.
    on_wait : callable or None, optional
        Called once before waiting for an existing owner, default None.

    Yields
    ------
    None
        Exclusive claim until the enclosing worker exits the context.
    """
    with claim_path(config).open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait is not None:
                on_wait()
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
