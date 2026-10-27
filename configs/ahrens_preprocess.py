"""Read local Ahrens preprocessing settings for one campaign.

Input is the ignored ``configs/local/ahrens_preprocess.json`` object.
It contains ``filters`` (nonempty list of filter names),
``pca_solver`` (full or randomized), ``pc_dim`` (positive integer),
``pca_seed`` (nonnegative integer), and ``storage_dtype`` (float32).
Output is an immutable settings object passed to each recording worker.
"""

from dataclasses import dataclass

from configs.local_config import LOCAL_CONFIG_DIR, load_local_json

FILTER_CHOICES = ('none', 'lowpass', 'highpass', 'bandpass')
PCA_SOLVERS = ('full', 'randomized')
STORAGE_DTYPE = 'float32'
AHRENS_CONFIG_NAME = 'ahrens_preprocess.json'
REQUIRED_KEYS = {
    'filters', 'pca_solver', 'pc_dim', 'pca_seed', 'storage_dtype',
}


@dataclass(frozen=True)
class AhrensPreprocessConfig:
    """Scientific and storage settings shared by all Ahrens workers.

    Attributes
    ----------
    filters : tuple[str, ...]
        Distinct filter conditions processed in parallel.
    pca_solver : str
        Full or randomized PCA.
    pc_dim : int
        Requested number of principal components.
    pca_seed : int
        Randomized PCA seed; recorded for either solver.
    storage_dtype : str
        Neural array storage dtype, currently float32.
    """

    filters: tuple[str, ...]
    pca_solver: str
    pc_dim: int
    pca_seed: int
    storage_dtype: str


def load_ahrens_preprocess_config() -> AhrensPreprocessConfig:
    """Validate Ahrens settings from the ignored local JSON file.

    Returns
    -------
    AhrensPreprocessConfig
        Validated settings for the complete preprocessing campaign.

    Raises
    ------
    ValueError
        If settings are missing, malformed, or unsupported.
    """
    settings = load_local_json(AHRENS_CONFIG_NAME)
    path = LOCAL_CONFIG_DIR / AHRENS_CONFIG_NAME
    actual_keys = set(settings)
    if actual_keys != REQUIRED_KEYS:
        raise ValueError(
            f'Expected keys {sorted(REQUIRED_KEYS)}, '
            f'got {sorted(actual_keys)} in {path}.'
        )
    filters = settings['filters']
    if (not isinstance(filters, list) or not filters or
            any(
                not isinstance(item, str) or item not in FILTER_CHOICES
                for item in filters
            ) or
            len(filters) != len(set(filters))):
        raise ValueError(
            f'Expected distinct nonempty filters from '
            f'{FILTER_CHOICES}, got {filters!r}.'
        )
    solver = settings['pca_solver']
    if solver not in PCA_SOLVERS:
        raise ValueError(
            f'Expected pca_solver in {PCA_SOLVERS}, got {solver!r}.'
        )
    pc_dim = settings['pc_dim']
    if type(pc_dim) is not int or pc_dim < 1:
        raise ValueError(
            f'Expected positive integer pc_dim, got {pc_dim!r}.'
        )
    seed = settings['pca_seed']
    if type(seed) is not int or seed < 0:
        raise ValueError(
            f'Expected nonnegative integer pca_seed, got {seed!r}.'
        )
    dtype = settings['storage_dtype']
    if dtype != STORAGE_DTYPE:
        raise ValueError(
            f'Expected storage_dtype={STORAGE_DTYPE!r}, got {dtype!r}.'
        )
    return AhrensPreprocessConfig(
        filters=tuple(filters), pca_solver=solver, pc_dim=pc_dim,
        pca_seed=seed, storage_dtype=dtype,
    )
