"""Load ignored machine-specific JSON settings from ``configs/local``.

Input is a named JSON object file under ``configs/local``. Output is its
validated top-level mapping. Scientific settings and machine paths live
in separate JSON files; tracked templates live under ``configs/templates``.
"""

import json
from pathlib import Path

LOCAL_CONFIG_DIR = Path(__file__).resolve().parent / 'local'


def load_local_json(filename: str) -> dict[str, object]:
    """Read one local JSON object by basename.

    Parameters
    ----------
    filename : str
        JSON filename within ``configs/local``.

    Returns
    -------
    dict[str, object]
        Parsed top-level mapping.

    Raises
    ------
    FileNotFoundError
        If the requested local configuration is absent.
    ValueError
        If the filename or JSON content is invalid.
    """
    if Path(filename).name != filename or not filename.endswith('.json'):
        raise ValueError(
            f'Expected a JSON basename, got {filename!r}.'
        )
    path = LOCAL_CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'Create local configuration file {path}.'
        )
    with path.open(encoding='utf-8') as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError(
            f'Expected a JSON object in {path}, '
            f'got {type(content).__name__}.'
        )
    return content
