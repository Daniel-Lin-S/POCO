"""Define project paths and shared plot settings.

Input is ignored ``configs/local/paths.json`` with an absolute
``artifact_root`` value. Fit artifacts and figures are written below that
root; Ahrens raw and processed roots can also be set locally.
"""

import os
import os.path as osp
import logging
from configs.local_config import load_local_json
import torch

NP_SEED = 233
TCH_SEED = 3407

# Directory paths
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
local_paths = load_local_json('paths.json')


def local_absolute_path(key: str, fallback: str | None = None) -> str:
    """Resolve and validate one locally configured absolute directory.

    Parameters
    ----------
    key : str
        Key in the ignored local configuration.
    fallback : str or None, optional
        Default absolute directory, default None.

    Returns
    -------
    str
        Absolute configured directory.
    """
    value = local_paths.get(key, fallback)
    if not isinstance(value, str) or not osp.isabs(value):
        raise ValueError(f'{key} must be an absolute directory path.')
    return value


ARTIFACT_ROOT = local_absolute_path('artifact_root')
FIG_DIR = osp.join(ARTIFACT_ROOT, 'figures')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

ZEBRAFISH_RAW_DIR = os.path.join(DATA_DIR, 'raw_zebrafish')
ZEBRAFISH_AHRENS_RAW_DIR = local_absolute_path(
    'ahrens_raw_dir', os.path.join(DATA_DIR, 'raw_zebrafish_ahrens')
)
EXP_TYPES = ['control', 'shocked', 'reshocked', 'ketamine']
CELEGANS_RAW_DIR = os.path.join(DATA_DIR, 'raw_celegans_zimmer')
CELEGANS_FLAVELL_RAW_DIR = os.path.join(DATA_DIR, 'raw_celegans_flavell')
MICE_RAW_DIR = os.path.join(DATA_DIR, 'raw_mice')
MICE_BRAIN_AREAS = ['PPC', 'RSP', 'V1', 'M2']
ZEBRAFISH_BRAIN_AREAS = ['LHb', 'MHb', 'gc', 'raphe', 'ctel', 'vent', 'dthal', 'vthal', ]

N_ZEBRAFISH_SESSIONS = 19
N_CELEGANS_SESSIONS = 5
N_CELEGANS_FLAVELL_SESSIONS = 40
N_MICE_SESSIONS = 12

RAW_DATA_SUFFIX = '_CNMF_compressed.h5'
ZEBRAFISH_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_zebrafish')
ZEBRAFISH_AHRENS_PROCESSED_DIR = local_absolute_path(
    'ahrens_processed_dir',
    os.path.join(DATA_DIR, 'processed_zebrafish_ahrens')
)
CELEGANS_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_celegans_zimmer')
CELEGANS_FLAVELL_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_celegans_flavell')
MICE_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_mice')
SIM_DIR = osp.join(DATA_DIR, 'simulations')

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MAP_LOC = "cuda:0" if USE_CUDA else torch.device('cpu')
LOG_LEVEL = logging.INFO
BASE_MODEL_COLORS = {
    'NLinear': '#2C4875', 'DLinear': '#5A7684', 'POYO': '#edae49', 'POCO': '#E67E22', 'TACO': '#E99675',# '#FF7F0E', #
    'Latent_PLRNN': '#9B3D3D', 'AR_Transformer': '#4E5D6C', 'MultiAR_Transformer': '#4E5D6C',
    'NetFormer': '#A0C4FF', 'TexFilter': '#68A4D3',
    'TCN': '#E99675', 'MLP': '#005F73', 'TSMixer': '#C9A66B', 'MLP_L': '#005F73'
}
MODEL_COLORS = {}

for model in BASE_MODEL_COLORS:
    MODEL_COLORS['All_' + model] = BASE_MODEL_COLORS[model] + '60'
    MODEL_COLORS['MS_' + model] = BASE_MODEL_COLORS[model]
    MODEL_COLORS[model] = BASE_MODEL_COLORS[model] + '85'