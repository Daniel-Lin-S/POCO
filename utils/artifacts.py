"""Identify, validate, and finalise immutable model fits.

Input is a resolved training configuration. Each fit is stored at
``artifact_root/fits/<scientific hash>/``. The directory contains
``config.json``, ``progress.txt`` (train and validation metrics),
``test.txt`` (test metrics), ``net_best.pth``, and the three
``*_best_info.npy`` dictionaries (mse, mae, pred_num, sample_trials).
``flops.json`` records forward and backward FLOPs for step 1, operator
coverage, batch example count, and trainable parameter count.
``complete.json`` stores fit_id, completed steps, and SHA-256 hashes
of required files. Incomplete fits move to ``quarantine/`` on retry.
"""

import hashlib
import fcntl
from contextlib import contextmanager
from functools import wraps
from collections.abc import Callable, Iterator
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config_global import ARTIFACT_ROOT
from configs.configs import NeuralPredictionConfig

FIT_VERSION = 1


class FitBusyError(RuntimeError):
    """Raised when another process owns a fit lock."""


NONSCIENTIFIC_FIELDS = {
    'cpu', 'hours', 'mem', 'num_workers', 'overwrite', 'save_path',
    'model_name', 'experiment_name', 'save_every', 'do_analysis',
    'tokenizer_dir', 'tokenizer_state_dict_file',
}
DATASET_NONSCIENTIFIC_FIELDS = {'num_workers', 'show_chance'}
MODEL_ARCHITECTURE_FIELDS = {
    'rnn_type', 'num_layers', 'teacher_forcing', 'rnn_layernorm',
    'hidden_size', 'rnn_alpha', 'rnn_rank', 'learnable_alpha',
    'rnn_residual_connection', 'stem_ratio', 'downsample_ratio',
    'ffn_ratio', 'patch_size', 'patch_stride', 'num_blocks',
    'large_size', 'small_size', 'dims', 'dw_dims',
    'small_kernel_merged', 'call_structural_reparam',
    'use_multi_scale', 'dropout', 'kernel_size', 'individual',
    'freq', 'revin', 'affine', 'subtract_last', 'head_dropout',
    'decomposition', 'tf_interval', 'latent_to_ob',
    'tokenizer_type', 'circular_conv', 'population_token',
    'population_token_dim', 'decoder_type', 'decoder_hidden_size',
    'separate_projs', 'decoder_proj_init', 'normalize_input',
    'mu_std_loss_coef', 'mu_module_mode', 'std_module_mode',
    'mu_std_separate_projs', 'compression_factor', 'conditioning',
    'conditioning_dim', 'conditioning_layers',
    'freeze_conditioned_net', 'decoder_context_length',
    'poyo_num_latents', 'latent_session_embedding',
    'unit_embedding_components', 'poyo_query_mode',
    'poyo_output_mode', 'decoder_num_layers', 'decoder_num_heads',
    'poyo_unit_dropout', 'rotary_attention_tmax',
    'poyo_embedding_only', 'mlp_hidden_size', 'mlp_hidden_layers',
    'tsmixer_ff_dim', 'filter_embed_size', 'netformer_embed_size',
}
MODEL_ACTIVE_FIELDS = {
    'NLinear': set(),
    'TexFilter': {'filter_embed_size', 'hidden_size', 'dropout'},
}
REQUIRED_FILES = (
    'config.json', 'progress.txt', 'test.txt', 'net_best.pth',
    'flops.json', 'train_complete.txt', 'train_best_info.npy',
    'val_best_info.npy', 'test_best_info.npy',
)


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Parameters
    ----------
    path : Path
        Existing file.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_config(config: NeuralPredictionConfig) -> dict:
    """Return fit parameters without runtime or reporting settings.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fully resolved fit configuration.

    Returns
    -------
    dict
        JSON-compatible scientific parameters.
    """
    result = config.to_dict().copy()
    for field in NONSCIENTIFIC_FIELDS:
        result.pop(field, None)
    active_fields = MODEL_ACTIVE_FIELDS.get(result['model_type'])
    if active_fields is not None:
        for field in MODEL_ARCHITECTURE_FIELDS - active_fields:
            result.pop(field, None)
    for dataset in result['dataset_config'].values():
        for field in DATASET_NONSCIENTIFIC_FIELDS:
            dataset.pop(field, None)
    if result.get('load_path'):
        source = Path(result['load_path'])
        if source.parent.parent.name == 'fits':
            result['load_path'] = source.parent.name + '/' + source.name
        elif source.is_file():
            result['load_path'] = file_digest(source)
        else:
            raise FileNotFoundError(
                f'Pretrained checkpoint does not exist: {source}'
            )
    return result


def fit_id(config: NeuralPredictionConfig) -> str:
    """Hash the canonical scientific configuration of one fit.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fully resolved fit configuration.

    Returns
    -------
    str
        SHA-256 fit identifier.
    """
    payload = json.dumps(scientific_config(config), sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def assign_fit_path(config: NeuralPredictionConfig) -> None:
    """Set a shared artifact path for a resolved fit configuration.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Configuration whose ``save_path`` is set in place.
    """
    config.save_path = str(Path(ARTIFACT_ROOT) / 'fits' / fit_id(config))


def validate_complete(config: NeuralPredictionConfig) -> bool:
    """Check the final marker, hashes, and exact training step count.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit configuration with an assigned save path.

    Returns
    -------
    bool
        True only for a complete, matching fit.
    """
    fit_dir = Path(config.save_path)
    marker = fit_dir / 'complete.json'
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
        if (record['version'] != FIT_VERSION or
                record['fit_id'] != fit_id(config)):
            return False
        if record['steps'] != config.max_batch:
            return False
        if set(record['files']) != set(REQUIRED_FILES):
            return False
        for name, expected in record['files'].items():
            path = fit_dir / name
            if not path.is_file() or file_digest(path) != expected:
                return False
        from utils.config_utils import load_config
        if fit_id(load_config(str(fit_dir))) != record['fit_id']:
            return False
        progress = pd.read_table(fit_dir / 'progress.txt')
        test = pd.read_table(fit_dir / 'test.txt')
        required_progress = {
            'BatchNum', 'TrainExamplesSeen', 'TrainLoss', 'ValLoss'
        }
        if progress.empty or len(test) != 1:
            return False
        if not required_progress.issubset(progress.columns):
            return False
        required_test = {'TestLoss'}
        for label in config.dataset_label:
            required_test.update({
                f'{label}_test_score', f'{label}_test_mse',
                f'{label}_test_mae', f'{label}_test_pred_num'
            })
        if not required_test.issubset(test.columns):
            return False
        if int(progress['BatchNum'].iloc[-1]) != config.max_batch:
            return False
        steps = progress['BatchNum'].to_numpy()
        examples = progress['TrainExamplesSeen'].to_numpy()
        if (steps[0] < 1 or np.any(np.diff(steps) <= 0) or
                examples[0] < 1 or np.any(np.diff(examples) <= 0)):
            return False
        values = list(
            progress[sorted(required_progress)].to_numpy().flat
        )
        values += list(test[sorted(required_test)].to_numpy().flat)
        if not np.isfinite(values).all():
            return False
        flops = json.loads((fit_dir / 'flops.json').read_text())
        count = flops['forward_backward_flops_per_step']
        if not np.isfinite(count) or count <= 0:
            return False
        if flops['training_examples_in_measured_step'] < 1:
            return False
        if (fit_dir / 'net_best.pth').stat().st_size < 1:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def lock_path(config: NeuralPredictionConfig) -> Path:
    """Return the shared lock file path for a scientific fit.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit configuration.

    Returns
    -------
    Path
        Absolute lock file path.
    """
    lock_dir = Path(ARTIFACT_ROOT) / 'locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / fit_id(config)


def is_fit_active(config: NeuralPredictionConfig) -> bool:
    """Report whether a process currently owns this fit.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit configuration.

    Returns
    -------
    bool
        True if another process holds its fit lock.
    """
    with lock_path(config).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


@contextmanager
def fit_lock(config: NeuralPredictionConfig) -> Iterator[None]:
    """Hold the exclusive lock for one fit until it finishes.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit configuration.

    Yields
    ------
    None
        Exclusive fit ownership.
    """
    with lock_path(config).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FitBusyError(
                f'Fit is already active: {config.save_path}'
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def locked_fit(function: Callable) -> Callable:
    """Protect a fit entry point from concurrent invocation.

    Parameters
    ----------
    function : Callable
        Function whose first argument is a fit configuration.

    Returns
    -------
    Callable
        Locked fit function.
    """
    @wraps(function)
    def run(config):
        with fit_lock(config):
            return function(config)
    return run


def prepare_fit(config: NeuralPredictionConfig) -> bool:
    """Skip a valid fit or quarantine invalid artifacts before a new fit.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit configuration.

    Returns
    -------
    bool
        True when a new fit should run.
    """
    fit_dir = Path(config.save_path)
    if validate_complete(config):
        return False
    try:
        with fit_lock(config):
            if validate_complete(config):
                return False
            if fit_dir.exists():
                stamp = datetime.now(timezone.utc).strftime(
                    '%Y%m%dT%H%M%S%fZ'
                )
                quarantine = (
                    Path(ARTIFACT_ROOT) / 'quarantine' / fit_dir.name
                )
                quarantine.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fit_dir), str(quarantine / stamp))
            fit_dir.mkdir(parents=True)
    except FitBusyError:
        return False
    return True


def mark_complete(config: NeuralPredictionConfig) -> None:
    """Atomically create a checksum marker after training and testing.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Completed fit configuration.
    """
    fit_dir = Path(config.save_path)
    marker = fit_dir / 'complete.json'
    if marker.exists():
        if validate_complete(config):
            return
        raise RuntimeError(
            f'Invalid completion marker already exists: {marker}'
        )
    files = {name: file_digest(fit_dir / name) for name in REQUIRED_FILES}
    record = {'version': FIT_VERSION, 'fit_id': fit_id(config),
              'steps': config.max_batch, 'files': files}
    pending = fit_dir / 'complete.json.pending'
    pending.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(pending, fit_dir / 'complete.json')
    if not validate_complete(config):
        raise RuntimeError(f'Fit completion validation failed: {fit_dir}')
