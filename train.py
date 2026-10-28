"""Train and evaluate one resolved POCO fit.

Input is ``config.json`` in a hashed fit directory. Training writes
``progress.txt`` with train and validation columns, a validation-selected
``net_best.pth``, and ``flops.json``. Final evaluation writes ``test.txt``
with the same objective on the test split and a validated ``complete.json``.
The head-sharing campaign also writes ``exposure.json``, mapping each
subject name to its actual training-window presentations.
"""

import logging
import os.path as osp
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import time

from configs.config_global import LOG_LEVEL, NP_SEED, TCH_SEED, USE_CUDA, DATA_DIR
from configs.configs import (SupervisedLearningBaseConfig,
                             NeuralPredictionConfig)
from datasets.dataloader import DatasetIters
from tasks.taskfunctions import TaskFunction
from models.model_utils import model_init
from utils.config_utils import load_config
from utils.logger import Logger
from utils.artifacts import (mark_complete, validate_complete,
                             locked_fit, fit_lock)
from utils.flops import measure_train_step
import json
from pathlib import Path
from utils.train_utils import grad_clipping, task_init, log_complete
from datetime import datetime

LOG_LOSS_EPSILON = 1e-4

def train_from_path(path: str) -> None:
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)

def eval_from_path(path: str) -> None:
    """Evaluate an incomplete fit from its saved configuration."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    complete_evaluation(config)

def require_positive_weights(weights: list[float]) -> None:
    """Reject empty, non-finite, or nonpositive dataset weights.

    Parameters
    ----------
    weights : list[float]
        One objective weight per dataset.
    """
    if not weights or not np.isfinite(weights).all():
        raise ValueError('Dataset weights must be finite and nonempty.')
    if any(weight <= 0 for weight in weights):
        raise ValueError('Dataset weights must be positive.')


def evaluate_performance(
    net: nn.Module, 
    config: SupervisedLearningBaseConfig, 
    eval_data: DatasetIters,
    task_func: TaskFunction, 
    logger: Logger, 
    validation_losses: list = None,
    i_b: int = 0,
    phase: str = 'val'
) -> None:
    """Measure validation or test loss and save the best validation model.

    Parameters
    ----------
    net : nn.Module
        Model evaluated without gradient updates.
    config : SupervisedLearningBaseConfig
        Resolved fit configuration.
    eval_data : DatasetIters
        Validation or test iterator.
    task_func : TaskFunction
        Prediction task and metrics.
    logger : Logger
        Split-specific metric writer.
    validation_losses : list, optional
        Prior validation losses, default None.
    i_b : int, optional
        Training step, default 0.
    phase : str, optional
        Evaluation phase, default 'val'.
    """

    require_positive_weights(config.mod_w)
    if phase not in ('val', 'test'):
        raise ValueError(f'Expected val or test phase, got {phase}.')
    if phase == 'val' and validation_losses is None:
        raise ValueError('Validation requires a loss history.')
    net.eval()
    start = time.time()
    with torch.no_grad():

        eval_data.reset()
        all_loss = 0
        batch_count = 0

        for i_tloader, eval_iter in enumerate(eval_data.data_iters):
            total = 0
            weighted_loss = 0.0
            while True:
                try:
                    t_data = next(eval_iter)
                    batch_count += 1
                    t_data = get_full_input(
                        t_data, i_tloader, config, eval_data.input_sizes
                    )
                    result = task_func.roll(net, t_data, phase)
                    loss, num = result[: 2]
                    weighted_loss += loss.item() * num
                    total += num
                except StopIteration:
                    break
            if total == 0:
                raise ValueError(
                    f'No predictions for {phase} dataset {i_tloader}.'
                )
            dataset_loss = weighted_loss / total
            if config.log_loss:
                dataset_loss = np.log(dataset_loss + LOG_LOSS_EPSILON)
            all_loss += dataset_loss * config.mod_w[i_tloader]
        eval_loss = all_loss
    
    eval_time = time.time() - start
    logging.info(
        'Avg inference time per batch: %.1f ms',
        eval_time * 1000 / batch_count
    )
    if not np.isfinite(eval_loss):
        raise ValueError(f'{phase} loss is non-finite.')
    logger.log_tabular(f'{phase.capitalize()}Loss', eval_loss)
    if phase == 'val':
        validation_losses.append(eval_loss)

    if phase == 'val':
        best = False
        if eval_loss <= min(validation_losses):
            best = True
            torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))
    else:
        best = True

    task_func.after_testing_callback(
        logger=logger, save_path=config.save_path, is_best=best,
        batch_num=i_b, phase=phase
    )

def get_full_input(
    batch: tuple, dataset_idx: int,
    config: NeuralPredictionConfig, input_sizes: list[list[int]]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    """Insert zero-batch placeholders for inactive sessions.

    Parameters
    ----------
    batch : tuple
        Per-session input and target tensors, each shaped
        ``(time, batch, units)``, plus session metadata.
    dataset_idx : int
        Index of the active dataset.
    config : NeuralPredictionConfig
        Sequence lengths used for empty placeholders.
    input_sizes : list[list[int]]
        Unit counts per session, grouped by dataset.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]
        Input, target, and metadata lists for all sessions.
    """
    # prepare an empty list
    input_size_list = list(itertools.chain(*input_sizes))
    n_sessions = [len(input_size) for input_size in input_sizes]
    dataset_start_idx = [sum(n_sessions[:i]) for i in range(len(n_sessions) + 1)]

    batch_input, batch_target, batch_info = batch

    input_list = []
    target_list = []
    info_list = []

    for i, size in enumerate(input_size_list):
        if i >= dataset_start_idx[dataset_idx] and i < dataset_start_idx[dataset_idx + 1]:
            idx = i - dataset_start_idx[dataset_idx]
            input_list.append(batch_input[idx])
            target_list.append(batch_target[idx])
            info_list.append(batch_info[idx])
        else:
            input_list.append(torch.zeros(config.seq_length - config.pred_length, 0, size))
            target_list.append(torch.zeros(config.seq_length - 1, 0, size))
            info_list.append({'session_idx': i, })

    return input_list, target_list, info_list

def forward_backward(
    config: NeuralPredictionConfig, net: nn.Module,
    optimizer: torch.optim.Optimizer, task_func: TaskFunction,
    train_data: DatasetIters
) -> tuple[torch.Tensor, int, list[int]]:
    """Run the weighted training loss and backward pass on one batch.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit configuration.
    net : nn.Module
        Neural model.
    optimizer : torch.optim.Optimizer
        Optimizer whose gradients are cleared.
    task_func : TaskFunction
        Prediction loss task.
    train_data : DatasetIters
        Active training iterators.

    Returns
    -------
    tuple[torch.Tensor, int, list[int]]
        Loss, total examples, and per-session presentation counts.
    """
    optimizer.zero_grad()
    loss = 0.0
    examples = 0
    session_counts = []
    for i_loader, train_iter in enumerate(train_data.data_iters):
        data = next(train_iter)
        data = get_full_input(
            data, i_loader, config, train_data.input_sizes
        )
        counts = [x.shape[1] for x in data[0]]
        examples += sum(counts)
        session_counts.extend(counts)
        dataset_loss = task_func.roll(net, data, 'train')
        if config.log_loss:
            dataset_loss = torch.log(dataset_loss + LOG_LOSS_EPSILON)
        loss += dataset_loss * config.mod_w[i_loader]
    expected = sum(loader.batch_size for loader in train_data.data_loaders)
    if examples != expected:
        raise ValueError(
            f'Expected {expected} training examples, got {examples}.'
        )
    if not torch.isfinite(loss).all():
        raise ValueError('Weighted training loss is non-finite.')
    loss.backward()
    return loss, examples, session_counts

@locked_fit
def model_train(config: NeuralPredictionConfig) -> None:
    """Fit, validate, test, and finalise one neural prediction model.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved fit configuration with a prepared directory.
    """

    if validate_complete(config):
        logging.info('Skipping completed fit: %s', config.save_path)
        return
    require_positive_weights(config.mod_w)
    contents = {path.name for path in
                Path(config.save_path).iterdir()}
    if contents != {'config.json'}:
        raise RuntimeError(
            f'Incomplete fit directory contains {sorted(contents)}; '
            'prepare it through main.py before retraining.'
        )
    start_time = time.time()

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    # set the torch hub directory
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    train_start_time = datetime.now()

    if USE_CUDA:
        logging.info("training with GPU")

    # initialize logger
    logger = Logger(output_dir=config.save_path)

    if config.max_batch < 1 or config.log_every < 1:
        raise ValueError(
            f'Expected positive max_batch and log_every, got '
            f'{config.max_batch} and {config.log_every}.'
        )
    # initialize dataset
    train_data = DatasetIters(config, 'train')
    if train_data.min_iter_len == 0:
        raise ValueError('Training dataset has no complete batches.')
    assert config.perform_val
    val_data = DatasetIters(config, 'val')
    val_baselines = val_data.get_baselines()

    # initialize task
    task_func: TaskFunction = task_init(config, train_data.input_sizes)
    task_func.mse_baseline['val'] = val_baselines

    # initialize model
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Model: {}, Total parameters: {}'.format(config.model_type, param_count))
    
    # initialize optimizer
    if config.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wdecay)
    elif config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wdecay, amsgrad=True)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr_SGD,
                                    momentum=0.9, weight_decay=config.wdecay)
    else:
        raise NotImplementedError('optimizer not implemented')

    # initialize Learning rate scheduler
    if config.use_lr_scheduler:
        if config.scheduler_type == 'ExponentialLR':
            scheduler = lrs.ExponentialLR(optimizer, gamma=0.1)
        elif config.scheduler_type == 'StepLR':
            scheduler = lrs.StepLR(optimizer, 1.0, gamma=0.1)
        elif config.scheduler_type == 'CosineAnnealing':
            scheduler = lrs.CosineAnnealingLR(
                optimizer, 
                T_max=config.max_batch // config.log_every + 1,
                eta_min=config.lr / 20
            )
        else:
            raise NotImplementedError('scheduler_type must be specified')

    i_b = 0
    validation_losses = []
    break_flag = False
    train_loss = 0.0
    train_examples_seen = 0
    session_examples_seen = [
        0 for group in train_data.input_sizes for _ in group
    ]

    for epoch in range(config.num_ep):
        train_data.reset()

        for step_ in range(train_data.min_iter_len):
            
            net.train()
            # save model
            if (i_b + 1) % config.save_every == 0:
                torch.save(net.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(i_b + 1)))

            if i_b == 0:
                flops, (loss, examples, counts) = measure_train_step(
                    lambda: forward_backward(
                        config, net, optimizer, task_func, train_data
                    )
                )
                flops['training_examples_in_measured_step'] = examples
                flops['trainable_parameters'] = param_count
                with open(osp.join(config.save_path, 'flops.json'), 'x') as f:
                    json.dump(flops, f, indent=2)
            else:
                loss, examples, counts = forward_backward(
                    config, net, optimizer, task_func, train_data
                )
            # gradient clipping
            if config.grad_clip is not None:
                grad_clipping(net, config.grad_clip)

            optimizer.step()
            train_loss += loss.item()
            train_examples_seen += examples
            if len(counts) != len(session_examples_seen):
                raise ValueError(
                    f'Expected {len(session_examples_seen)} session '
                    f'counts, got {len(counts)}.'
                )
            session_examples_seen = [
                previous + current for previous, current
                in zip(session_examples_seen, counts)
            ]

            # log performance
            log_count = i_b % config.log_every + 1
            if (log_count == config.log_every or
                    i_b + 1 == config.max_batch):

                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('BatchNum', i_b + 1)
                logger.log_tabular(
                    'TrainExamplesSeen', train_examples_seen
                )
                logger.log_tabular('TrainLoss', train_loss / log_count)

                evaluate_performance(
                    net, config, val_data, task_func, logger,
                    validation_losses, i_b + 1
                )
                logger.dump_tabular()
                train_loss = 0

                if config.use_lr_scheduler:
                    scheduler.step()

            i_b += 1
            if i_b >= config.max_batch:
                break_flag = True
                break

        if break_flag:
            break

    if i_b != config.max_batch:
        raise RuntimeError(
            f'Expected {config.max_batch} training steps, got {i_b}.'
        )
    duration_min = (time.time() - start_time) / 60
    logging.info('Total training time: %.2f mins', duration_min)

    if config.experiment_name == 'ahrens_head_sharing_factorial':
        label = config.dataset_label[0]
        subjects = config.dataset_config[label].session_ids
        if (len(subjects) != len(session_examples_seen) or
                sum(session_examples_seen) != train_examples_seen):
            raise ValueError(
                'Per-subject training exposure does not match '
                'the recorded training examples.'
            )
        exposure = dict(zip(subjects, session_examples_seen))
        with (Path(config.save_path) / 'exposure.json').open(
            'x', encoding='utf-8'
        ) as stream:
            json.dump(exposure, stream, indent=2)
    task_func.after_training_callback(config, net)
    log_complete(config.save_path, train_start_time)

    if config.perform_test:
        model_eval(config)
        mark_complete(config)

def complete_evaluation(config: NeuralPredictionConfig) -> None:
    """Evaluate and finalise a trained fit under its exclusive lock.

    Parameters
    ----------
    config : SupervisedLearningBaseConfig
        Fit configuration with completed training artifacts.
    """
    with fit_lock(config):
        if validate_complete(config):
            return
        model_eval(config)
        mark_complete(config)


def model_eval(config: NeuralPredictionConfig) -> None:
    """Evaluate the validation-selected checkpoint on the test split.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fit configuration with a saved best checkpoint.
    """
    if validate_complete(config):
        logging.info('Skipping completed fit: %s', config.save_path)
        return
    np.random.seed(NP_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(config.seed)

    test_data = DatasetIters(config, 'test')
    task_func: TaskFunction = task_init(config, test_data.input_sizes)
    task_func.mse_baseline['test'] = test_data.get_baselines()
    net = model_init(config, test_data.input_sizes, test_data.unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    test_path = osp.join(config.save_path, 'test.txt')
    if osp.exists(test_path):
        raise FileExistsError(
            f'Test results already exist for incomplete fit: {test_path}'
        )
    logger = Logger(
        output_dir=config.save_path, output_fname='test.txt'
    )
    evaluate_performance(
        net, config, test_data, task_func, logger, phase='test'
    )
    logger.dump_tabular()