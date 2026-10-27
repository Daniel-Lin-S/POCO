"""Compute phase-specific neural prediction losses and metrics.

Input is per-session input and target tensors with time, batch, and unit
axes. Output is a scalar training loss or an evaluation loss with a
prediction count. Best-split ``*_best_info.npy`` files hold mse, mae,
pred_num, and sample_trials arrays for each selected session.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import analysis.plots as plots
import itertools

from configs.config_global import DEVICE
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig
from utils.logger import Logger

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device) for data in data_b]
    else:
        raise NotImplementedError("input type not recognized")

class TaskFunction:

    def __init__(self, config: SupervisedLearningBaseConfig):
        """
        initailize information from config
        """
        pass

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        """
        Phase can be 'train', 'val', or 'test'.
        In the training mode (train=True), this method should return a scalar representing the training loss.
        In the eval or test mode, this method should return a pair of scalars: the loss and the number of predictions
        """
        raise NotImplementedError

    def after_testing_callback(self, logger: Logger, save_path: str,
                               is_best: bool, batch_num: int,
                               phase: str = 'val'):
        """Save metrics after validation or final test evaluation.

        Parameters
        ----------
        logger : Logger
            Split-specific metric writer.
        save_path : str
            Absolute fit directory.
        is_best : bool
            Whether to save current split diagnostics.
        batch_num : int
            Number of completed training steps.
        phase : str, optional
            Evaluation split, default 'val'.
        """
        pass

    def after_training_callback(self, config, model):
        """
        An optional function called after training is done
        
        :param model: the best model
        """
        pass

class NeuralPrediction(TaskFunction):
    """
    Task function for autoregressive neural prediction
    """

    def __init__(self, config: NeuralPredictionConfig, input_size):
        self.criterion = nn.MSELoss(reduction='sum')
        self.do_analysis = config.do_analysis
        self.loss_mode = config.loss_mode
        self.pred_step = config.pred_length
        self.seq_len = config.seq_length
        self.config = config
        self.mse_baseline = {}

        self.session_ids = {} # session id for each dataset
        n_sessions = 0
        for dataset, size in zip(config.dataset_label, input_size):
            self.session_ids[dataset] = list(range(n_sessions, n_sessions + len(size)))
            n_sessions += len(size)
        print(f'session_ids: {self.session_ids}')
        self.input_size = list(itertools.chain(*input_size))
        self._init_info()

        assert self.loss_mode in ['prediction', 'autoregressive', 'reconstruction', 'reconstruction_prediction']
        if self.loss_mode == 'reconstruction':
            assert self.pred_step == 0, 'reconstruction mode does not support prediction'
        else:
            assert self.pred_step > 0, 'prediction step should be larger than 0'

    def _init_info(self):
        self.sample_trials = {'train': [], 'val': [], 'test': []}
        self.pred_num = {'train': [], 'val': [], 'test': []}
        self.sum_mse = {'train': [], 'val': [], 'test': []}
        self.sum_mae = {'train': [], 'val': [], 'test': []}
        
        for phase in ['train', 'val', 'test']:
            for size in self.input_size:
                self.pred_num[phase].append(np.zeros(size))
                L = self.pred_step if self.pred_step > 0 else self.seq_len
                self.sum_mse[phase].append(np.zeros((L, size)))
                self.sum_mae[phase].append(np.zeros((L, size))) # L, D

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        train_flag = phase == 'train'
        
        input, target, _ = data_batch
        input, target = data_batch_to_device((input, target))
        output = model(input)
    
        task_loss = torch.zeros(1).to(DEVICE)
        pred_num = 0

        for idx, (inp, out, tar) in enumerate(
            zip(input, output, target)
        ):
            if tar is None or np.prod(tar.shape) == 0:
                continue

            if self.loss_mode[: 14] == 'reconstruction':
                tar = torch.cat([inp[: 1], tar], dim=0)
            elif self.loss_mode == 'prediction':
                out = out[-self.pred_step: ]
                tar = tar[-self.pred_step: ]
            else:
                assert self.loss_mode == 'autoregressive'

            if len(self.sample_trials[phase]) < len(input):
                self.sample_trials[phase].append((
                    out.detach().cpu().numpy(), 
                    tar.detach().cpu().numpy(), 
                    inp.detach().cpu().numpy()
                ))
            
            with torch.no_grad():
                if self.pred_step > 0:
                    loss_array = torch.abs(out - tar)[-self.pred_step: ] # shape: L, B, D
                    assert loss_array.shape[0] == self.pred_step
                else:
                    loss_array = torch.abs(out - tar)

                self.sum_mse[phase][idx] += loss_array.pow(2).sum(dim=1).cpu().numpy()
                self.sum_mae[phase][idx] += loss_array.sum(dim=1).cpu().numpy()
                self.pred_num[phase][idx] += loss_array.shape[1]

            task_loss += self.criterion(out, tar) 
            pred_num += np.prod(tar.shape)

        if pred_num == 0:
            raise ValueError(f'No prediction targets in {phase} batch.')
        task_loss = task_loss / pred_num
        if not torch.isfinite(task_loss).all():
            raise ValueError(f'Non-finite prediction loss in {phase} batch.')

        if train_flag:
            return task_loss
        else:
            return task_loss, pred_num
        
    def plot_prediction(self, save_path: str, phase: str):
        # visualize prediction for selected dimensions
        for pc in [0, 20, 50, -1]:
            if pc >= max(self.input_size):
                continue

            preds = []
            targets = []

            for iter in range(10):
                batch_idx = np.random.randint(len(self.sample_trials[phase]))
                data = self.sample_trials[phase][batch_idx]
                sample_idx = np.random.randint(data[0].shape[1])

                pred = data[0][:, sample_idx, pc]
                target = data[1][:, sample_idx, pc]
                inp = data[2][:, sample_idx, pc]

                if self.loss_mode == 'prediction': # get the whole trial
                    pred = np.concatenate([inp, pred])
                    target = np.concatenate([inp, target])
                preds.append(pred)
                targets.append(target)

            plots.pred_vs_target_plot(
                list(zip(preds, targets)), 
                os.path.join(save_path, f'{phase}_pred_vs_target'), 
                f'best_dim{pc}',
                len(preds[0]) - self.pred_step
            )
        
    def after_testing_callback(self, logger: Logger, save_path: str,
                               is_best: bool, batch_num: int,
                               phase: str = 'val'):

        # compute mean loss for the prediction part
        phases = ['train', 'val'] if phase == 'val' else ['test']
        horizon = self.pred_step if self.pred_step > 0 else self.seq_len
        for split in phases:
            if split in ('val', 'test') and split not in self.mse_baseline:
                raise ValueError(
                    f'Missing {split} baseline for prediction score.'
                )

            for dataset, session_ids in self.session_ids.items():
                sum_pred_num = 0
                sum_mse = 0
                sum_mae = 0
                avg_mse_score = 0

                for idx in session_ids:
                    count = self.pred_num[split][idx].sum()
                    sum_pred_num += count * horizon
                    sum_mse += self.sum_mse[split][idx].sum()
                    sum_mae += self.sum_mae[split][idx].sum()

                    if split in ('val', 'test'):
                        baseline = self.mse_baseline[split][idx]
                        if count <= 0 or baseline <= 0:
                            raise ValueError(
                                f'{split} session {idx} has no predictions '
                                'or has a nonpositive baseline.'
                            )
                        mse = self.sum_mse[split][idx].sum()
                        mse /= count * horizon
                        avg_mse_score += 1 - mse / baseline

                if sum_pred_num <= 0:
                    raise ValueError(
                        f'No {split} predictions for {dataset}.'
                    )
                avg_mse_score /= len(session_ids)
                if not np.isfinite((avg_mse_score, sum_mse,
                                    sum_mae)).all():
                    raise ValueError(
                        f'Non-finite {split} metrics for {dataset}.'
                    )

                if split in ('val', 'test'):
                    logger.log_tabular(
                        f'{dataset}_{split}_pred_num', int(sum_pred_num)
                    )
                    logger.log_tabular(
                        f'{dataset}_{split}_score', avg_mse_score
                    )
                
                logger.log_tabular(
                    f'{dataset}_{split}_mse', sum_mse / sum_pred_num
                )
                logger.log_tabular(
                    f'{dataset}_{split}_mae', sum_mae / sum_pred_num
                )
            
            if is_best and len(self.sample_trials[split]) > 0:
                self.plot_prediction(save_path, split)

        if is_best:
            for split in phases:
                info = {
                    'mse': self.sum_mse[split],
                    'mae': self.sum_mae[split],
                    'pred_num': self.pred_num[split],
                    'sample_trials': self.sample_trials[split]
                }
                np.save(os.path.join(save_path, f'{split}_best_info.npy'), info)

        self._init_info()
    
    def after_training_callback(self, config, model):
        if not self.do_analysis:
            return