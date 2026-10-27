"""Launch POCO experiment fits and analyses.

Input is an experiment factory name from ``configs.experiments`` and a local
``configs/local/paths.json`` containing ``artifact_root``. Output is
one hashed fit directory per scientific configuration, figures under
the artifact root,
and launch logs under the repository's ``logs`` directory.
"""

import os
import subprocess
import argparse
import logging
import time
import shlex
import sys
import json
from datetime import datetime, timezone

import configs.experiments as experiments
import torch

from utils.config_utils import save_config, configs_dict_unpack
from train import model_train, complete_evaluation
from configs.config_global import LOG_LEVEL, ROOT_DIR
from utils.artifacts import validate_complete, is_fit_active
from configs.configs import BaseConfig

def train_cmd(config_: BaseConfig) -> str:
    """Return a command to train one fit with the active interpreter."""
    code = f'import train; train.train_from_path({config_.save_path!r})'
    return f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'


def eval_cmd(eval_save_path: str) -> str:
    """Return a command to evaluate one incomplete fit."""
    code = f'import train; train.eval_from_path({eval_save_path!r})'
    return f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'


def analysis_cmd(exp_name: str) -> str:
    """Return the analysis command for a named experiment."""
    code = (
        'import configs.exp_analysis as analysis; '
        f'analysis.{exp_name}_analysis()'
    )
    return f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'


def get_jobfile(
    cmd: str, job_name: str, dep_ids: list[str],
    sbatch_path: str | None = None, hours: int = 8,
    partition: str = 'normal', mem: int = 32, cpu: int = 2,
    account: str = 'kempner_krajan_lab'
) -> str:
    """
    Create a job file.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        cmd: python command to be execute by the cluster
        job_name: str, name of the job file
        dep_ids: list, a list of job ids used for job dependency
        sbatch_path : str, Directory to store SBATCH file in.
        hours : int, number of hours to train
    Returns:
        job_file : str, Path to the job file.
    """
    assert type(dep_ids) is list, 'dependency ids must be list'
    assert all(type(id_) is str for id_ in dep_ids), 'dependency ids must all be strings'

    if len(dep_ids) == 0:
        dependency_line = ''
    else:
        dependency_line = '#SBATCH --dependency=afterok:' \
                          + ':'.join(dep_ids) + '\n'

    if sbatch_path is None:
        sbatch_path = os.path.join(ROOT_DIR, 'logs', 'jobs')
    sbatch_path = os.path.abspath(sbatch_path)
    os.makedirs(sbatch_path, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'logs'), exist_ok=True)
    job_file = os.path.join(sbatch_path, job_name + '.sh')

    with open(job_file, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '# Usage: sbatch this script\n'
            + '#SBATCH -t {}:00:00\n'.format(hours)
            + '#SBATCH -N 1\n'
            + '#SBATCH -n {}\n'.format(cpu)
            + '#SBATCH --mem={}G\n'.format(mem)
            + '#SBATCH --gres=gpu:1\n'
            + f'#SBATCH --account {account}\n'
            + '#SBATCH -p {}\n'.format(partition)
            + f'#SBATCH -e {ROOT_DIR}/logs/slurm-%j.out\n'
            + f'#SBATCH -o {ROOT_DIR}/logs/slurm-%j.out\n'
            + dependency_line
            + '\n'
            + 'source ~/.bashrc\n'
            + f'cd {ROOT_DIR}\n'
            + cmd + '\n'
            + '\n'
            )
        print(job_file)
    return job_file

def get_idle_gpu(subprocess_list: list, config_list: list) -> int:
    while True:
        for idx, pipe in enumerate(subprocess_list):
            if pipe is not None and pipe.poll() is not None:
                if pipe.returncode:
                    error = RuntimeError(
                        f'Fit process exited with code {pipe.returncode}.'
                    )
                    logging.error('%s: %s', config_list[idx].save_path,
                                  error)
                    record_failure(config_list[idx], error)
                subprocess_list[idx] = None
                config_list[idx] = None
            if subprocess_list[idx] is None:
                return idx
        time.sleep(10)

def record_failure(config: BaseConfig, error: Exception) -> None:
    """Append one fit failure to the local launch log.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Failed fit configuration.
    error : Exception
        Failure to record.
    """
    log_dir = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    record = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'fit_id': os.path.basename(config.save_path),
        'error': str(error),
    }
    with open(os.path.join(log_dir, 'failures.jsonl'), 'a') as stream:
        stream.write(json.dumps(record) + '\n')


def train_experiment(
    experiment: str, on_cluster: bool, on_server: bool,
    partition: str, account: str, eval: bool = False
) -> list[str]:
    """Train model across platforms given experiment name.
    adapted from https://github.com/gyyang/olfaction_evolution
    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        on_cluster: bool, whether to run experiments on cluster
        on_server: bool, if use on server

    Returns:
        return_ids: list, a list of job ids that are last in the dependency sequence
            if not using cluster, return is an empty list
    """
    if experiment in dir(experiments):
        # Get list of configurations from experiment function
        exp_configs = getattr(experiments, experiment)()
    else:
        raise ValueError('Experiment config not found: ', experiment)

    return_ids = []
    exp_configs = configs_dict_unpack(exp_configs)
    if not exp_configs or not isinstance(exp_configs[0], BaseConfig):
        raise ValueError(f'No valid fit configurations for {experiment}.')

    pending = []
    seen_paths = set()
    for config in exp_configs:
        if config.save_path in seen_paths:
            continue
        seen_paths.add(config.save_path)
        if validate_complete(config) or is_fit_active(config):
            continue
        if eval and not os.path.isfile(
            os.path.join(config.save_path, 'config.json')
        ):
            logging.warning(
                'No saved training configuration for %s',
                config.save_path
            )
            continue
        pending.append(config)

    if not pending:
        logging.info('No pending fits for %s', experiment)
        return return_ids
    if input(f'Continue running {len(pending)} jobs? (y/n)') != 'y':
        return return_ids

    if not eval:
        prepared = []
        for config in pending:
            try:
                if save_config(config, config.save_path):
                    prepared.append(config)
            except Exception as error:
                logging.exception('Cannot prepare fit %s', config.save_path)
                record_failure(config, error)
        pending = prepared
    if not pending:
        return return_ids

    log_dir = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if on_cluster:
        for config in pending:
            command = eval_cmd(config.save_path) if eval else train_cmd(config)
            job_name = config.save_path.rsplit('/', 1)[-1]
            try:
                job_file = get_jobfile(
                    command, job_name, dep_ids=[], hours=config.hours,
                    partition=partition, mem=config.mem, cpu=config.cpu,
                    sbatch_path=os.path.join(log_dir, 'jobs'),
                    account=account
                )
                result = subprocess.run(
                    ['sbatch', job_file], capture_output=True, check=True
                )
                stdout = result.stdout.decode().strip()
                logging.info('%s: %s', job_name, stdout)
                return_ids.append(stdout.rsplit(' ', 1)[-1])
            except (OSError, subprocess.CalledProcessError) as error:
                logging.exception(
                    'Submission failed for %s', config.save_path
                )
                record_failure(config, error)
    elif on_server:
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError('Server mode requires at least one GPU.')
        processes = [None] * num_gpus
        process_configs = [None] * num_gpus
        streams = []
        for config in pending:
            gpu = get_idle_gpu(processes, process_configs)
            command = eval_cmd(config.save_path) if eval else train_cmd(config)
            log_path = os.path.join(
                log_dir, config.save_path.rsplit('/', 1)[-1] + '.log'
            )
            stream = open(log_path, 'a', encoding='utf-8')
            streams.append(stream)
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
            try:
                processes[gpu] = subprocess.Popen(
                    command, stdout=stream, stderr=stream,
                    shell=True, env=environment
                )
                process_configs[gpu] = config
            except OSError as error:
                logging.exception('Cannot launch fit %s', config.save_path)
                record_failure(config, error)
                stream.close()
        for gpu, process in enumerate(processes):
            if process is not None and process.wait():
                error = RuntimeError(
                    f'Fit process exited with code {process.returncode}.'
                )
                logging.error('%s: %s', process_configs[gpu].save_path,
                              error)
                record_failure(process_configs[gpu], error)
        for stream in streams:
            stream.close()
    else:
        for config in pending:
            try:
                if eval:
                    complete_evaluation(config)
                else:
                    model_train(config)
            except Exception as error:
                logging.exception('Fit failed: %s', config.save_path)
                record_failure(config, error)

    return return_ids

def analyze_experiment(
    experiment: str, prev_ids: list[str], on_cluster: bool,
    partition: str, account: str
) -> None:
    """analyze experiments
     adapted from https://github.com/gyyang/olfaction_evolution

     Args:
         experiment: str, name of experiment to be analyzed
             must correspond to a function in exp_analysis.py
         prev_ids: list, list of previous job ids
         on_cluster: bool, if use on the cluster
     """
    import configs.exp_analysis as exp_analysis

    print('Analyzing {:s} experiment'.format(experiment))
    if (experiment + '_analysis') in dir(exp_analysis):
        if on_cluster:
            python_cmd = analysis_cmd(experiment)
            job_n = experiment + '_analysis'
            slurm_cmd = ['sbatch', get_jobfile(
                python_cmd, job_n, dep_ids=prev_ids,
                partition=partition, hours=24, mem=64,
                account=account
            )]
            cp_process = subprocess.run(slurm_cmd, capture_output=True,
                                        check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
        else:
            getattr(exp_analysis, experiment + '_analysis')()
    else:
        raise ValueError('Experiment analysis not found: ', experiment + '_analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs='+', help='Train experiments', default=[])
    parser.add_argument('-e', '--eval', nargs='+', help='Evaluate experiments', default=[])
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on Slurm cluster')
    parser.add_argument('-s', '--server', action='store_true', help='Run experiments on a Linux server with multiple GPUs')
    parser.add_argument('-p', '--partition', default='kempner', help='Partition of resource on cluster to use')
    parser.add_argument('--acc', default='kempner_krajan_lab', help='Account to use on cluster')
    
    args = parser.parse_args()
    experiments2train = args.train
    experiments2analyze = args.analyze
    experiments2eval = args.eval
    use_cluster = args.cluster
    use_server = args.server
    assert not (use_cluster and use_server), "Either use cluster or directly run on a server"

    logging.basicConfig(level=LOG_LEVEL)

    # evaluation jobs are executed after training jobs,
    # analysis jobs are executed after training and evaluation jobs.
    train_ids = []
    if experiments2train:
        for exp in experiments2train:
            exp_ids = train_experiment(
                exp, on_cluster=use_cluster, on_server=use_server,
                partition=args.partition, account=args.acc
            )
            train_ids += exp_ids

    eval_ids = []
    if experiments2eval:
        for exp in experiments2eval:
            exp_ids = train_experiment(exp, on_cluster=use_cluster, on_server=use_server, partition=args.partition, account=args.acc, eval=True)
            eval_ids += exp_ids

    if experiments2analyze:
        for exp in experiments2analyze:
            analyze_experiment(exp, prev_ids=train_ids + eval_ids,
                               on_cluster=use_cluster, partition=args.partition, account=args.acc)