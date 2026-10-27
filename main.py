"""Launch POCO experiment fits and analyses.

Input is an experiment factory name from ``configs.experiments`` and a local
``configs/local/paths.json`` containing ``artifact_root``. Output is
one hashed fit directory per scientific configuration, figures under
the artifact root, and per-campaign lifecycle folders under the
repository's ``logs`` directory.
"""

import os
import subprocess
import argparse
import logging
import time
import shlex
import sys

import configs.experiments as experiments
import torch

from utils.config_utils import save_config, configs_dict_unpack
from train import model_train, complete_evaluation
from configs.config_global import LOG_LEVEL, ROOT_DIR
from utils.artifacts import validate_complete, is_fit_active
from utils.campaigns.lifecycle import CampaignLog
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
    campaign: CampaignLog, fit_path: str | None = None,
    hours: int = 8, partition: str = 'normal',
    mem: int = 32, cpu: int = 2,
    account: str = 'kempner_krajan_lab',
) -> str:
    """Write one Slurm script and its output path inside a campaign.

    Parameters
    ----------
    cmd : str
        Shell command for the job body.
    job_name : str
        Unique job identifier within the campaign.
    dep_ids : list[str]
        Slurm IDs required before this job can start.
    campaign : CampaignLog
        Per-launch lifecycle log folder.
    fit_path : str or None, optional
        Fit directory to validate before marking finished, default None.
    hours : int, optional
        Wall-time hours, default 8.
    partition : str, optional
        Slurm partition, default normal.
    mem : int, optional
        Requested memory in GiB, default 32.
    cpu : int, optional
        Requested CPU count, default 2.
    account : str, optional
        Slurm account, default kempner_krajan_lab.

    Returns
    -------
    str
        Absolute path to the generated script.
    """
    if not isinstance(dep_ids, list) or not all(
        isinstance(job_id, str) for job_id in dep_ids
    ):
        raise ValueError('Slurm dependencies must be a list of IDs.')
    log_path = campaign.job_log_path(job_name)
    job_file = log_path.with_suffix('.sh')
    marker = (
        f'{shlex.quote(sys.executable)} -m utils.campaigns.lifecycle '
        f'{shlex.quote(str(campaign.path))} '
        f'{shlex.quote(job_name)}'
    )
    finished = f'{marker} finished'
    if fit_path is not None:
        finished += f' --fit-path {shlex.quote(fit_path)}'
    lines = [
        '#!/bin/bash',
        '# Usage: sbatch this script',
        f'#SBATCH -t {hours}:00:00',
        '#SBATCH -N 1',
        f'#SBATCH -n {cpu}',
        f'#SBATCH --mem={mem}G',
        '#SBATCH --gres=gpu:1',
        f'#SBATCH --account {account}',
        f'#SBATCH -p {partition}',
        f'#SBATCH -e {log_path}',
        f'#SBATCH -o {log_path}',
    ]
    if dep_ids:
        lines.append(
            '#SBATCH --dependency=afterok:' + ':'.join(dep_ids)
        )
    lines.extend([
        'source ~/.bashrc',
        f'cd {shlex.quote(ROOT_DIR)}',
        f'{marker} running',
        cmd,
        'poco_exit_code=$?',
        'if [ "$poco_exit_code" -eq 0 ]; then',
        f'    {finished}',
        '    poco_mark_code=$?',
        '    if [ "$poco_mark_code" -ne 0 ]; then',
        '        exit "$poco_mark_code"',
        '    fi',
        'else',
        f'    {marker} failed --exit-code "$poco_exit_code"',
        'fi',
        'exit "$poco_exit_code"',
    ])
    job_file.write_text('\n'.join(lines) + '\n')
    print(job_file)
    return str(job_file)


def fit_job_id(config: BaseConfig) -> str:
    """Return the hash-named fit directory basename."""
    return os.path.basename(config.save_path)


def record_failure(
    campaign: CampaignLog, config: BaseConfig, error: Exception,
) -> None:
    """Record one failed fit attempt in its campaign folder."""
    campaign.mark(
        fit_job_id(config), 'failed',
        f'{type(error).__name__}: {error}',
    )


def finish_fit_process(
    process: subprocess.Popen, config: BaseConfig,
    campaign: CampaignLog,
) -> None:
    """Validate a finished server process and update its job state."""
    if process.returncode == 0 and validate_complete(config):
        campaign.mark(fit_job_id(config), 'finished')
    else:
        error = RuntimeError(
            f'Fit process exited with code {process.returncode} '
            'or has incomplete artifacts.'
        )
        logging.error('%s: %s', config.save_path, error)
        record_failure(campaign, config, error)


def get_idle_gpu(
    processes: list, configs: list, streams: list,
    campaign: CampaignLog,
) -> int:
    """Return one free GPU slot after logging completed processes."""
    while True:
        for index, process in enumerate(processes):
            if process is not None and process.poll() is not None:
                finish_fit_process(process, configs[index], campaign)
                streams[index].close()
                processes[index] = None
                configs[index] = None
                streams[index] = None
            if processes[index] is None:
                return index
        time.sleep(10)

def train_experiment(
    experiment: str, on_cluster: bool, on_server: bool,
    partition: str, account: str, eval: bool = False,
) -> list[str]:
    """Launch missing fits and track one timestamped campaign.

    Parameters
    ----------
    experiment : str
        Experiment factory in configs.experiments.
    on_cluster : bool
        Submit independent fits through Slurm.
    on_server : bool
        Run independent fits on visible server GPUs.
    partition : str
        Slurm partition when on_cluster is true.
    account : str
        Slurm account when on_cluster is true.
    eval : bool, optional
        Finalize evaluation instead of training, default False.

    Returns
    -------
    list[str]
        Submitted Slurm job IDs, empty for local execution.
    """
    factory = getattr(experiments, experiment, None)
    if factory is None or not callable(factory):
        raise ValueError(f'Experiment config not found: {experiment}.')
    exp_configs = configs_dict_unpack(factory())
    if not exp_configs or not isinstance(exp_configs[0], BaseConfig):
        raise ValueError(f'No valid fit configurations for {experiment}.')
    unique = {}
    for config in exp_configs:
        unique.setdefault(config.save_path, config)
    decisions = {}
    pending = []
    for config in unique.values():
        job_id = fit_job_id(config)
        if validate_complete(config):
            decisions[job_id] = 'reused'
        elif is_fit_active(config):
            decisions[job_id] = 'busy'
        elif eval and not os.path.isfile(
            os.path.join(config.save_path, 'config.json')
        ):
            decisions[job_id] = 'missing_config'
        else:
            decisions[job_id] = 'pending'
            pending.append(config)
    if pending and input(
        f'Continue running {len(pending)} jobs? (y/n)'
    ) != 'y':
        return []
    action = 'evaluation' if eval else 'training'
    campaign = CampaignLog.create(
        f'{experiment}_{action}', list(decisions)
    )
    print(f'Campaign status: {campaign.path / "status.json"}')
    for job_id, decision in decisions.items():
        if decision == 'missing_config':
            campaign.mark(
                job_id, 'failed',
                'No saved training configuration for evaluation.',
            )
        elif decision != 'pending':
            campaign.mark(job_id, decision)
    if not eval:
        prepared = []
        for config in pending:
            try:
                if save_config(config, config.save_path):
                    prepared.append(config)
                elif validate_complete(config):
                    campaign.mark(fit_job_id(config), 'reused')
                else:
                    campaign.mark(fit_job_id(config), 'busy')
            except Exception as error:
                logging.exception('Cannot prepare fit %s', config.save_path)
                record_failure(campaign, config, error)
        pending = prepared
    if not pending:
        return []
    return_ids = []
    if on_cluster:
        for config in pending:
            job_id = fit_job_id(config)
            command = eval_cmd(config.save_path) if eval else train_cmd(config)
            try:
                job_file = get_jobfile(
                    command, job_id, [], campaign,
                    fit_path=config.save_path,
                    hours=config.hours, partition=partition,
                    mem=config.mem, cpu=config.cpu, account=account,
                )
                campaign.mark(job_id, 'submitted')
                result = subprocess.run(
                    ['sbatch', job_file], capture_output=True, check=True
                )
                stdout = result.stdout.decode().strip()
                submitted_id = stdout.rsplit(' ', 1)[-1]
                if not submitted_id.isdecimal():
                    raise RuntimeError(
                        f'Invalid Slurm submission for {job_id}: '
                        f'{stdout!r}.'
                    )
                logging.info('%s: %s', job_id, stdout)
                return_ids.append(submitted_id)
            except (OSError, subprocess.CalledProcessError,
                    RuntimeError) as error:
                logging.exception('Submission failed for %s', job_id)
                record_failure(campaign, config, error)
    elif on_server:
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            error = RuntimeError(
                'Server mode requires at least one GPU.'
            )
            for config in pending:
                record_failure(campaign, config, error)
            raise error
        processes = [None] * num_gpus
        process_configs = [None] * num_gpus
        streams = [None] * num_gpus
        for config in pending:
            gpu = get_idle_gpu(
                processes, process_configs, streams, campaign
            )
            job_id = fit_job_id(config)
            command = eval_cmd(config.save_path) if eval else train_cmd(config)
            stream = campaign.job_log_path(job_id).open(
                'a', encoding='utf-8'
            )
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
            try:
                campaign.mark(job_id, 'running')
                processes[gpu] = subprocess.Popen(
                    command, stdout=stream, stderr=stream,
                    shell=True, env=environment,
                )
                process_configs[gpu] = config
                streams[gpu] = stream
            except OSError as error:
                logging.exception('Cannot launch fit %s', config.save_path)
                record_failure(campaign, config, error)
                stream.close()
        for gpu, process in enumerate(processes):
            if process is not None:
                process.wait()
                finish_fit_process(process, process_configs[gpu], campaign)
                streams[gpu].close()
    else:
        for config in pending:
            job_id = fit_job_id(config)
            campaign.mark(job_id, 'running')
            try:
                if eval:
                    complete_evaluation(config)
                else:
                    model_train(config)
                if not validate_complete(config):
                    raise RuntimeError(
                        f'Fit artifacts are incomplete: {config.save_path}'
                    )
                campaign.mark(job_id, 'finished')
            except Exception as error:
                logging.exception('Fit failed: %s', config.save_path)
                record_failure(campaign, config, error)
    return return_ids

def analyze_experiment(
    experiment: str, prev_ids: list[str], on_cluster: bool,
    partition: str, account: str,
) -> None:
    """Run or submit one analysis with campaign-local lifecycle logs.

    Parameters
    ----------
    experiment : str
        Experiment name with a corresponding analysis function.
    prev_ids : list[str]
        Slurm job IDs that analysis must await.
    on_cluster : bool
        Submit analysis through Slurm when true.
    partition : str
        Slurm partition.
    account : str
        Slurm account.
    """
    import configs.exp_analysis as exp_analysis

    function = getattr(exp_analysis, experiment + '_analysis', None)
    if function is None or not callable(function):
        raise ValueError(
            f'Experiment analysis not found: {experiment}_analysis.'
        )
    campaign = CampaignLog.create(
        f'{experiment}_analysis', ['analysis']
    )
    print(f'Campaign status: {campaign.path / "status.json"}')
    if on_cluster:
        try:
            command = analysis_cmd(experiment)
            script = get_jobfile(
                command, 'analysis', prev_ids, campaign,
                partition=partition, hours=24, mem=64,
                account=account,
            )
            campaign.mark('analysis', 'submitted')
            result = subprocess.run(
                ['sbatch', script], capture_output=True, check=True
            )
            stdout = result.stdout.decode().strip()
            if not stdout.rsplit(' ', 1)[-1].isdecimal():
                raise RuntimeError(
                    f'Invalid Slurm analysis submission: {stdout!r}.'
                )
            print(stdout)
        except (OSError, subprocess.CalledProcessError,
                RuntimeError) as error:
            campaign.mark(
                'analysis', 'failed',
                f'{type(error).__name__}: {error}',
            )
            raise
    else:
        campaign.mark('analysis', 'running')
        try:
            function()
        except Exception as error:
            campaign.mark(
                'analysis', 'failed',
                f'{type(error).__name__}: {error}',
            )
            raise
        campaign.mark('analysis', 'finished')


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