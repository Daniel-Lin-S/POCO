"""Launch POCO experiment fits and analyses.

Input is an experiment factory name from ``configs.experiments`` and a local
``configs/local/paths.json`` containing ``artifact_root``. Output is
one hashed fit directory per scientific configuration, figures under
the artifact root, and per-campaign lifecycle folders under the
repository's ``logs`` directory.
"""

import os
from collections import deque
import subprocess
import argparse
import logging
import time
import shlex
import sys

import configs.experiments as experiments
import torch

from utils.config_utils import configs_dict_unpack
from configs.config_global import LOG_LEVEL, ROOT_DIR
from utils.artifacts import fit_id, validate_complete
from utils.campaigns.lifecycle import CampaignLog
from utils.campaigns.fit_worker import fit_command, run_fit
from utils.campaigns.claims import is_claimed
from configs.configs import BaseConfig

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
    managed_lifecycle: bool = False,
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
    managed_lifecycle : bool, optional
        Worker records its own lifecycle, default False.
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
    lines.extend(['source ~/.bashrc', f'cd {shlex.quote(ROOT_DIR)}'])
    if managed_lifecycle:
        lines.append(cmd)
    else:
        lines.extend([
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
    """Return the scientific fit hash for lifecycle tracking."""
    return fit_id(config)


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
    state = campaign.read_status()['jobs'][fit_job_id(config)]
    if state in ('finished', 'reused', 'failed'):
        return
    if process.returncode == 0 and validate_complete(config):
        campaign.mark(fit_job_id(config), 'finished')
    else:
        error = RuntimeError(
            f'Fit process exited with code {process.returncode} '
            'or has incomplete artifacts.'
        )
        logging.error('%s: %s', config.save_path, error)
        record_failure(campaign, config, error)


CLAIM_POLL_SECONDS = 10


def run_server_queue(
    pending: list[BaseConfig], selected_gpus: list[int],
    campaign: CampaignLog, action: str,
) -> None:
    """Run independent fits while claimed fits wait off GPU slots.

    Parameters
    ----------
    pending : list[BaseConfig]
        Unfinished fit configurations in scientific queue order.
    selected_gpus : list[int]
        Physical CUDA devices assigned one concurrent fit each.
    campaign : CampaignLog
        Shared lifecycle record for the queue.
    action : str
        ``train`` or ``eval`` worker action.
    """
    waiting = deque(pending)
    processes = [None] * len(selected_gpus)
    process_configs = [None] * len(selected_gpus)
    streams = [None] * len(selected_gpus)
    while waiting or any(process is not None for process in processes):
        for slot, process in enumerate(processes):
            if process is not None and process.poll() is not None:
                finish_fit_process(
                    process, process_configs[slot], campaign
                )
                streams[slot].close()
                processes[slot] = None
                process_configs[slot] = None
                streams[slot] = None
            if processes[slot] is not None:
                continue
            for _ in range(len(waiting)):
                config = waiting.popleft()
                job_id = fit_job_id(config)
                if validate_complete(config):
                    campaign.mark(job_id, 'reused')
                    continue
                if is_claimed(config):
                    campaign.mark(job_id, 'busy')
                    waiting.append(config)
                    continue
                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = str(
                    selected_gpus[slot]
                )
                stream = campaign.job_log_path(job_id).open(
                    'a', encoding='utf-8'
                )
                command = fit_command(config, campaign, action)
                try:
                    processes[slot] = subprocess.Popen(
                        command, stdout=stream, stderr=stream,
                        env=environment,
                    )
                    process_configs[slot] = config
                    streams[slot] = stream
                except OSError as error:
                    logging.exception(
                        'Cannot launch fit %s', config.save_path
                    )
                    record_failure(campaign, config, error)
                    stream.close()
                break
        if waiting or any(process is not None for process in processes):
            time.sleep(CLAIM_POLL_SECONDS)


def most_vacant_gpus(count: int) -> list[int]:
    """Select visible CUDA devices by free memory at launch.

    Parameters
    ----------
    count : int
        Required number of concurrent fit slots.

    Returns
    -------
    list[int]
        CUDA device indices in descending free-memory order.
    """
    available = torch.cuda.device_count()
    if count < 1 or available < count:
        raise RuntimeError(
            f'Expected {count} visible CUDA devices, got {available}.'
        )
    free = {
        index: torch.cuda.mem_get_info(index)[0]
        for index in range(available)
    }
    selected = sorted(free, key=free.get, reverse=True)[:count]
    logging.info('Selected GPUs by free memory: %s', selected)
    return selected


def train_experiment(
    experiment: str, on_cluster: bool, on_server: bool,
    partition: str, account: str, eval: bool = False,
    max_gpus: int | None = None, yes: bool = False,
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
    max_gpus : int or None, optional
        Number of most vacant visible GPUs, default all visible GPUs.
    yes : bool, optional
        Skip the interactive launch prompt, default False.

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
        elif eval and not os.path.isfile(
            os.path.join(config.save_path, 'config.json')
        ):
            decisions[job_id] = 'missing_config'
        else:
            decisions[job_id] = 'pending'
            pending.append(config)
    if pending and not yes and input(
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
    if not pending:
        return []
    return_ids = []
    if on_cluster:
        waiting = deque(pending)
        while waiting:
            for _ in range(len(waiting)):
                config = waiting.popleft()
                job_id = fit_job_id(config)
                if validate_complete(config):
                    campaign.mark(job_id, 'reused')
                    continue
                if is_claimed(config):
                    campaign.mark(job_id, 'busy')
                    waiting.append(config)
                    continue
                command = shlex.join(fit_command(
                    config, campaign, 'eval' if eval else 'train'
                ))
                try:
                    job_file = get_jobfile(
                        command, job_id, [], campaign,
                        managed_lifecycle=True,
                        hours=config.hours, partition=partition,
                        mem=config.mem, cpu=config.cpu, account=account,
                    )
                    campaign.mark(job_id, 'submitted')
                    result = subprocess.run(
                        ['sbatch', job_file], capture_output=True,
                        check=True
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
            if waiting:
                time.sleep(CLAIM_POLL_SECONDS)
    elif on_server:
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            error = RuntimeError(
                'Server mode requires at least one GPU.'
            )
            for config in pending:
                record_failure(campaign, config, error)
            raise error
        try:
            selected_gpus = (
                most_vacant_gpus(max_gpus)
                if max_gpus is not None else list(range(num_gpus))
            )
        except RuntimeError as error:
            for config in pending:
                record_failure(campaign, config, error)
            raise
        run_server_queue(
            pending, selected_gpus, campaign,
            'eval' if eval else 'train',
        )
    else:
        for config in pending:
            job_id = fit_job_id(config)
            try:
                run_fit(config, campaign, 'eval' if eval else 'train')
            except Exception:
                logging.exception('Fit failed: %s', config.save_path)
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
    parser.add_argument(
        '-t', '--train', nargs='+', help='Train experiments', default=[]
    )
    parser.add_argument(
        '-e', '--eval', nargs='+', help='Evaluate experiments', default=[]
    )
    parser.add_argument(
        '-a', '--analyze', nargs='+', help='Analyze experiments',
        default=[]
    )
    parser.add_argument(
        '-c', '--cluster', action='store_true',
        help='Use batch submission on Slurm cluster'
    )
    parser.add_argument(
        '-s', '--server', action='store_true',
        help='Run experiments on a Linux server with multiple GPUs'
    )
    parser.add_argument(
        '-p', '--partition', default='kempner',
        help='Partition of resource on cluster to use'
    )
    parser.add_argument('--acc', default='kempner_krajan_lab',
                        help='Account to use on cluster')
    parser.add_argument('--top-gpus', type=int, default=None,
                        help='Use this many most vacant visible GPUs')
    parser.add_argument('--yes', action='store_true',
                        help='Launch pending fits without prompting')
    
    args = parser.parse_args()
    experiments2train = args.train
    experiments2analyze = args.analyze
    experiments2eval = args.eval
    use_cluster = args.cluster
    use_server = args.server
    assert not (use_cluster and use_server), (
        'Either use cluster or directly run on a server'
    )
    if args.top_gpus is not None and not use_server:
        parser.error('--top-gpus requires --server.')

    logging.basicConfig(level=LOG_LEVEL)

    # evaluation jobs are executed after training jobs,
    # analysis jobs are executed after training and evaluation jobs.
    train_ids = []
    if experiments2train:
        for exp in experiments2train:
            exp_ids = train_experiment(
                exp, on_cluster=use_cluster, on_server=use_server,
                partition=args.partition, account=args.acc,
                max_gpus=args.top_gpus, yes=args.yes
            )
            train_ids += exp_ids

    eval_ids = []
    if experiments2eval:
        for exp in experiments2eval:
            exp_ids = train_experiment(
                exp, on_cluster=use_cluster, on_server=use_server,
                partition=args.partition, account=args.acc, eval=True
            )
            eval_ids += exp_ids

    if experiments2analyze:
        for exp in experiments2analyze:
            analyze_experiment(
                exp, prev_ids=train_ids + eval_ids,
                on_cluster=use_cluster, partition=args.partition,
                account=args.acc
            )