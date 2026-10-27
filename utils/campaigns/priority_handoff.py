"""Hand the live server queue to a priority campaign and resume it.

Input is an existing paused main.py launcher PID, its campaign
status.json, and a priority experiment factory name. The four active
workers finish first. Output is the normal priority campaign lifecycle
folder, launcher.log and launcher.pid there, and a resumed original
launcher after priority training and analysis exit. No fit artifacts
are modified by this coordinator.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from utils.campaigns.lifecycle import LOG_ROOT

POLL_SECONDS = 10
GPU_SLOTS = 4
LAUNCH_STATUS_TIMEOUT_SECONDS = 120
WORKER_RELEASE_SECONDS = 5
PRIORITY_EXPERIMENT = 'ahrens_session_lowpass_fixed_priority'
ORIGINAL_EXPERIMENT = 'ahrens_session_lowpass_extended'


def process_start(pid: int) -> str:
    """Read one Linux process start tick to guard against PID reuse.

    Parameters
    ----------
    pid : int
        Process identifier to inspect.

    Returns
    -------
    str
        Kernel start tick from process stat field 22.
    """
    path = Path('/proc') / str(pid) / 'stat'
    fields = path.read_text().rsplit(')', 1)[1].split()
    return fields[19]


def verify_original(pid: int) -> str:
    """Require the exact live original campaign launcher process."""
    arguments = (
        (Path('/proc') / str(pid) / 'cmdline')
        .read_bytes().decode().split('\0')
    )
    if ('main.py' not in arguments or
            ORIGINAL_EXPERIMENT not in arguments):
        raise ValueError(
            f'PID {pid} is not the original Ahrens launcher.'
        )
    return process_start(pid)


def original_alive(pid: int, start: str) -> bool:
    """Check that a PID still refers to the verified original process."""
    try:
        return process_start(pid) == start
    except FileNotFoundError:
        return False


def wait_for_workers(
    status_path: Path, original_pid: int, start: str,
) -> None:
    """Wait for paused coordinator's active workers to finish."""
    from utils.campaigns.lifecycle import CampaignLog

    campaign = CampaignLog(status_path.parent)
    while True:
        if not original_alive(original_pid, start):
            raise RuntimeError(
                f'Original launcher PID {original_pid} disappeared.'
            )
        status = campaign.read_status()
        if status['running'] == 0:
            time.sleep(WORKER_RELEASE_SECONDS)
            return
        time.sleep(POLL_SECONDS)


def priority_status(log_path: Path, process: subprocess.Popen) -> Path:
    """Wait for a new priority lifecycle folder printed by main.py."""
    prefix = 'Campaign status: '
    deadline = time.monotonic() + LAUNCH_STATUS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if log_path.exists():
            for line in log_path.read_text(
                encoding='utf-8', errors='replace'
            ).splitlines():
                if line.startswith(prefix):
                    status = Path(line.removeprefix(prefix))
                    if (not status.is_absolute() or
                            status.parent.parent != LOG_ROOT or
                            not status.parent.name.startswith(
                                f'{PRIORITY_EXPERIMENT}_training_'
                            )):
                        raise ValueError(
                            f'Unexpected priority status path {status}.'
                        )
                    return status
        if process.poll() is not None:
            raise RuntimeError(
                f'Priority launcher exited {process.returncode} '
                'before creating its status.'
            )
        time.sleep(1)
    raise TimeoutError(
        'Priority launcher did not create a status within '
        f'{LAUNCH_STATUS_TIMEOUT_SECONDS} seconds.'
    )


def launch_priority() -> int:
    """Run priority fits and their early comparison on four GPUs."""
    descriptor, temporary = tempfile.mkstemp(
        prefix='poco_fixed_priority_', suffix='.log'
    )
    temporary_path = Path(temporary)
    command = [
        sys.executable, '-u', 'main.py',
        '-t', PRIORITY_EXPERIMENT,
        '-a', PRIORITY_EXPERIMENT,
        '-s', '--top-gpus', str(GPU_SLOTS), '--yes',
    ]
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=stream,
            start_new_session=True,
        )
        try:
            status_path = priority_status(temporary_path, process)
            log_dir = status_path.parent
            os.replace(temporary_path, log_dir / 'launcher.log')
            (log_dir / 'launcher.pid').write_text(
                f'{process.pid}\n', encoding='utf-8'
            )
            print(
                f'Priority status: {status_path}',
                flush=True,
            )
            return process.wait()
        finally:
            if process.poll() is None:
                process.wait()
            temporary_path.unlink(missing_ok=True)


def guard_resume(
    original_pid: int, original_start: str,
    priority_pid: int,
) -> None:
    """Release a paused queue if its priority launcher has exited.

    Parameters
    ----------
    original_pid : int
        Paused original queue coordinator PID.
    original_start : str
        Verified kernel start tick for that coordinator.
    priority_pid : int
        Live priority launcher PID to monitor independently.
    """
    arguments = (
        (Path('/proc') / str(priority_pid) / 'cmdline')
        .read_bytes().decode().split('\0')
    )
    if ('main.py' not in arguments or
            PRIORITY_EXPERIMENT not in arguments):
        raise ValueError(
            f'PID {priority_pid} is not the priority launcher.'
        )
    priority_start = process_start(priority_pid)
    while original_alive(priority_pid, priority_start):
        time.sleep(POLL_SECONDS)
    if not original_alive(original_pid, original_start):
        raise RuntimeError(
            f'Original launcher PID {original_pid} disappeared.'
        )
    time.sleep(WORKER_RELEASE_SECONDS)
    state = (
        (Path('/proc') / str(original_pid) / 'stat')
        .read_text().rsplit(')', 1)[1].split()[0]
    )
    if state in ('T', 't'):
        os.kill(original_pid, signal.SIGCONT)
        print('Resume guard released original queue.', flush=True)
    else:
        print('Original queue already resumed.', flush=True)


def main() -> None:
    """Run the verified handoff, resuming original queue on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-pid', type=int, required=True)
    parser.add_argument('--guard-priority-pid', type=int)
    parser.add_argument(
        '--original-status', type=Path, required=True
    )
    args = parser.parse_args()
    start = verify_original(args.original_pid)
    if not args.original_status.is_absolute():
        raise ValueError(
            f'Expected absolute status path, got '
            f'{args.original_status}.'
        )
    if args.guard_priority_pid is not None:
        guard_resume(
            args.original_pid, start,
            args.guard_priority_pid,
        )
        return
    try:
        wait_for_workers(
            args.original_status, args.original_pid, start
        )
        print('Original active workers finished.', flush=True)
        result = launch_priority()
        print(
            f'Priority launcher exited with code {result}.',
            flush=True,
        )
        if result != 0:
            raise RuntimeError(
                f'Priority launcher exited with code {result}.'
            )
    finally:
        if original_alive(args.original_pid, start):
            os.kill(args.original_pid, signal.SIGCONT)
            print('Original queue resumed.', flush=True)


if __name__ == '__main__':
    main()
