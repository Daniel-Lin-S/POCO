"""Store launch lifecycle records under one folder per campaign.

Input is a readable campaign name and unique job identifiers. Output is
``logs/<name>_<UTC launch time>/``. Its ``status.json`` records
campaign, launch/update times, state, total, finished, pending,
running, submitted, busy, failed, reused, remaining, current_jobs,
job states, and small metadata. ``events.jsonl`` records time, job ID,
and state transitions. Optional ``failures.jsonl`` adds error text;
``jobs/<fit_id>.log`` and launcher logs hold text output. Fit artifacts
stay outside logs.
"""

import argparse
import fcntl
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LOG_ROOT = Path(__file__).resolve().parents[2] / 'logs'
JOB_STATES = {
    'pending', 'running', 'submitted', 'finished', 'reused',
    'failed', 'busy',
}
FINISHED_STATES = {'finished', 'reused'}
TERMINAL_STATES = FINISHED_STATES | {'failed'}
NAME_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]*$')
JOB_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
TIME_FORMAT = '%Y%m%dT%H%M%S.%fZ'


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def atomic_json(path: Path, content: dict) -> None:
    """Replace one status snapshot without exposing a partial JSON file.

    Parameters
    ----------
    path : Path
        Destination status.json path.
    content : dict
        JSON-compatible lifecycle snapshot.
    """
    descriptor, pending = tempfile.mkstemp(
        prefix='.status.', suffix='.pending', dir=path.parent
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(content, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def refresh_counts(status: dict) -> None:
    """Recompute campaign counts from the authoritative job states.

    Parameters
    ----------
    status : dict
        Mutable status snapshot containing a ``jobs`` mapping.
    """
    states = status['jobs']
    status['total'] = len(states)
    for state in JOB_STATES:
        status[state] = sum(value == state for value in states.values())
    status['finished'] += status['reused']
    status['active'] = status['running']
    status['remaining'] = (
        status['total'] - status['finished'] - status['failed']
    )
    status['current_jobs'] = sorted(
        key for key, value in states.items() if value == 'running'
    )
    if status['finished'] == status['total']:
        status['state'] = 'complete'
    elif status['active']:
        status['state'] = 'running'
    elif status['submitted']:
        status['state'] = 'submitted'
    elif status['failed']:
        status['state'] = 'failed'
    elif status['busy']:
        status['state'] = 'deferred'
    else:
        status['state'] = 'pending'


class CampaignLog:
    """Coordinate small lifecycle records across independent workers."""

    def __init__(self, path: Path) -> None:
        """Refer to an existing campaign log folder.

        Parameters
        ----------
        path : Path
            Absolute campaign log directory.
        """
        self.path = path.resolve()
        if not (self.path / 'status.json').is_file():
            raise FileNotFoundError(
                f'Campaign status is missing: {self.path / "status.json"}'
            )

    @classmethod
    def create(
        cls, name: str, job_ids: list[str],
        launched_at: datetime | None = None,
        metadata: dict[str, object] | None = None,
    ) -> 'CampaignLog':
        """Create a unique timestamped campaign log directory.

        Parameters
        ----------
        name : str
            Readable lowercase campaign name with underscores or hyphens.
        job_ids : list[str]
            Unique identifiers for all planned jobs.
        launched_at : datetime or None, optional
            UTC launch time, default current time. Used for log migration.
        metadata : dict[str, object] or None, optional
            Small campaign settings, default None.

        Returns
        -------
        CampaignLog
            New campaign lifecycle writer.
        """
        if not NAME_PATTERN.fullmatch(name):
            raise ValueError(f'Invalid campaign name {name!r}.')
        if not job_ids or not all(
            isinstance(job, str) and job for job in job_ids
        ):
            raise ValueError('Every campaign job needs a nonempty ID.')
        if len(job_ids) != len(set(job_ids)):
            raise ValueError('Campaign jobs must be unique.')
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Campaign metadata must be a mapping.')
        start = launched_at or utc_now()
        if start.tzinfo is None:
            raise ValueError('Campaign launch time must include timezone.')
        start = start.astimezone(timezone.utc)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        path = LOG_ROOT / f'{name}_{start.strftime(TIME_FORMAT)}'
        path.mkdir()
        status = {
            'campaign': name,
            'launched_at_utc': start.isoformat(),
            'updated_at_utc': utc_now().isoformat(),
            'jobs': {job: 'pending' for job in job_ids},
            'metadata': metadata if metadata is not None else {},
        }
        refresh_counts(status)
        atomic_json(path / 'status.json', status)
        with (path / 'events.jsonl').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({
                'time_utc': start.isoformat(),
                'event': 'campaign_created',
                'total': len(job_ids),
            }) + '\n')
        return cls(path)

    def read_status(self) -> dict:
        """Read the current complete status snapshot."""
        with (self.path / 'status.json').open(encoding='utf-8') as stream:
            return json.load(stream)

    def mark(
        self, job_id: str, state: str, error: str | None = None,
    ) -> None:
        """Record an idempotent job transition and update status.

        Parameters
        ----------
        job_id : str
            Job identifier already listed in status.json.
        state : str
            New job state from JOB_STATES.
        error : str or None, optional
            Failure detail when state is failed, default None.
        """
        if state not in JOB_STATES:
            raise ValueError(f'Unknown campaign job state {state!r}.')
        if error is not None and state != 'failed':
            raise ValueError('Failure detail requires state failed.')
        lock_path = self.path / '.status.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            status = self.read_status()
            if job_id not in status['jobs']:
                raise KeyError(f'Unknown campaign job {job_id!r}.')
            previous = status['jobs'][job_id]
            if previous == state:
                return
            if previous in TERMINAL_STATES:
                raise ValueError(
                    f'Cannot change completed job {job_id!r} from '
                    f'{previous} to {state}.'
                )
            if previous == 'running' and state == 'submitted':
                return
            status['jobs'][job_id] = state
            status['updated_at_utc'] = utc_now().isoformat()
            refresh_counts(status)
            atomic_json(self.path / 'status.json', status)
            event = {
                'time_utc': status['updated_at_utc'],
                'job_id': job_id,
                'state': state,
            }
            with (self.path / 'events.jsonl').open(
                'a', encoding='utf-8'
            ) as stream:
                stream.write(json.dumps(event) + '\n')
            if error is not None:
                failure = {**event, 'error': error}
                with (self.path / 'failures.jsonl').open(
                    'a', encoding='utf-8'
                ) as stream:
                    stream.write(json.dumps(failure) + '\n')

    def job_log_path(self, job_id: str) -> Path:
        """Return a safe per-job text log path inside this campaign."""
        if not JOB_NAME_PATTERN.fullmatch(job_id):
            raise ValueError(f'Invalid job log name {job_id!r}.')
        directory = self.path / 'jobs'
        directory.mkdir(exist_ok=True)
        return directory / f'{job_id}.log'


def main() -> None:
    """Mark one Slurm worker transition from a generated job script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign_dir', type=Path)
    parser.add_argument('job_id')
    parser.add_argument('state', choices=sorted(JOB_STATES))
    parser.add_argument('--exit-code', type=int)
    parser.add_argument('--fit-path', type=Path)
    args = parser.parse_args()
    campaign = CampaignLog(args.campaign_dir)
    if args.fit_path is not None:
        if args.state != 'finished':
            parser.error('--fit-path requires state finished.')
        from utils.artifacts import validate_complete
        from utils.config_utils import load_config
        try:
            config = load_config(str(args.fit_path))
            complete = validate_complete(config)
        except (OSError, ValueError, KeyError) as error:
            campaign.mark(
                args.job_id, 'failed',
                f'Cannot validate fit artifacts: {error}',
            )
            raise SystemExit(1) from error
        if not complete:
            campaign.mark(
                args.job_id, 'failed',
                f'Fit artifacts are incomplete: {args.fit_path}',
            )
            raise SystemExit(1)
    error = None
    if args.exit_code is not None:
        error = f'Job exited with code {args.exit_code}.'
    campaign.mark(args.job_id, args.state, error)


if __name__ == '__main__':
    main()
