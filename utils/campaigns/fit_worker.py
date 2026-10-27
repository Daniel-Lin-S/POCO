"""Run one claimed POCO fit from a campaign launcher.

Input is a base64-encoded resolved configuration, action, campaign log
folder, and hash job ID. Output is the standard hash-named fit artifact
folder and lifecycle transitions in that campaign's ``status.json``.
The claim stays held while any incomplete artifact is prepared, trained,
evaluated, and validated. Completed fits are reused unchanged.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from configs.configs import NeuralPredictionConfig
from train import model_train, complete_evaluation
from utils.artifacts import (
    fit_id, is_fit_active, validate_complete,
)
from utils.campaigns.claims import fit_claim
from utils.campaigns.lifecycle import CampaignLog
from utils.config_utils import config_from_dict, save_config

FIT_LOCK_POLL_SECONDS = 10


def fit_command(
    config: NeuralPredictionConfig, campaign: CampaignLog,
    action: str,
) -> list[str]:
    """Build a worker command without a separate configuration file.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Resolved scientific fit and assigned artifact path.
    campaign : CampaignLog
        Lifecycle folder shared with the launching process.
    action : str
        ``train`` or ``eval``.

    Returns
    -------
    list[str]
        Executable and arguments for one independent worker.
    """
    payload = json.dumps(config.to_dict(), sort_keys=True)
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return [
        sys.executable, '-m', 'utils.campaigns.fit_worker',
        '--payload', encoded, '--campaign', str(campaign.path),
        '--job-id', fit_id(config), '--action', action,
    ]


def wait_for_old_fit(
    config: NeuralPredictionConfig, campaign: CampaignLog,
    job_id: str,
) -> bool:
    """Wait for a legacy worker's fit lock and test its final artifact.

    Returns
    -------
    bool
        True when the existing fit completed successfully.
    """
    while is_fit_active(config):
        campaign.mark(job_id, 'busy')
        time.sleep(FIT_LOCK_POLL_SECONDS)
    return validate_complete(config)


def run_fit(
    config: NeuralPredictionConfig, campaign: CampaignLog,
    action: str = 'train',
) -> str:
    """Claim, reuse or execute, and validate one fit.

    Parameters
    ----------
    config : NeuralPredictionConfig
        Fully resolved configuration and artifact path.
    campaign : CampaignLog
        Lifecycle writer containing this fit ID.
    action : str, optional
        ``train`` or ``eval``, default ``train``.

    Returns
    -------
    str
        ``reused`` or ``finished`` on success.
    """
    if action not in ('train', 'eval'):
        raise ValueError(f'Unknown fit action {action!r}.')
    job_id = fit_id(config)
    if Path(config.save_path).name != job_id:
        raise ValueError(
            f'Fit path {config.save_path} does not match {job_id}.'
        )
    try:
        with fit_claim(
            config, on_wait=lambda: campaign.mark(job_id, 'busy')
        ):
            if validate_complete(config) or wait_for_old_fit(
                config, campaign, job_id
            ):
                campaign.mark(job_id, 'reused')
                return 'reused'
            if action == 'train':
                if not save_config(config, config.save_path):
                    if validate_complete(config):
                        campaign.mark(job_id, 'reused')
                        return 'reused'
                    raise RuntimeError(
                        f'Fit became busy during preparation: '
                        f'{config.save_path}'
                    )
                campaign.mark(job_id, 'running')
                model_train(config)
            else:
                if not (Path(config.save_path) / 'config.json').is_file():
                    raise FileNotFoundError(
                        f'No training configuration for evaluation: '
                        f'{config.save_path}'
                    )
                campaign.mark(job_id, 'running')
                complete_evaluation(config)
            if not validate_complete(config):
                raise RuntimeError(
                    f'Fit artifacts are incomplete: {config.save_path}'
                )
            campaign.mark(job_id, 'finished')
            return 'finished'
    except Exception as error:
        campaign.mark(
            job_id, 'failed', f'{type(error).__name__}: {error}'
        )
        raise


def main() -> None:
    """Restore a serialized configuration and execute its campaign fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--action', choices=('train', 'eval'), required=True)
    args = parser.parse_args()
    campaign = CampaignLog(args.campaign)
    content = base64.urlsafe_b64decode(args.payload.encode())
    config = config_from_dict(json.loads(content))
    if fit_id(config) != args.job_id:
        raise ValueError(
            f'Expected fit ID {args.job_id}, got {fit_id(config)}.'
        )
    run_fit(config, campaign, args.action)


if __name__ == '__main__':
    main()
