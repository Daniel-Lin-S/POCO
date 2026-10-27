"""Run one neural preprocessing task or an Ahrens campaign.

Usage: ``python run_preprocess.py --dataset zebrafish_ahrens`` reads the
ignored ``configs/local/ahrens_preprocess.json`` settings and
processes each listed filter.
Other datasets use ``--dataset NAME [--filter MODE]``. Ahrens input and
output roots are configured in ``configs/local/paths.json``.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout

from configs.ahrens_preprocess import load_ahrens_preprocess_config
from utils.ahrens_sessions import raw_subject_directories
from utils.campaigns.lifecycle import CampaignLog
from preprocess.celegans import (
    celegans_flavell_preprocess, celegans_zimmer_preprocess,
)
from preprocess.mice import mice_preprocess
from preprocess.zebrafish import (
    PreprocessBusyError, process_zebrafish_activity,
    process_zebrafish_ahrens_activity,
)

FILTERED_PROCESSORS = {
    'celegans_zimmer': celegans_zimmer_preprocess,
    'celegans_flavell': celegans_flavell_preprocess,
    'mice': mice_preprocess,
}
FILTER_CHOICES = ('none', 'lowpass', 'highpass', 'bandpass')



def run_ahrens_campaign() -> None:
    """Process configured Ahrens filters in independent CPU workers."""
    settings = load_ahrens_preprocess_config()
    subjects = raw_subject_directories()
    jobs = [
        f'{mode}:{subject.name}'
        for mode in settings.filters
        for subject in subjects
    ]
    campaign = CampaignLog.create(
        'ahrens_preprocess', jobs,
        metadata={
            'filters': list(settings.filters),
            'pca_solver': settings.pca_solver,
            'pc_dim': settings.pc_dim,
            'pca_seed': settings.pca_seed,
            'storage_dtype': settings.storage_dtype,
        },
    )
    failures = []
    output = campaign.path / 'launcher.log'
    with output.open('a', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            with ProcessPoolExecutor(
                max_workers=len(settings.filters)
            ) as pool:
                futures = {
                    pool.submit(
                        process_zebrafish_ahrens_activity,
                        mode, settings, subjects, campaign.path,
                    ): mode
                    for mode in settings.filters
                }
                for future in as_completed(futures):
                    mode = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(
                            f'{mode}: {type(error).__name__}: {error}'
                        )
                        print(f'{mode} launch error: {error}')
                        states = campaign.read_status()['jobs']
                        state = (
                            'busy' if isinstance(
                                error, PreprocessBusyError
                            ) else 'failed'
                        )
                        for subject in subjects:
                            job_id = f'{mode}:{subject.name}'
                            if states[job_id] == 'pending':
                                detail = (
                                    str(error) if state == 'failed'
                                    else None
                                )
                                campaign.mark(job_id, state, detail)
    print(f'Ahrens campaign status: {campaign.path / "status.json"}')
    if failures:
        raise RuntimeError(
            'Ahrens preprocessing failures: ' + '; '.join(failures)
        )


def main() -> None:
    """Run the requested preprocessing task or local Ahrens campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset', required=True,
        choices=(*FILTERED_PROCESSORS, 'zebrafish',
                 'zebrafish_ahrens'),
    )
    parser.add_argument('--filter', choices=FILTER_CHOICES)
    args = parser.parse_args()
    if args.dataset == 'zebrafish_ahrens':
        if args.filter is not None:
            parser.error(
                'Ahrens filters are set in '
                'configs/local/ahrens_preprocess.json.'
            )
        run_ahrens_campaign()
    elif args.dataset == 'zebrafish':
        if args.filter not in (None, 'none'):
            parser.error('zebrafish supports only --filter none.')
        process_zebrafish_activity()
    else:
        FILTERED_PROCESSORS[args.dataset](
            filter_mode=args.filter or 'none'
        )


if __name__ == '__main__':
    main()
