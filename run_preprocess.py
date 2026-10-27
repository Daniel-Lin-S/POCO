"""Run one neural preprocessing task or an Ahrens campaign.

Usage: ``python run_preprocess.py --dataset zebrafish_ahrens`` reads the
ignored ``configs/local/ahrens_preprocess.json`` settings and
processes each listed filter.
Other datasets use ``--dataset NAME [--filter MODE]``. Ahrens input and
output roots are configured in ``configs/local/paths.json``.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from configs.ahrens_preprocess import load_ahrens_preprocess_config
from preprocess.celegans import (
    celegans_flavell_preprocess, celegans_zimmer_preprocess,
)
from preprocess.mice import mice_preprocess
from preprocess.zebrafish import (
    process_zebrafish_activity, process_zebrafish_ahrens_activity,
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
    failures = []
    with ProcessPoolExecutor(max_workers=len(settings.filters)) as pool:
        futures = {
            pool.submit(
                process_zebrafish_ahrens_activity, mode, settings
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
