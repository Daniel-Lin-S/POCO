"""Run one named neural preprocessing task.

Usage: ``python run_preprocess.py --dataset zebrafish_ahrens
[--filter none|lowpass|highpass|bandpass]``. Ahrens input is the locally
configured raw subject directory; output is subject-named NPZ archives in
the locally configured processed directory. Other datasets use their
configured raw and processed directories.
"""

import argparse

from preprocess.celegans import (
    celegans_flavell_preprocess, celegans_zimmer_preprocess,
)
from preprocess.mice import mice_preprocess
from preprocess.zebrafish import (
    process_zebrafish_activity, process_zebrafish_ahrens_activity,
)

FILTERED_PROCESSORS = {
    'zebrafish_ahrens': process_zebrafish_ahrens_activity,
    'celegans_zimmer': celegans_zimmer_preprocess,
    'celegans_flavell': celegans_flavell_preprocess,
    'mice': mice_preprocess,
}
FILTER_CHOICES = ('none', 'lowpass', 'highpass', 'bandpass')


def main() -> None:
    """Run exactly the requested preprocessing task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset', required=True,
        choices=(*FILTERED_PROCESSORS, 'zebrafish'),
    )
    parser.add_argument('--filter', choices=FILTER_CHOICES, default='none')
    args = parser.parse_args()
    if args.dataset == 'zebrafish':
        if args.filter != 'none':
            parser.error('zebrafish supports only --filter none.')
        process_zebrafish_activity()
    else:
        FILTERED_PROCESSORS[args.dataset](filter_mode=args.filter)


if __name__ == '__main__':
    main()
