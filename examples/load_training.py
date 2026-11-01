"""Read a few text examples without preparing the full Arrow source corpus."""
import argparse
from itertools import islice

from finepdf_release import FinePDFResolver, iter_release_rows, iter_training_examples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--release-dir', required=True)
parser.add_argument('--config', choices=('eng_Latn', 'fra_Latn'), required=True)
parser.add_argument('--local-source-root')
args = parser.parse_args()
with FinePDFResolver(args.release_dir, config=args.config, local_source_root=args.local_source_root) as resolver:
    rows = iter_release_rows(args.release_dir, config=args.config)
    for example in islice(iter_training_examples(rows, resolver, num_negatives=8), 3):
        print(example)
