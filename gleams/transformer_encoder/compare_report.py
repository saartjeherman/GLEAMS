"""Score the CNN and transformer on the common holdout pairs and report.

Run after both `compare_cnn.py` and `compare_transformer.py` have written their
distance files into the run folder. Keeps only pairs both models could embed,
then writes `comparison.csv` and `comparison.png` (both stamped with which
model/checkpoint produced each result) into the same run folder.

    python compare_report.py [--run-dir DIR]

Without --run-dir the most recent run folder is used.
"""

import argparse
import os

import compare_common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=None,
                        help='Comparison run folder (default: the latest).')
    args = parser.parse_args()
    run_dir = compare_common.resolve_run_dir(args.run_dir, make_new=False)
    print(f"[report] run dir: {run_dir}")

    distance_files = {
        'CNN (original)': os.path.join(run_dir, 'cnn_distances.npz'),
        'Transformer': os.path.join(run_dir, 'transformer_distances.npz'),
    }
    missing = [p for p in distance_files.values() if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "Missing distance files: " + ", ".join(missing) +
            "\nRun compare_cnn.py (gleams env) and compare_transformer.py "
            "(base env) into this run folder first.")
    compare_common.build_report(distance_files, run_dir)


if __name__ == '__main__':
    main()
