"""Entry point for one transformer training run.

Creates `results/<YYYY-MM-DD_HH-MM-SS>/` and writes everything from the run
(training.log, loss_log.csv, best_model.pt, final_model.pt, plots/) into it.

Run from this folder:
    python train.py
"""

import os
import sys
import time
import traceback

import config
from training import train_model


class _Tee:
    """Fan stdout/stderr writes out to multiple file-like objects."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    run_id = time.strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(config.RESULTS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'training.log')

    with open(log_path, 'a') as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

        banner = "=" * 80
        print(banner)
        print(f"Starting new training run - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Run directory: {run_dir}")
        print(banner)

        try:
            train_model(run_dir)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
