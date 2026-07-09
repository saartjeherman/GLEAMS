"""Inspect the spectra / pairs data used for transformer training.

Answers, from the artifacts in `data/`:

  1. Metadata collection      — load the combined metadata parquet.
  2. Grouping                 — group spectra by (peptide sequence, charge).
  3. Replicates               — how many peptides have >1 spectrum; total spectra.
  4. Peptide-level split      — split train/val so NO peptide (sequence) appears
                                in both. Contrast with the *existing* split, which
                                is a row-level train_test_split and therefore leaks
                                peptides across train/val.
  5. Spectra per split.
  6. Positive pairs           — all same-(sequence,charge) combinations, per split.
  7. Negative pairs           — reported from the existing .npy files (their
                                generation is expensive; see notes below).

Run from this folder:
    python inspect_spectra.py

The full report is echoed to the terminal and written to a log file
(`inspect_spectra.log` by default; override with --log, disable with --no-log).

Nothing else is written to disk unless you pass --write-split, which materializes
the peptide-level split parquets next to the existing metadata (without overwriting).
"""

import argparse
import os
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

# Append gleams/ to the path (not insert-at-0) so this folder's local config.py
# wins over the top-level gleams/config.py, matching how train.py resolves it.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _count_pairs_on_disk(metadata_filename: str) -> pd.DataFrame:
    """Read the per-charge pos/neg .npy pair files for a metadata parquet."""
    base = metadata_filename.replace('.parquet', '')
    rows = []
    for charge in range(config.CHARGES[0], config.CHARGES[1] + 1):
        pos_p = Path(f'{base}_pairs_pos_{charge}.npy')
        neg_p = Path(f'{base}_pairs_neg_{charge}.npy')
        n_pos = len(np.load(pos_p)) if pos_p.exists() else None
        n_neg = len(np.load(neg_p)) if neg_p.exists() else None
        rows.append({'charge': charge, 'pos': n_pos, 'neg': n_neg})
    return pd.DataFrame(rows).set_index('charge')


def _positive_pairs_from_metadata(df: pd.DataFrame) -> pd.Series:
    """Count positive pairs per charge directly from a metadata frame.

    Matches gleams.metadata.generate_pairs_positive: all C(n, 2) combinations of
    rows that share the same (sequence, charge). Identity aside, this is exactly
    the number of same-peptide pairs. Sequences are I->L normalised first, as the
    real generator does, so isoleucine/leucine variants collapse together.
    """
    norm = df.copy()
    norm['sequence'] = norm['sequence'].str.replace('I', 'L')
    per_charge = {}
    for charge in range(config.CHARGES[0], config.CHARGES[1] + 1):
        sub = norm[norm['charge'] == charge]
        sizes = sub.groupby('sequence', sort=False).size()
        per_charge[charge] = int(sizes.map(lambda n: comb(n, 2)).sum())
    return pd.Series(per_charge, name='pos_pairs').rename_axis('charge')


# -------------------------------------------------------------------
# 1-3. Metadata, grouping, replicates
# -------------------------------------------------------------------

def inspect_metadata() -> pd.DataFrame:
    _rule("1-3. Metadata, grouping by (sequence, charge), replicates")
    path = config.MASSIVEKB_METADATA_FILE
    df = pd.read_parquet(path)
    print(f"Loaded combined metadata: {path}")
    print(f"  columns          : {list(df.columns)}")
    print(f"  total spectra     : {len(df):,}")
    print(f"  unique sequences  : {df['sequence'].nunique():,}")

    groups = df.groupby(['sequence', 'charge'])
    sizes = groups.size()
    print(f"  (sequence,charge) groups           : {groups.ngroups:,}")
    print(f"  groups with replicate spectra (>1) : {(sizes > 1).sum():,}")
    print(f"  groups that are singletons  (==1)  : {(sizes == 1).sum():,}")

    print("\n  replicate-count distribution (spectra per group):")
    dist = sizes.value_counts().sort_index()
    for n, cnt in dist.items():
        print(f"    {n:>3} spectra/group : {cnt:>8,} groups")

    print("\n  spectra per charge:")
    for charge, cnt in df['charge'].value_counts().sort_index().items():
        print(f"    charge {charge}: {cnt:>10,} spectra")
    return df


# -------------------------------------------------------------------
# 4-5. Peptide-level split
# -------------------------------------------------------------------

def _overlap_report(train: pd.DataFrame, val: pd.DataFrame, label: str) -> None:
    ts, vs = set(train['sequence']), set(val['sequence'])
    tk = set(map(tuple, train[['sequence', 'charge']].values))
    vk = set(map(tuple, val[['sequence', 'charge']].values))
    print(f"  [{label}]")
    print(f"    train spectra {len(train):>9,} | val spectra {len(val):>9,}")
    print(f"    train sequences {len(ts):>7,} | val sequences {len(vs):>7,}")
    print(f"    sequences in BOTH        : {len(ts & vs):>7,}")
    print(f"    (sequence,charge) in BOTH: {len(tk & vk):>7,}")


def peptide_level_split(df: pd.DataFrame,
                        val_ratio: float = config.TEST_SPLIT_RATIO,
                        seed: int = config.SPLIT_RANDOM_STATE):
    """Split on *sequence* so no peptide appears in both train and val.

    All spectra (and all charges) of a given sequence go to the same side.
    """
    rng = np.random.default_rng(seed)
    seqs = df['sequence'].unique()
    rng.shuffle(seqs)
    n_val = int(round(len(seqs) * val_ratio))
    val_seqs = set(seqs[:n_val])
    is_val = df['sequence'].isin(val_seqs)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def inspect_split(df: pd.DataFrame):
    _rule("4-5. Train/val split — existing (row-level) vs. peptide-level")

    print("EXISTING split (prepare_data.py: train_test_split at the ROW level):")
    try:
        ex_train = pd.read_parquet(config.TRAIN_METADATA_FILE)
        ex_val = pd.read_parquet(config.TEST_METADATA_FILE)
        _overlap_report(ex_train, ex_val, "existing")
        print("    ^ peptides leak across train/val — the same sequence's "
              "replicates\n      land on both sides. This inflates val metrics.")
    except FileNotFoundError:
        print("  (existing split parquets not found — skipping)")

    print("\nPEPTIDE-LEVEL split (split on sequence — no peptide overlap):")
    pl_train, pl_val = peptide_level_split(df)
    _overlap_report(pl_train, pl_val, "peptide-level")
    return pl_train, pl_val


# -------------------------------------------------------------------
# 6-7. Pairs
# -------------------------------------------------------------------

def inspect_pairs(pl_train: pd.DataFrame, pl_val: pd.DataFrame) -> None:
    _rule("6-7. Positive & negative pairs")

    print("EXISTING pair files on disk (from the row-level split):")
    for label, meta in [("train", config.TRAIN_METADATA_FILE),
                        ("val (test)", config.TEST_METADATA_FILE)]:
        counts = _count_pairs_on_disk(meta)
        pos_tot = counts['pos'].dropna().sum()
        neg_tot = counts['neg'].dropna().sum()
        print(f"\n  [{label}]  ({os.path.basename(meta)})")
        print(counts.to_string())
        print(f"  TOTAL  pos {int(pos_tot):>12,}   neg {int(neg_tot):>12,}")

    print("\n" + "-" * 70)
    print("Positive pairs for the PEPTIDE-LEVEL split (computed directly,")
    print("no leakage — same-(sequence,charge) combinations C(n,2)):")
    for label, sub in [("train", pl_train), ("val", pl_val)]:
        pos = _positive_pairs_from_metadata(sub)
        print(f"\n  [{label}]")
        print(pos.to_string())
        print(f"  TOTAL positive pairs: {int(pos.sum()):>12,}")

    print("\n" + "-" * 70)
    print("Negative pairs: generation depends on precursor m/z windows and")
    print("theoretical-fragment overlap (metadata.generate_pairs_negative),")
    print("which is expensive to run here. To materialize them for the")
    print("peptide-level split, write the split parquets (--write-split) and run:")
    print("    from metadata import metadata")
    print("    metadata.generate_pairs_negative(<split>.parquet, config.CHARGES,")
    print("        config.PAIR_MZ_TOLERANCE,")
    print("        config.NEGATIVE_PAIR_FRAGMENT_TOLERANCE,")
    print("        config.NEGATIVE_PAIR_MATCHING_FRAGMENTS_THRESHOLD)")


# -------------------------------------------------------------------
# Optional: materialize the peptide-level split
# -------------------------------------------------------------------

def write_split(pl_train: pd.DataFrame, pl_val: pd.DataFrame) -> None:
    out_dir = Path(config.TRAIN_METADATA_FILE).parent
    train_p = out_dir / 'train_metadata_peptidelevel.parquet'
    val_p = out_dir / 'val_metadata_peptidelevel.parquet'
    pl_train.to_parquet(train_p, index=False)
    pl_val.to_parquet(val_p, index=False)
    print(f"\n[write-split] wrote {train_p} ({len(pl_train):,} rows)")
    print(f"[write-split] wrote {val_p} ({len(pl_val):,} rows)")


class _Tee:
    """Fan writes out to several file-like objects (e.g. terminal + log file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def _run_report(args) -> None:
    df = inspect_metadata()
    pl_train, pl_val = inspect_split(df)
    inspect_pairs(pl_train, pl_val)
    if args.write_split:
        write_split(pl_train, pl_val)
    print()


def main() -> None:
    default_log = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'inspect_spectra.log')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write-split', action='store_true',
                    help="materialize the peptide-level split parquets in data/")
    ap.add_argument('--log', default=default_log,
                    help=f"path to write the report log (default: {default_log})")
    ap.add_argument('--no-log', action='store_true',
                    help="print to the terminal only, don't write a log file")
    args = ap.parse_args()

    if args.no_log:
        _run_report(args)
        return

    # Tee stdout/stderr to the terminal and the log file for the whole report.
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with open(args.log, 'w') as log_file:
        sys.stdout = _Tee(orig_stdout, log_file)
        sys.stderr = _Tee(orig_stderr, log_file)
        try:
            print(f"inspect_spectra — {time.strftime('%Y-%m-%d %H:%M:%S')}")
            _run_report(args)
        finally:
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
    print(f"[log] report written to {args.log}")


if __name__ == "__main__":
    main()
