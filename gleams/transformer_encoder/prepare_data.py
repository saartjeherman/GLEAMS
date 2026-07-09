"""Three-stage data preparation pipeline:

  1. `build_metadata`  → walk the MGF once, collect per-spectrum metadata
                         (scan, rtinseconds, sequence, mz, charge), keeping
                         MIN_REPLICATES..MAX_REPLICATES spectra per (sequence,
                         charge) group, save as MASSIVEKB_METADATA_FILE.
  2. `split_metadata`  → train_test_split → TRAIN_METADATA_FILE + TEST_METADATA_FILE.
  3. `prepare_pairs`   → run gleams' generate_pairs_positive/negative for both
                         splits, writing the .npy pair files train.py reads.

Each step skips itself if its output already exists. Flip `OVERWRITE = True`
at the top to force regeneration of everything.

To process the full MGF, set `SPECTRA_TARGET = None` in config.py.

Run from this folder:
    python prepare_data.py
"""

import os
import sys
from pathlib import Path

# Append gleams/ to the path (not insert-at-0) so we can pull in the
# metadata.generate_pairs_* helpers while this folder's local config.py still
# wins over the top-level gleams/config.py.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pyteomics import mgf
from tqdm import tqdm

import config
from metadata import metadata

OVERWRITE = False  # set True to force all three steps to rerun


# -------------------------------------------------------------------
# Stage 1: build combined metadata from MGF
# -------------------------------------------------------------------

_METADATA_COLS = ['scan', 'rtinseconds', 'sequence', 'pepmass', 'charge']


def _normalize_spectrum(params: dict, scan_idx: int) -> dict:
    """Extract the fields we care about from a pyteomics MGF params dict."""
    pepmass = params.get('pepmass')
    if isinstance(pepmass, (tuple, list)):
        pepmass = pepmass[0]

    charge = params.get('charge')
    if isinstance(charge, (list, tuple)) and len(charge) > 0:
        charge = charge[0]
    if isinstance(charge, str):
        charge = charge.rstrip('+')
    try:
        charge = int(charge)
    except (TypeError, ValueError):
        charge = None

    return {
        'scan': scan_idx,       # MGF enumerate position — matches the encoder
        'rtinseconds': params.get('rtinseconds'),
        'sequence': params.get('seq'),
        'pepmass': pepmass,
        'charge': charge,
    }


def build_metadata() -> None:
    """Walk the MGF, keep MIN_REPLICATES..MAX_REPLICATES spectra per
    (sequence, charge) group, save as MASSIVEKB_METADATA_FILE."""
    out_parquet = Path(config.MASSIVEKB_METADATA_FILE)
    if out_parquet.exists() and not OVERWRITE:
        print(f"[build_metadata] {out_parquet} exists, skipping.")
        return

    csv_tmp = out_parquet.with_suffix('.csv')
    if csv_tmp.exists():
        csv_tmp.unlink()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    replicate_counts: dict = {}
    buffers: dict = {}
    rows_to_flush: list = []
    committed_spectra = 0  # spectra in groups that have reached MIN_REPLICATES

    def flush(rows: list) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows, columns=_METADATA_COLS)
        write_header = not csv_tmp.exists()
        df.to_csv(csv_tmp, mode='a', index=False, header=write_header)
        rows.clear()

    file_size = os.path.getsize(config.MGF_PATH)
    pbar = tqdm(
        total=file_size, unit='B', unit_scale=True,
        desc='Reading MGF', mininterval=2.0, smoothing=0.05,
    )

    try:
        with mgf.read(config.MGF_PATH, use_index=False) as reader:
            src = getattr(reader, '_source', None)
            last_pos = 0
            for scan_idx, spectrum in enumerate(reader):
                if src is not None:
                    try:
                        cur = src.tell()
                        if cur > last_pos:
                            pbar.update(cur - last_pos)
                            last_pos = cur
                    except (OSError, AttributeError):
                        src = None

                row = _normalize_spectrum(spectrum.get('params', {}), scan_idx)
                if row['sequence'] is None or row['charge'] is None:
                    continue

                key = (row['sequence'], row['charge'])
                count = replicate_counts.get(key, 0)
                if count >= config.MAX_REPLICATES:
                    continue  # group already full (and already flushed)

                count += 1
                replicate_counts[key] = count
                buffers.setdefault(key, []).append(row)

                # Count spectra in groups that have crossed MIN_REPLICATES.
                # Crossing MIN commits all MIN buffered rows at once; every
                # further replicate (up to MAX) commits one more.
                if count == config.MIN_REPLICATES:
                    committed_spectra += config.MIN_REPLICATES
                elif count > config.MIN_REPLICATES:
                    committed_spectra += 1

                # A group that hits the cap is complete: flush it and free the
                # buffer so memory doesn't grow with popular peptides.
                if count == config.MAX_REPLICATES:
                    rows_to_flush.extend(buffers.pop(key))
                    if len(rows_to_flush) >= config.FLUSH_EVERY:
                        flush(rows_to_flush)

                if (config.SPECTRA_TARGET is not None
                        and committed_spectra >= config.SPECTRA_TARGET):
                    break
    finally:
        # Flush every still-buffered group that reached MIN_REPLICATES; groups
        # below MIN (e.g. singletons) are dropped as a quality filter.
        for key, rows in buffers.items():
            if replicate_counts[key] >= config.MIN_REPLICATES:
                rows_to_flush.extend(rows)
        flush(rows_to_flush)
        if pbar.total is not None and pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

    n_groups_kept = sum(1 for c in replicate_counts.values()
                        if c >= config.MIN_REPLICATES)
    print(f"[build_metadata] kept {n_groups_kept:,} (sequence, charge) groups "
          f"with {config.MIN_REPLICATES}-{config.MAX_REPLICATES} replicates each "
          f"({committed_spectra:,} spectra).")

    df = pd.read_csv(csv_tmp)
    df.rename(columns={'pepmass': 'mz'}, inplace=True)
    df.to_parquet(out_parquet, index=False)
    csv_tmp.unlink()
    print(f"[build_metadata] wrote {out_parquet} "
          f"({len(df):,} rows; scan range {df['scan'].min()}-{df['scan'].max()}).")


# -------------------------------------------------------------------
# Stage 2: train/test split
# -------------------------------------------------------------------

def split_metadata() -> None:
    """Split the combined metadata into train + val + test parquets on the
    PEPTIDE level (80/10/10 by peptide, like the original GLEAMS model): every
    spectrum of a given sequence (all charges, all replicates) goes to exactly
    one split, so no peptide appears in more than one. This prevents the leakage
    where a peptide memorized in training reappears in val/test and inflates
    those metrics. Train fits the model, val drives early stopping, test is held
    out for final evaluation."""
    train_p = Path(config.TRAIN_METADATA_FILE)
    val_p = Path(config.VAL_METADATA_FILE)
    test_p = Path(config.TEST_METADATA_FILE)
    if train_p.exists() and val_p.exists() and test_p.exists() and not OVERWRITE:
        print(f"[split_metadata] {train_p.name}, {val_p.name} and {test_p.name} "
              f"exist, skipping.")
        return

    print(f"[split_metadata] reading {config.MASSIVEKB_METADATA_FILE} ...")
    df = pd.read_parquet(config.MASSIVEKB_METADATA_FILE)

    # Partition on the I->L-normalized sequence — the same identity the pair
    # generators use (metadata.generate_pairs_positive maps I->L before grouping).
    # Splitting on the raw sequence would let a peptide and its I/L twin land in
    # different splits and effectively leak. Partitioning by *peptide* means the
    # spectra fractions land near 80/10/10 (replicate counts are similar across
    # peptides), not exactly.
    groups = df['sequence'].str.replace('I', 'L')
    peptides = groups.unique()
    rng = np.random.default_rng(config.SPLIT_RANDOM_STATE)
    rng.shuffle(peptides)

    n = len(peptides)
    n_test = round(n * config.TEST_SPLIT_RATIO)
    n_val = round(n * config.VAL_SPLIT_RATIO)
    test_peptides = set(peptides[:n_test])
    val_peptides = set(peptides[n_test:n_test + n_val])
    # Train = every peptide not assigned to val or test.

    is_test = groups.isin(test_peptides)
    is_val = groups.isin(val_peptides)
    test_df = df[is_test].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    train_df = df[~(is_test | is_val)].reset_index(drop=True)

    # Guard against leakage / lost rows: splits must be disjoint and cover all.
    assert not (is_test & is_val).any(), "val/test peptide overlap"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "rows lost in split"

    train_p.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_p, index=False)
    val_df.to_parquet(val_p, index=False)
    test_df.to_parquet(test_p, index=False)
    print(f"[split_metadata] peptide-level {int((1-config.VAL_SPLIT_RATIO-config.TEST_SPLIT_RATIO)*100)}"
          f"/{int(config.VAL_SPLIT_RATIO*100)}/{int(config.TEST_SPLIT_RATIO*100)} "
          f"split by peptide (no shared sequences).")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[split_metadata] {name:5s}: {len(d):>12,} rows, "
              f"{d['sequence'].nunique():>10,} sequences "
              f"({len(d) / len(df) * 100:4.1f}% of spectra)")


# -------------------------------------------------------------------
# Stage 3: positive + negative pair generation
# -------------------------------------------------------------------

def _pairs_exist_for(parquet: str) -> bool:
    """True iff every charge in config.CHARGES already has both pair files on disk."""
    base = parquet.replace('.parquet', '')
    for charge in range(config.CHARGES[0], config.CHARGES[1] + 1):
        if not (Path(f'{base}_pairs_pos_{charge}.npy').exists()
                and Path(f'{base}_pairs_neg_{charge}.npy').exists()):
            return False
    return True


def prepare_pairs() -> None:
    """Run generate_pairs_positive/negative for the train, val and test parquets."""
    for label, parquet in [
        ("train", config.TRAIN_METADATA_FILE),
        ("val",   config.VAL_METADATA_FILE),
        ("test",  config.TEST_METADATA_FILE),
    ]:
        if _pairs_exist_for(parquet) and not OVERWRITE:
            print(f"[prepare_pairs] {label}: pair .npy files exist, skipping.")
            continue

        print(f"\n[prepare_pairs] {label}: generating positive pairs from {parquet} ...")
        metadata.generate_pairs_positive(parquet, config.CHARGES)

        print(f"[prepare_pairs] {label}: generating negative pairs from {parquet} ...")
        metadata.generate_pairs_negative(
            parquet,
            config.CHARGES,
            config.PAIR_MZ_TOLERANCE,
            config.NEGATIVE_PAIR_FRAGMENT_TOLERANCE,
            config.NEGATIVE_PAIR_MATCHING_FRAGMENTS_THRESHOLD,
        )


if __name__ == "__main__":
    build_metadata()
    split_metadata()
    prepare_pairs()
    print("\n✓ Data preparation complete.")
