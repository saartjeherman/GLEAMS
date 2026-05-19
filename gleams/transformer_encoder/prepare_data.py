"""Three-stage data preparation pipeline:

  1. `build_metadata`  → walk the MGF once, collect per-spectrum metadata
                         (scan, rtinseconds, sequence, mz, charge) for peptides
                         that hit N_REPLICATES, save as MASSIVEKB_METADATA_FILE.
  2. `split_metadata`  → train_test_split → TRAIN_METADATA_FILE + TEST_METADATA_FILE.
  3. `prepare_pairs`   → run gleams' generate_pairs_positive/negative for both
                         splits, writing the .npy pair files train.py reads.

Each step skips itself if its output already exists. Flip `OVERWRITE = True`
at the top to force regeneration of everything.

To process the full MGF, set `N_PEPTIDES_TARGET = None` in config.py.

Run from this folder:
    python prepare_data.py
"""

import os
import sys
from pathlib import Path

# Make gleams/ importable so we can pull in the metadata.generate_pairs_* helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pyteomics import mgf
from sklearn.model_selection import train_test_split
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
    """Walk the MGF, keep up to N_REPLICATES spectra per (sequence, charge),
    save as MASSIVEKB_METADATA_FILE."""
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
    completed_peptides = 0

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
                if replicate_counts.get(key, 0) >= config.N_REPLICATES:
                    continue

                replicate_counts[key] = replicate_counts.get(key, 0) + 1
                buffers.setdefault(key, []).append(row)

                if replicate_counts[key] == config.N_REPLICATES:
                    rows_to_flush.extend(buffers.pop(key))
                    completed_peptides += 1

                    if completed_peptides % config.FLUSH_EVERY == 0:
                        flush(rows_to_flush)

                    if (config.N_PEPTIDES_TARGET is not None
                            and completed_peptides >= config.N_PEPTIDES_TARGET):
                        break
    finally:
        flush(rows_to_flush)
        if pbar.total is not None and pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

    print(f"[build_metadata] {completed_peptides:,} complete peptides "
          f"x {config.N_REPLICATES} replicates.")

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
    """Split the combined metadata into train + test parquets."""
    train_p = Path(config.TRAIN_METADATA_FILE)
    test_p = Path(config.TEST_METADATA_FILE)
    if train_p.exists() and test_p.exists() and not OVERWRITE:
        print(f"[split_metadata] {train_p.name} and {test_p.name} exist, skipping.")
        return

    print(f"[split_metadata] reading {config.MASSIVEKB_METADATA_FILE} ...")
    df = pd.read_parquet(config.MASSIVEKB_METADATA_FILE)
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SPLIT_RATIO,
        random_state=config.TEST_SPLIT_RANDOM_STATE,
    )

    train_p.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_p, index=False)
    test_df.to_parquet(test_p, index=False)
    print(f"[split_metadata] train: {len(train_df):,} rows "
          f"(scan range {train_df['scan'].min()}-{train_df['scan'].max()})")
    print(f"[split_metadata] test:  {len(test_df):,} rows "
          f"(scan range {test_df['scan'].min()}-{test_df['scan'].max()})")


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
    """Run generate_pairs_positive/negative for the train and test parquets."""
    for label, parquet in [
        ("train", config.TRAIN_METADATA_FILE),
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
