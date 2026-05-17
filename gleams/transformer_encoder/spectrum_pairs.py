"""MGF → packed-tensor pipeline for contrastive pair training.

Three pieces:

1. `preencode_mgf_to_npz` walks an MGF once and packs the spectra that appear
   in `metadata['scan']` into a single .npz, aligned to metadata row order.
   Variable-length peaks are stored CSR-style (flat array + offsets).

2. `SpectrumPairDataset` mmaps that .npz and indexes pairs of metadata rows.

3. `collate_pairs` pads each batch to the longest spectrum and stacks scalars.

The Dataset returns ((mz, intensity, pepmass, charge), (mz, intensity, pepmass,
charge), label) tuples, matching the unpacking already in `train_model`.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pyteomics import mgf
from torch.utils.data import Dataset
from tqdm import tqdm


_SCAN_RE = re.compile(r'scan[=:](\d+)', re.IGNORECASE)


def _extract_scan(params: dict) -> Optional[int]:
    scans = params.get('scans')
    if scans is not None:
        try:
            return int(scans)
        except (TypeError, ValueError):
            pass
    match = _SCAN_RE.search(params.get('title', ''))
    return int(match.group(1)) if match else None


def _extract_pepmass(params: dict) -> float:
    pep = params.get('pepmass')
    if isinstance(pep, (list, tuple)):
        pep = pep[0]
    return float(pep) if pep is not None else 0.0


def _extract_charge(params: dict) -> int:
    chg = params.get('charge')
    if isinstance(chg, (list, tuple)):
        chg = chg[0]
    if chg is None:
        return 0
    return int(str(chg).rstrip('+'))


def preencode_mgf_to_npzs(
    mgf_path: str,
    outputs: Sequence[Tuple[pd.DataFrame, str]],
    overwrite: bool = False,
) -> None:
    """Walk `mgf_path` once and pack matching spectra into one .npz per (metadata, out_path).

    A scan that appears in multiple metadata files lights up multiple output
    slots in the same pass. Outputs whose .npz already exists are skipped
    (unless `overwrite=True`); if every output exists the MGF isn't opened.

    Per-output .npz keys:
      mz_flat, int_flat : float32, concatenated peak arrays
      offsets           : int64, length N+1; row i lives at mz_flat[offsets[i]:offsets[i+1]]
      pepmass           : float32, shape (N,)
      charge            : int16,   shape (N,)
    """
    pending: List[Tuple[pd.DataFrame, Path]] = []
    for metadata, out_path in outputs:
        p = Path(out_path)
        if p.exists() and not overwrite:
            print(f"[preencode] {p} exists, skipping.")
        else:
            pending.append((metadata, p))
    if not pending:
        return

    sizes = [len(md) for md, _ in pending]
    mz_buffers: List[List[Optional[np.ndarray]]] = [[None] * n for n in sizes]
    int_buffers: List[List[Optional[np.ndarray]]] = [[None] * n for n in sizes]
    pepmasses = [np.zeros(n, dtype=np.float32) for n in sizes]
    charges = [np.zeros(n, dtype=np.int16) for n in sizes]

    # Combined scan -> [(output_idx, row_idx), ...]
    scan_to_slots: Dict[int, List[Tuple[int, int]]] = {}
    dupes = 0
    for out_idx, (metadata, _) in enumerate(pending):
        seen_in_this_metadata = set()
        for row, scan in enumerate(metadata['scan'].astype(np.int64).values):
            scan = int(scan)
            if scan in seen_in_this_metadata:
                dupes += 1
                continue
            seen_in_this_metadata.add(scan)
            scan_to_slots.setdefault(scan, []).append((out_idx, row))
    if dupes:
        print(f"[preencode] WARNING: {dupes} duplicate scan(s) within a single "
              f"metadata file; later rows are ignored.")

    # Highest enumerate position any metadata row asks for — stop reading the
    # MGF after we pass it, since nothing further can match.
    max_scan_needed = max(scan_to_slots) if scan_to_slots else -1
    print(f"[preencode] highest scan referenced: {max_scan_needed:,}; "
          f"will stop reading MGF after that.")

    file_size = os.path.getsize(mgf_path)
    pbar = tqdm(
        total=file_size, unit='B', unit_scale=True,
        desc='Parsing MGF', mininterval=2.0, smoothing=0.05,
    )
    found = [0] * len(pending)

    # The bundled MGF has no SCANS= field and per-file `scan:N` values in the
    # TITLE repeat across source files, so they're not unique. The metadata's
    # `scan` column is the spectrum's enumerate position across the whole MGF.
    # `use_index=False` matches the call used to generate the metadata, so
    # iteration order is guaranteed identical.
    with mgf.read(str(mgf_path), use_index=False) as reader:
        src = getattr(reader, '_source', None)
        last_pos = 0
        for position, spec in enumerate(reader):
            if position > max_scan_needed:
                break  # no metadata row references anything past here

            if src is not None:
                try:
                    cur_pos = src.tell()
                    if cur_pos > last_pos:
                        pbar.update(cur_pos - last_pos)
                        last_pos = cur_pos
                except (OSError, AttributeError):
                    src = None  # stop trying; bar will just stall

            slots = scan_to_slots.get(position)
            if not slots:
                continue

            params = spec.get('params', {})
            mz = np.asarray(spec['m/z array'], dtype=np.float32)
            it = np.asarray(spec['intensity array'], dtype=np.float32)
            pep = _extract_pepmass(params)
            chg = _extract_charge(params)
            for out_idx, row in slots:
                mz_buffers[out_idx][row] = mz
                int_buffers[out_idx][row] = it
                pepmasses[out_idx][row] = pep
                charges[out_idx][row] = chg
                found[out_idx] += 1

    # Make sure the bar reaches 100% even if `tell()` lagged behind.
    if pbar.total is not None and pbar.n < pbar.total:
        pbar.update(pbar.total - pbar.n)
    pbar.close()

    for out_idx, (_metadata, out_path) in enumerate(pending):
        n = sizes[out_idx]
        print(f"[preencode] {out_path}: matched {found[out_idx]:,}/{n:,} rows "
              f"({n - found[out_idx]:,} missing).")

        lengths = np.array(
            [len(b) if b is not None else 0 for b in mz_buffers[out_idx]],
            dtype=np.int64,
        )
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])

        if offsets[-1] > 0:
            mz_flat = np.concatenate(
                [b for b in mz_buffers[out_idx] if b is not None]
            ).astype(np.float32)
            int_flat = np.concatenate(
                [b for b in int_buffers[out_idx] if b is not None]
            ).astype(np.float32)
        else:
            mz_flat = np.zeros(0, dtype=np.float32)
            int_flat = np.zeros(0, dtype=np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            mz_flat=mz_flat,
            int_flat=int_flat,
            offsets=offsets,
            pepmass=pepmasses[out_idx],
            charge=charges[out_idx],
        )
        print(f"[preencode] wrote {out_path} ({mz_flat.size:,} peaks).")


def preencode_mgf_to_npz(
    mgf_path: str,
    metadata: pd.DataFrame,
    out_path: str,
    overwrite: bool = False,
) -> None:
    """Single-output wrapper around `preencode_mgf_to_npzs`."""
    preencode_mgf_to_npzs(mgf_path, [(metadata, out_path)], overwrite=overwrite)


class SpectrumPairDataset(Dataset):
    """Indexes contrastive pairs into a pre-encoded .npz.

    `pos_pairs` and `neg_pairs` are (N, 2) int arrays of metadata row indices.
    """

    def __init__(
        self,
        npz_path: str,
        pos_pairs: np.ndarray,
        neg_pairs: np.ndarray,
        max_peaks: Optional[int] = None,
    ):
        with np.load(npz_path) as data:
            self.mz_flat = data['mz_flat']
            self.int_flat = data['int_flat']
            self.offsets = data['offsets']
            self.pepmass = data['pepmass']
            self.charge = data['charge']
        # If set, each spectrum is truncated to its `max_peaks` most intense peaks
        # at __getitem__ time. This bounds the per-batch padded length so one
        # outlier spectrum can't blow up the whole batch (attention is O(L^2)).
        self.max_peaks = max_peaks

        pairs = np.vstack([pos_pairs, neg_pairs]).astype(np.int64)
        labels = np.concatenate([
            np.ones(len(pos_pairs), dtype=np.float32),
            np.zeros(len(neg_pairs), dtype=np.float32),
        ])

        lengths = np.diff(self.offsets)
        valid_row = lengths > 0
        keep = valid_row[pairs[:, 0]] & valid_row[pairs[:, 1]]
        dropped = int((~keep).sum())
        if dropped:
            print(f"[dataset] dropping {dropped:,} / {len(pairs):,} pairs "
                  f"referencing rows with no spectrum.")
        self.pairs = pairs[keep]
        self.labels = labels[keep]

    def __len__(self) -> int:
        return len(self.pairs)

    def _spectrum(self, row: int) -> Tuple[np.ndarray, np.ndarray, float, int]:
        start = int(self.offsets[row])
        end = int(self.offsets[row + 1])
        mz = self.mz_flat[start:end]
        it = self.int_flat[start:end]
        if self.max_peaks is not None and len(mz) > self.max_peaks:
            # Keep the `max_peaks` highest-intensity peaks, then resort by m/z
            # so downstream code that assumes sorted m/z stays valid.
            top_idx = np.argpartition(it, -self.max_peaks)[-self.max_peaks:]
            top_idx = top_idx[np.argsort(mz[top_idx])]
            mz, it = mz[top_idx], it[top_idx]
        return (
            mz,
            it,
            float(self.pepmass[row]),
            int(self.charge[row]),
        )

    def __getitem__(self, idx: int):
        r1, r2 = self.pairs[idx]
        return (
            self._spectrum(int(r1)),
            self._spectrum(int(r2)),
            float(self.labels[idx]),
        )


def _stack_spectra(spec_list):
    mzs, ints, peps, chgs = zip(*spec_list)
    max_len = max(len(m) for m in mzs)
    batch = len(mzs)
    mz_pad = torch.zeros(batch, max_len, dtype=torch.float32)
    int_pad = torch.zeros(batch, max_len, dtype=torch.float32)
    for i, (m, it) in enumerate(zip(mzs, ints)):
        mz_pad[i, :len(m)] = torch.from_numpy(np.ascontiguousarray(m))
        int_pad[i, :len(it)] = torch.from_numpy(np.ascontiguousarray(it))
    # depthcharge's FloatEncoder requires shape (batch, n_values); give it
    # (batch, 1) so the global_token_hook can squeeze(1) back to (batch, d_model).
    pepmass = torch.tensor(peps, dtype=torch.float32).unsqueeze(-1)
    charge = torch.tensor(chgs, dtype=torch.float32).unsqueeze(-1)
    return mz_pad, int_pad, pepmass, charge


def collate_pairs(batch):
    spec1_list, spec2_list, labels = zip(*batch)
    return (
        _stack_spectra(spec1_list),
        _stack_spectra(spec2_list),
        torch.tensor(labels, dtype=torch.float32),
    )
