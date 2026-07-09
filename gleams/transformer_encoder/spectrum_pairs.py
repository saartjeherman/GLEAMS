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


def build_mgf_offset_index(
    mgf_path: str,
    index_path: Optional[str] = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Byte offset of every ``BEGIN IONS`` in the MGF, indexed by spectrum position.

    Position ``i`` — matching the enumerate order of ``mgf.read(use_index=False)``,
    and therefore the metadata ``scan`` column — begins at byte ``offsets[i]``.
    Built once by scanning the raw bytes for the block marker (no spectrum
    parsing, so it's I/O-bound rather than CPU-bound) and cached next to the MGF
    as ``<mgf>.beginions.npy``, letting later runs seek straight to the spectra
    they need instead of streaming the whole file.
    """
    if index_path is None:
        index_path = str(mgf_path) + '.beginions.npy'
    p = Path(index_path)
    if p.exists() and not overwrite:
        return np.load(p)

    marker = b'\nBEGIN IONS'
    keep = len(marker) - 1  # carry these bytes so a marker split across reads is caught
    offsets: List[int] = []
    file_size = os.path.getsize(mgf_path)
    chunk_size = 64 * 1024 * 1024

    with open(mgf_path, 'rb') as f:
        if f.read(len(b'BEGIN IONS')) == b'BEGIN IONS':
            offsets.append(0)  # first spectrum is at the very start (no leading '\n')
        f.seek(0)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                    desc='Indexing MGF', mininterval=2.0, smoothing=0.05)
        carry = b''
        file_pos = 0  # byte offset of the start of `chunk`
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf = carry + chunk
            base = file_pos - len(carry)
            i = buf.find(marker)
            while i != -1:
                offsets.append(base + i + 1)  # +1: point at 'B', past the '\n'
                i = buf.find(marker, i + 1)
            carry = buf[-keep:]
            file_pos += len(chunk)
            pbar.update(len(chunk))
        pbar.close()

    arr = np.asarray(offsets, dtype=np.int64)
    np.save(p, arr)
    print(f"[preencode] built MGF offset index: {len(arr):,} spectra -> {p}")
    return arr


def _parse_mgf_block(block: bytes) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Parse one ``BEGIN IONS ... END IONS`` block into (mz, intensity, pepmass, charge).

    Mirrors the field extraction of the streaming pyteomics path exactly: peak
    lines are ``m/z intensity`` pairs cast to float32, PEPMASS takes its first
    value, and CHARGE drops a trailing '+'. Produces byte-identical arrays to
    ``mgf.read`` for these fields.
    """
    mz_list: List[float] = []
    int_list: List[float] = []
    pepmass = 0.0
    charge = 0
    for raw in block.split(b'\n'):
        line = raw.strip()
        if not line:
            continue
        if line[0:1].isdigit():  # peak line: "mz intensity [extra...]"
            parts = line.split()
            if len(parts) >= 2:
                mz_list.append(float(parts[0]))
                int_list.append(float(parts[1]))
        elif line[:8] == b'PEPMASS=':
            tok = line[8:].split()
            if tok:
                pepmass = float(tok[0])
        elif line[:7] == b'CHARGE=':
            val = line[7:].strip().rstrip(b'+')
            try:
                charge = int(val)
            except ValueError:
                charge = 0
        # BEGIN IONS / END IONS / TITLE / RTINSECONDS / SCANS lines are ignored.
    return (
        np.asarray(mz_list, dtype=np.float32),
        np.asarray(int_list, dtype=np.float32),
        pepmass,
        charge,
    )


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

    # Fetch only the referenced spectra by seeking to their byte offsets, rather
    # than streaming and parsing the entire multi-hundred-GB MGF. The offset
    # index is built once and cached next to the MGF, so later runs skip that too.
    #
    # The bundled MGF has no SCANS= field and per-file `scan:N` values in the
    # TITLE repeat across source files, so they're not unique. The metadata's
    # `scan` column is the spectrum's enumerate position across the whole MGF,
    # which is exactly what the offset index is keyed on.
    offsets = build_mgf_offset_index(mgf_path)
    n_spectra = len(offsets)
    file_size = os.path.getsize(mgf_path)
    found = [0] * len(pending)

    positions = sorted(pos for pos in scan_to_slots if 0 <= pos < n_spectra)
    skipped = len(scan_to_slots) - len(positions)
    if skipped:
        print(f"[preencode] WARNING: {skipped} referenced position(s) fall "
              f"outside the {n_spectra:,} spectra in the MGF index; skipped.")
    print(f"[preencode] seeking {len(positions):,} referenced spectra "
          f"(of {n_spectra:,} in the MGF).")

    pbar = tqdm(total=len(positions), unit='spec', unit_scale=True,
                desc='Seeking spectra', mininterval=2.0, smoothing=0.05)
    # Ascending offset order keeps the seeks moving mostly forward.
    with open(mgf_path, 'rb') as f:
        for position in positions:
            start = int(offsets[position])
            end = (int(offsets[position + 1])
                   if position + 1 < n_spectra else file_size)
            f.seek(start)
            mz, it, pep, chg = _parse_mgf_block(f.read(end - start))
            for out_idx, row in scan_to_slots[position]:
                mz_buffers[out_idx][row] = mz
                int_buffers[out_idx][row] = it
                pepmasses[out_idx][row] = pep
                charges[out_idx][row] = chg
                found[out_idx] += 1
            pbar.update(1)
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
