"""Transformer side of the holdout comparison (run in the torch `base` env).

Embeds the validation spectra referenced by the shared selected pairs with a
trained transformer checkpoint, then writes per-pair Euclidean distances
(aligned to the selection order, NaN where a spectrum has no peaks) into the
comparison run folder for `compare_report.py` to score.

    python compare_transformer.py [--checkpoint CKPT.pt] [--run-dir DIR]

Without --run-dir a fresh timestamped run folder is created; pass the same
--run-dir to compare_cnn.py / compare_report.py (the orchestrator does this).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import config
import compare_common
from model import CustomSpectrumTransformerEncoder
from spectrum_pairs import _stack_spectra, preencode_mgf_to_npzs
from training import _compact_referenced_spectra

# Scoped checkpoint (overridable on the command line).
DEFAULT_CHECKPOINT = os.path.join(
    config.RESULTS_ROOT, '2026-05-19_22-44-29', 'best_model.pt')


def _load_model(checkpoint: str, device: torch.device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    hp = ck['hyperparameters']
    print(f"[transformer] checkpoint {checkpoint}\n"
          f"[transformer] epoch={ck.get('epoch')} val_loss={ck.get('val_loss')}\n"
          f"[transformer] hyperparameters={hp}")
    enc = CustomSpectrumTransformerEncoder(
        d_model=hp['dim_model'], nhead=hp['n_head'],
        dim_feedforward=hp['dim_feedforward'], n_layers=hp['n_layers'],
        dropout=hp['dropout'])
    enc.load_state_dict(ck['model_state_dict'])
    # source = "<run-folder>/<checkpoint-file>" so the report shows which model.
    source = os.path.join(os.path.basename(os.path.dirname(checkpoint)),
                          os.path.basename(checkpoint))
    meta = {'label': 'Transformer', 'source': source,
            'epoch': ck.get('epoch'), 'val_loss': ck.get('val_loss')}
    return enc.to(device).eval(), meta


def _embed_unique_rows(npz_path, model, device, max_peaks, batch_size=128):
    """Embed every referenced spectrum once. Returns (embeddings, valid_mask)."""
    with np.load(npz_path) as d:
        mz_flat, int_flat = d['mz_flat'], d['int_flat']
        offsets, pepmass, charge = d['offsets'], d['pepmass'], d['charge']
    n = len(pepmass)
    lengths = np.diff(offsets)
    valid = lengths > 0

    embeddings = None
    order = np.flatnonzero(valid)
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        specs = []
        for r in rows:
            a, b = int(offsets[r]), int(offsets[r + 1])
            mz, it = mz_flat[a:b], int_flat[a:b]
            if max_peaks is not None and len(mz) > max_peaks:
                top = np.argpartition(it, -max_peaks)[-max_peaks:]
                top = top[np.argsort(mz[top])]
                mz, it = mz[top], it[top]
            specs.append((mz, it, float(pepmass[r]), int(charge[r])))
        mz_pad, int_pad, pep, chg = _stack_spectra(specs)
        with torch.no_grad():
            emb_full, _ = model(mz_pad.to(device), int_pad.to(device),
                                pepmass=pep.to(device), charge=chg.to(device))
            emb = emb_full[:, 0, :].cpu().numpy()
        if embeddings is None:
            embeddings = np.full((n, emb.shape[1]), np.nan, dtype=np.float64)
        embeddings[rows] = emb
        print(f"\r[transformer] embedded {min(start + batch_size, len(order)):,}"
              f"/{len(order):,} unique spectra", end='', flush=True)
    print()
    if embeddings is None:
        embeddings = np.full((n, 1), np.nan)
    return embeddings, valid


def _pair_distances(embeddings, valid, pairs):
    """Euclidean distance per pair; NaN if either endpoint is invalid."""
    a, b = pairs[:, 0], pairs[:, 1]
    ok = valid[a] & valid[b]
    dist = np.full(len(pairs), np.nan, dtype=np.float64)
    dist[ok] = np.linalg.norm(embeddings[a[ok]] - embeddings[b[ok]], axis=1)
    return dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='Transformer .pt checkpoint to evaluate.')
    parser.add_argument('--run-dir', default=None,
                        help='Comparison run folder (default: a fresh '
                             'timestamped one under COMPARISONS_ROOT).')
    # Positional checkpoint kept for backwards compatibility.
    parser.add_argument('checkpoint_pos', nargs='?', default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    checkpoint = args.checkpoint_pos or args.checkpoint

    run_dir = compare_common.resolve_run_dir(args.run_dir)
    tmp_npz = os.path.join(run_dir, 'transformer_val_referenced.npz')
    out_npz = os.path.join(run_dir, 'transformer_distances.npz')
    print(f"[transformer] run dir: {run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[transformer] device={device}")

    pos_sel, neg_sel = compare_common.select_pairs(
        config.VAL_POS_PAIRS_FILE, config.VAL_NEG_PAIRS_FILE)
    print(f"[transformer] selected {len(pos_sel):,} pos, {len(neg_sel):,} neg pairs")

    val_meta = pd.read_parquet(config.VAL_METADATA_FILE)
    # Compact to referenced spectra and remap pair indices onto the compact rows
    # (same helper the trainer uses), so we only seek/encode what we need.
    compact_meta, (pos_c, neg_c) = _compact_referenced_spectra(
        val_meta, [pos_sel, neg_sel])
    print(f"[transformer] {len(compact_meta):,} unique referenced spectra")

    preencode_mgf_to_npzs(config.MGF_PATH, [(compact_meta, tmp_npz)],
                          overwrite=True)

    model, meta = _load_model(checkpoint, device)
    embeddings, valid = _embed_unique_rows(
        tmp_npz, model, device, config.MAX_PEAKS)

    pos_dist = _pair_distances(embeddings, valid, pos_c)
    neg_dist = _pair_distances(embeddings, valid, neg_c)
    compare_common.save_distances(
        out_npz, pos_dist, neg_dist,
        compare_common.DEFAULT_CAP, compare_common.DEFAULT_SEED, meta=meta)


if __name__ == '__main__':
    main()
