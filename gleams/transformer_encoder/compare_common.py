"""Shared plumbing for the CNN-vs-transformer holdout comparison.

Pure NumPy + matplotlib so it imports cleanly in *both* the torch env (base,
transformer) and the TensorFlow env (gleams, CNN). It must NOT import `config`
(the two packages each ship one and they collide), torch, or tensorflow.

The comparison protocol
-----------------------
1. Each comparison runs in its own timestamped folder under COMPARISONS_ROOT
   (mirroring the `results/<timestamp>/` layout). All three steps share that one
   folder — the orchestrator mints it and passes it via `--run-dir`.
2. `select_pairs` deterministically subsamples the same positive/negative
   validation pairs in both evaluators (same file + cap + seed => identical
   arrays, in identical order).
3. Each evaluator embeds the referenced spectra with its own model and writes a
   `.npz` of per-pair Euclidean distances *aligned to the selected pair order*
   (NaN where a spectrum was dropped), plus a `meta` record identifying the
   model/checkpoint used.
4. `build_report` loads every model's `.npz`, keeps only pairs finite for ALL
   models, scores each on that common subset, and writes `comparison.csv` /
   `comparison.png` — both stamped with which model/checkpoint produced them.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np


COMPARISONS_ROOT = '/home/saartje/Desktop/Research/GLEAMS/comparisons'

DEFAULT_CAP = 25000   # pairs per class (positive / negative) to evaluate
DEFAULT_SEED = 0      # subsample seed — identical selection across evaluators


# ---------------------------------------------------------------------------
# Run directories (one timestamped folder per comparison, like results/)
# ---------------------------------------------------------------------------

def new_run_dir(label: Optional[str] = None) -> str:
    """Create and return a fresh `COMPARISONS_ROOT/<timestamp>[_label]/` folder."""
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f'{stamp}_{label}' if label else stamp
    run_dir = os.path.join(COMPARISONS_ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def latest_run_dir() -> str:
    """Most recently created run folder under COMPARISONS_ROOT (for the report)."""
    if not os.path.isdir(COMPARISONS_ROOT):
        raise SystemExit(f"No comparisons yet under {COMPARISONS_ROOT}")
    runs = [os.path.join(COMPARISONS_ROOT, d)
            for d in os.listdir(COMPARISONS_ROOT)
            if os.path.isdir(os.path.join(COMPARISONS_ROOT, d))]
    if not runs:
        raise SystemExit(f"No comparison run folders under {COMPARISONS_ROOT}")
    return max(runs, key=os.path.getmtime)


def resolve_run_dir(arg: Optional[str], make_new: bool = True) -> str:
    """Pick the run dir: explicit `arg` wins; else a fresh one (or the latest)."""
    if arg:
        os.makedirs(arg, exist_ok=True)
        return arg
    return new_run_dir() if make_new else latest_run_dir()


# ---------------------------------------------------------------------------
# Pair selection + distance I/O
# ---------------------------------------------------------------------------

def select_pairs(pos_file: str, neg_file: str,
                 cap: Optional[int] = DEFAULT_CAP,
                 seed: int = DEFAULT_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically pick the pairs both models will score.

    Returns (pos_pairs, neg_pairs) as int64 (N, 2) arrays of metadata row
    indices. With `cap` set, each class is subsampled without replacement to
    `cap` rows using `seed`; the same (file, cap, seed) always yields the same
    rows in the same order, so both evaluators align pair-for-pair.
    """
    pos = np.load(pos_file).astype(np.int64)
    neg = np.load(neg_file).astype(np.int64)
    rng = np.random.default_rng(seed)

    def _take(pairs: np.ndarray) -> np.ndarray:
        if cap is None or len(pairs) <= cap:
            return pairs
        idx = np.sort(rng.choice(len(pairs), cap, replace=False))
        return pairs[idx]

    pos_sel = _take(pos)
    neg_sel = _take(neg)
    return pos_sel, neg_sel


def save_distances(out_path: str, pos_dist: np.ndarray, neg_dist: np.ndarray,
                   cap: Optional[int], seed: int,
                   meta: Optional[Dict] = None) -> None:
    """Write per-pair distances (NaN = dropped) + selection provenance + model meta.

    `meta` identifies the model that produced these distances (e.g. label,
    checkpoint path, epoch, val_loss); it is echoed into the report's CSV/PNG.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        pos_dist=np.asarray(pos_dist, dtype=np.float64),
        neg_dist=np.asarray(neg_dist, dtype=np.float64),
        cap=np.array([-1 if cap is None else cap]),
        seed=np.array([seed]),
        meta_json=np.array(json.dumps(meta or {})),
    )
    print(f"[compare] wrote {out_path}: "
          f"{np.isfinite(pos_dist).sum():,}/{len(pos_dist):,} pos, "
          f"{np.isfinite(neg_dist).sum():,}/{len(neg_dist):,} neg finite.")


def load_distances(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return d['pos_dist'], d['neg_dist']


def load_meta(path: str) -> Dict:
    with np.load(path) as d:
        if 'meta_json' in d:
            return json.loads(str(d['meta_json'].item()))
    return {}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(distance_files: Dict[str, str], out_dir: str,
                 fdr_target: float = 0.01) -> Dict[str, dict]:
    """Score every model on the pairs finite for ALL models; write CSV + plot.

    `distance_files` maps a fallback model name -> its saved `.npz`. The display
    label and source (checkpoint) are taken from each file's stored `meta` when
    present. All models must share the same selection (cap/seed) so their arrays
    line up.
    """
    import csv

    from metrics import ranking_metrics  # local import: pure NumPy

    keys = list(distance_files)
    pos = {k: load_distances(p)[0] for k, p in distance_files.items()}
    neg = {k: load_distances(p)[1] for k, p in distance_files.items()}
    meta = {k: load_meta(p) for k, p in distance_files.items()}
    label = {k: meta[k].get('label', k) for k in keys}
    source = {k: _source_str(meta[k]) for k in keys}

    lengths_pos = {len(v) for v in pos.values()}
    lengths_neg = {len(v) for v in neg.values()}
    if len(lengths_pos) != 1 or len(lengths_neg) != 1:
        raise ValueError(f"Distance arrays differ in length across models "
                         f"(pos={lengths_pos}, neg={lengths_neg}); regenerate "
                         f"with the same cap/seed.")

    # Common mask: a pair counts only if every model produced a finite distance.
    pos_mask = np.all([np.isfinite(pos[k]) for k in keys], axis=0)
    neg_mask = np.all([np.isfinite(neg[k]) for k in keys], axis=0)
    print(f"[compare] common evaluable pairs: {pos_mask.sum():,} positive, "
          f"{neg_mask.sum():,} negative "
          f"(of {len(pos_mask):,}/{len(neg_mask):,} selected).")

    results = {k: ranking_metrics(pos[k][pos_mask], neg[k][neg_mask], fdr_target)
               for k in keys}

    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(out_dir))
    csv_path = os.path.join(out_dir, 'comparison.csv')
    cols = ['auc', 'fnr_at_fdr', 'fdr_threshold', 'fdr_target', 'eer',
            'n_pos', 'n_neg']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'model', 'source'] + cols)
        for k in keys:
            w.writerow([run_name, label[k], source[k]]
                       + [results[k][c] for c in cols])
    print(f"[compare] wrote {csv_path}")

    _print_table(results, keys, label, source, cols)
    _plot_comparison(
        {k: (pos[k][pos_mask], neg[k][neg_mask]) for k in keys},
        results, label, source, run_name,
        os.path.join(out_dir, 'comparison.png'))
    return results


def _source_str(meta: Dict) -> str:
    """One-line description of the model/checkpoint from its meta record."""
    src = meta.get('source', '')
    extra = []
    if meta.get('epoch') is not None:
        extra.append(f"ep{meta['epoch']}")
    if meta.get('val_loss') is not None:
        extra.append(f"val_loss={meta['val_loss']:.4f}")
    return f"{src} ({', '.join(extra)})" if extra else src


def _print_table(results, keys, label, source, cols) -> None:
    print("\n=== Holdout separation comparison (common pairs) ===")
    header = f"{'metric':<16}" + "".join(f"{label[k]:>22}" for k in keys)
    print(header)
    print("-" * len(header))
    for c in cols:
        row = f"{c:<16}"
        for k in keys:
            v = results[k][c]
            row += f"{v:>22.4f}" if isinstance(v, float) else f"{v:>22}"
        print(row)
    print("\nsource:")
    for k in keys:
        print(f"  {label[k]}: {source[k]}")
    print("\nHigher AUC = better separation; lower FNR@FDR and EER = better.\n")


def _plot_comparison(dist_by_model, results, label, source, run_name,
                     out_path) -> None:
    """One KDE panel per model (positive vs negative distances)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    pos_color, neg_color = '#a3164e', '#7eb6d9'
    keys = list(dist_by_model)
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 4.3 * len(keys)),
                             squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        p, q = dist_by_model[k]
        p, q = p[np.isfinite(p)], q[np.isfinite(q)]
        # Per-panel x-range: the two models live on very different distance
        # scales, so a shared axis would flatten one of them.
        finite = np.concatenate([p, q]) if len(p) or len(q) else np.array([1.0])
        x_max = float(np.percentile(finite, 99.5)) if len(finite) else 1.0
        x = np.linspace(0, x_max * 1.05, 400)
        for d, lbl, color in [(p, 'Positive (same peptide)', pos_color),
                              (q, 'Negative (diff peptide)', neg_color)]:
            if len(d) < 2:
                continue
            y = gaussian_kde(d)(x)
            ax.fill_between(x, 0, y, alpha=0.5, color=color, label=lbl)
            ax.plot(x, y, color=color, linewidth=1.5)
        m = results[k]
        ax.set_title(
            f"{label[k]}  —  AUC={m['auc']:.4f}, "
            f"FNR@{int(m['fdr_target']*100)}%FDR={m['fnr_at_fdr']:.4f}, "
            f"EER={m['eer']:.4f}\n{source[k]}", fontsize=10)
        ax.set_xlabel('Embedded distance')
        ax.set_ylabel('Density')
        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(f"GLEAMS holdout comparison — {run_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")
