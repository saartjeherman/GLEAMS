"""Model-agnostic embedding-quality metrics for comparing spectrum embedders.

The contrastive *loss* is not comparable across models: it depends on the loss
formulation (label-certainty weighting, margin ramp), the margin value, and the
positive/negative balancing — all of which differ between the CNN and the
transformer. These metrics instead score only the *separation* between
same-peptide (positive) and different-peptide (negative) pairs in the embedded
space, using the Euclidean distances each model produces. They are therefore
directly comparable between any two embedders trained on the same pairs.

All functions take two 1-D arrays of embedded distances — `pos_dist` for
same-peptide pairs, `neg_dist` for different-peptide pairs — and are NumPy-only
(no scikit-learn dependency).

Mirrors the paper's evaluation (Bittremieux et al. 2022, Extended Data Fig. 3-4):
same-peptide pairs should have *small* distances, so a smaller distance is a
more confident "same peptide" prediction.
"""

from typing import Dict

import numpy as np


def roc_auc(pos_dist: np.ndarray, neg_dist: np.ndarray) -> float:
    """AUC that a random positive pair is closer than a random negative pair.

    Computed as the Mann-Whitney U statistic (rank-based), so it needs no
    threshold sweep and equals the area under the ROC curve of the classifier
    that scores pairs by negative distance. 1.0 = perfect separation, 0.5 =
    chance. Returns NaN if either group is empty.
    """
    n_pos, n_neg = len(pos_dist), len(neg_dist)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # Rank all distances together; ties get average ranks. AUC = P(neg > pos).
    all_dist = np.concatenate([pos_dist, neg_dist])
    order = all_dist.argsort(kind='mergesort')
    ranks = np.empty(len(all_dist), dtype=np.float64)
    ranks[order] = np.arange(1, len(all_dist) + 1)
    # Average ranks within tie groups so the statistic is exact under ties.
    _assign_tie_averaged_ranks(all_dist, order, ranks)
    rank_sum_pos = ranks[:n_pos].sum()
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    # Ranks ascend with distance, so u_pos counts (pos_dist > neg_dist) pairs.
    # We want P(pos_dist < neg_dist) — positives are the *closer* pairs — hence
    # the complement.
    return float(1.0 - u_pos / (n_pos * n_neg))


def _assign_tie_averaged_ranks(values: np.ndarray, order: np.ndarray,
                               ranks: np.ndarray) -> None:
    """Replace ranks of tied values (in `order`) with their group mean, in place."""
    sorted_vals = values[order]
    i = 0
    n = len(sorted_vals)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # average of ranks (i+1)..j
            ranks[order[i:j]] = avg
        i = j


def fnr_at_fdr(pos_dist: np.ndarray, neg_dist: np.ndarray,
               fdr_target: float = 0.01) -> Dict[str, float]:
    """False-negative rate at a distance threshold giving <= `fdr_target` FDR.

    Sweeps the distance threshold from small to large. A pair is predicted
    "same peptide" when its distance <= threshold; among those predictions the
    FDR is (negatives predicted same) / (all predicted same). We take the
    largest threshold whose FDR stays within `fdr_target` (maximising recall of
    true positives), then report:

      - ``fnr``       : fraction of positive pairs with distance above that
                        threshold (missed same-peptide pairs). NaN if no
                        threshold achieves the target FDR.
      - ``threshold`` : the chosen distance cutoff (NaN if unachievable).
      - ``tpr``       : 1 - fnr, the recovered fraction of positives.

    This is the metric behind the paper's "1% FNR at 1% FDR" result.
    """
    n_pos, n_neg = len(pos_dist), len(neg_dist)
    out = {'fnr': float('nan'), 'threshold': float('nan'), 'tpr': float('nan')}
    if n_pos == 0 or n_neg == 0:
        return out

    dist = np.concatenate([pos_dist, neg_dist])
    is_pos = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    order = dist.argsort(kind='mergesort')
    dist_sorted = dist[order]
    is_pos_sorted = is_pos[order]

    tp = np.cumsum(is_pos_sorted)          # positives with dist <= threshold
    fp = np.cumsum(~is_pos_sorted)         # negatives with dist <= threshold
    predicted = tp + fp
    fdr = np.where(predicted > 0, fp / predicted, 0.0)

    # Only consider thresholds at the *end* of a tie run, so the cutoff is a
    # real achievable distance and FDR/recall are evaluated consistently.
    tie_end = np.ones(len(dist_sorted), bool)
    tie_end[:-1] = dist_sorted[1:] != dist_sorted[:-1]
    feasible = (fdr <= fdr_target) & tie_end
    if not feasible.any():
        return out

    best = np.flatnonzero(feasible)[-1]    # largest threshold meeting the FDR
    out['threshold'] = float(dist_sorted[best])
    out['tpr'] = float(tp[best] / n_pos)
    out['fnr'] = float(1.0 - tp[best] / n_pos)
    return out


def equal_error_rate(pos_dist: np.ndarray, neg_dist: np.ndarray) -> float:
    """Rate at which FPR == FNR — a single threshold-free summary of separation.

    Lower is better (0 = perfect). Returns NaN if either group is empty.
    """
    n_pos, n_neg = len(pos_dist), len(neg_dist)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    dist = np.concatenate([pos_dist, neg_dist])
    is_pos = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    order = dist.argsort(kind='mergesort')
    is_pos_sorted = is_pos[order]
    tp = np.cumsum(is_pos_sorted)
    fp = np.cumsum(~is_pos_sorted)
    tpr = tp / n_pos                       # recall of positives at each cutoff
    fpr = fp / n_neg                       # negatives wrongly called "same"
    fnr = 1.0 - tpr
    # EER is where fnr crosses fpr; take the threshold minimising |fnr - fpr|.
    idx = np.argmin(np.abs(fnr - fpr))
    return float((fnr[idx] + fpr[idx]) / 2.0)


def ranking_metrics(pos_dist: np.ndarray, neg_dist: np.ndarray,
                    fdr_target: float = 0.01) -> Dict[str, float]:
    """All comparison metrics in one call, keyed for CSV logging.

    Returns ``auc``, ``fnr_at_fdr``, ``fdr_threshold``, ``fdr_target``, ``eer``,
    ``n_pos``, ``n_neg``. Feed the same function the CNN's and the transformer's
    distances to get directly comparable numbers.
    """
    pos_dist = np.asarray(pos_dist, dtype=np.float64).ravel()
    neg_dist = np.asarray(neg_dist, dtype=np.float64).ravel()
    fdr = fnr_at_fdr(pos_dist, neg_dist, fdr_target)
    return {
        'auc': roc_auc(pos_dist, neg_dist),
        'fnr_at_fdr': fdr['fnr'],
        'fdr_threshold': fdr['threshold'],
        'fdr_target': fdr_target,
        'eer': equal_error_rate(pos_dist, neg_dist),
        'n_pos': int(len(pos_dist)),
        'n_neg': int(len(neg_dist)),
    }
