"""Plotting helpers for training-time diagnostics.

All plots are saved to disk (no GUI backend) so this module is safe to use
inside `train_model`, under stdout teeing, or in a headless cron job.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend; safe under stdout-tee + no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


_POSITIVE_COLOR = '#a3164e'  # pink — same-peptide pairs
_NEGATIVE_COLOR = '#7eb6d9'  # blue — different-peptide pairs


def plot_distance_distributions(
    pos_dist: np.ndarray,
    neg_dist: np.ndarray,
    out_path: str,
    title: str = "",
) -> None:
    """Save a KDE plot of positive vs. negative embedded distances (Fig. 3-style).

    Use after every epoch on train and test sets to track separation. Returns
    silently when both distance arrays are empty or one-element (KDE undefined).
    """
    if len(pos_dist) < 2 and len(neg_dist) < 2:
        return
    x_max = max(
        float(pos_dist.max()) if len(pos_dist) else 1.0,
        float(neg_dist.max()) if len(neg_dist) else 1.0,
    )
    x = np.linspace(0, x_max * 1.05, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    for d, label, color in [
        (pos_dist, 'Positive pairs', _POSITIVE_COLOR),
        (neg_dist, 'Negative pairs', _NEGATIVE_COLOR),
    ]:
        if len(d) < 2:
            continue
        kde = gaussian_kde(d)
        y = kde(x)
        ax.fill_between(x, 0, y, alpha=0.5, color=color, label=label)
        ax.plot(x, y, color=color, linewidth=1.5)

    ax.set_xlabel('Embedded distance')
    ax.set_ylabel('Density')
    ax.set_xlim(0, x_max * 1.05)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=False)
    if title:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _format_hyperparams_subtitle(df: pd.DataFrame) -> str:
    """Two-line hyperparameter summary pulled from a loss-log dataframe.

    The CSV writer stamps every row with the same hyperparameters, so any
    train_epoch row gives the full architecture; the test_epoch row contributes
    only its pair count (test data size).
    """
    train_rows = df[df['split'] == 'train_epoch']
    test_rows = df[df['split'] == 'test_epoch']
    if len(train_rows) == 0:
        return ""
    row = train_rows.iloc[0]

    arch = [
        f"d_model={int(row['dim_model'])}",
        f"n_layers={int(row['n_layers'])}",
        f"n_head={int(row['n_head'])}",
        f"dim_ff={int(row['dim_feedforward'])}",
        f"dropout={row['dropout']}",
    ]
    optim_ = [
        f"margin={row['margin']}",
        f"batch_size={int(row['batch_size'])}",
        f"lr_init={float(row['lr']):.0e}",
        f"train_pairs={int(row['n_pairs']):,}",
    ]
    if len(test_rows) > 0:
        optim_.append(f"test_pairs={int(test_rows.iloc[0]['n_pairs']):,}")
    # max_peaks may be absent from older CSVs written before this column existed.
    if 'max_peaks' in row.index and pd.notna(row['max_peaks']):
        optim_.append(f"max_peaks={int(row['max_peaks'])}")
    return " · ".join(arch) + "\n" + " · ".join(optim_)


def plot_loss_curves(
    loss_log_csv: str,
    out_path: str,
    title: str = "",
) -> None:
    """Render train + test loss vs. epoch from a `loss_log.csv` file.

    Reads rows with `split == 'train_epoch'` and `split == 'test_epoch'`. Annotates
    the plot with the architecture + training hyperparameters that were stamped
    onto every CSV row. Safe to call mid-training (each call overwrites `out_path`
    with the latest data) so opening the PNG and refreshing acts as a live monitor.
    """
    df = pd.read_csv(loss_log_csv)
    train = df[df['split'] == 'train_epoch'].sort_values('epoch')
    test = df[df['split'] == 'test_epoch'].sort_values('epoch')

    if len(train) == 0 and len(test) == 0:
        return  # nothing logged yet

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(train) > 0:
        ax.plot(train['epoch'], train['loss'], marker='o', linewidth=1.8,
                color=_POSITIVE_COLOR, label='Train')
    if len(test) > 0:
        ax.plot(test['epoch'], test['loss'], marker='o', linewidth=1.8,
                color=_NEGATIVE_COLOR, label='Test')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive loss')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=False)
    if title:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Two-line hyperparameter subtitle below the x-axis label.
    params_str = _format_hyperparams_subtitle(df)
    if params_str:
        # Reserve ~13% of the figure for the subtitle, then drop the text in.
        fig.tight_layout(rect=[0, 0.13, 1, 1])
        fig.text(
            0.5, 0.04, params_str,
            ha='center', va='bottom',
            fontsize=8, color='#555555',
            multialignment='center',
        )
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
