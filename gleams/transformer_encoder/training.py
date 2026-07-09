"""Training orchestrator: builds data + model, runs the contrastive loop,
saves checkpoints, CSV loss log, and per-epoch distance-distribution plots.
"""

import csv
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from metrics import ranking_metrics
from model import ContrastiveLoss, CustomSpectrumTransformerEncoder
from spectrum_pairs import (
    SpectrumPairDataset,
    collate_pairs,
    preencode_mgf_to_npzs,
)
from visualization import plot_distance_distributions, plot_loss_curves


def evaluate_contrastive(
    encoder,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
):
    """Compute mean loss on a monitor set and collect positive/negative distances.

    Returns (avg_loss, pos_distances, neg_distances).
    """
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    pos_dist_parts: List[np.ndarray] = []
    neg_dist_parts: List[np.ndarray] = []

    with torch.no_grad():
        eval_pbar = tqdm(dataloader, desc="Test", unit="batch")

        for batch in eval_pbar:
            (mz1, int1, pepmass1, charge1), (mz2, int2, pepmass2, charge2), labels = batch

            mz1 = mz1.to(device)
            int1 = int1.to(device)
            pepmass1 = pepmass1.to(device)
            charge1 = charge1.to(device)

            mz2 = mz2.to(device)
            int2 = int2.to(device)
            pepmass2 = pepmass2.to(device)
            charge2 = charge2.to(device)

            labels = labels.to(device)

            emb1_full, _ = encoder(mz1, int1, pepmass=pepmass1, charge=charge1)
            emb2_full, _ = encoder(mz2, int2, pepmass=pepmass2, charge=charge2)

            emb1 = emb1_full[:, 0, :]
            emb2 = emb2_full[:, 0, :]

            loss = criterion(emb1, emb2, labels)
            total_loss += float(loss.item())
            n_batches += 1

            dists = torch.nn.functional.pairwise_distance(emb1, emb2).cpu().numpy()
            lbls = labels.cpu().numpy()
            pos_dist_parts.append(dists[lbls == 1])
            neg_dist_parts.append(dists[lbls == 0])

            eval_pbar.set_postfix({'test_loss': f'{total_loss/n_batches:.4f}'})

    avg_loss = total_loss / max(n_batches, 1)
    pos_dist = np.concatenate(pos_dist_parts) if pos_dist_parts else np.array([])
    neg_dist = np.concatenate(neg_dist_parts) if neg_dist_parts else np.array([])
    return avg_loss, pos_dist, neg_dist


def _compact_referenced_spectra(metadata, pair_arrays):
    """Keep only the metadata rows referenced by `pair_arrays`, remapping the
    pair indices onto the compacted row order.

    The pre-encoded .npz only needs the spectra the (subsampled) pairs actually
    touch — typically a small fraction of the full metadata. Encoding every row
    instead builds tens of GB of peak arrays in RAM (OOM-kills the WSL VM) and a
    correspondingly huge .npz that the Dataset then loads whole. Compacting first
    keeps memory proportional to the pairs we train on, not the whole dataset.

    Returns (compact_metadata, [remapped_pair_array, ...]).
    """
    nonempty = [p for p in pair_arrays if len(p)]
    if not nonempty:
        return metadata.iloc[:0].reset_index(drop=True), list(pair_arrays)
    used_rows = np.unique(np.concatenate([p.reshape(-1) for p in nonempty]))
    # old metadata row index -> position within the compacted frame
    remap = np.full(len(metadata), -1, dtype=np.int64)
    remap[used_rows] = np.arange(len(used_rows), dtype=np.int64)
    compact = metadata.iloc[used_rows].reset_index(drop=True)
    remapped = [remap[p] for p in pair_arrays]
    return compact, remapped


def train_model(run_dir: str):
    """Train one model and write all artifacts (log, loss CSV, checkpoints, plots) into `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)

    log_csv_path = os.path.join(run_dir, 'loss_log.csv')

    # -----------------------------
    # Load metadata and pairs
    # -----------------------------
    print("Loading metadata files...")
    train_metadata = pd.read_parquet(config.TRAIN_METADATA_FILE)
    # Training uses train (fit) + test (in-training monitor: early stopping, LR
    # schedule, per-epoch loss). The val split is deliberately left untouched as
    # a later holdout, so it is not loaded here.
    test_metadata = pd.read_parquet(config.TEST_METADATA_FILE)

    print(f"Train metadata: {len(train_metadata):,} rows")
    print(f"Test metadata:  {len(test_metadata):,} rows")

    if 'scan' not in train_metadata.columns:
        raise ValueError(
            "Metadata must have a 'scan' column to link pairs to MGF positions."
        )

    train_pos_pairs = np.load(config.TRAIN_POS_PAIRS_FILE).astype(int)
    train_neg_pairs = np.load(config.TRAIN_NEG_PAIRS_FILE).astype(int)
    test_pos_pairs = np.load(config.TEST_POS_PAIRS_FILE).astype(int)
    test_neg_pairs = np.load(config.TEST_NEG_PAIRS_FILE).astype(int)

    def _subsample(pairs: np.ndarray, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
        if cap is None or len(pairs) <= cap:
            return pairs
        idx = rng.choice(len(pairs), cap, replace=False)
        return pairs[idx]

    rng = np.random.default_rng(seed=config.SUBSAMPLE_SEED)
    train_pos_pairs = _subsample(train_pos_pairs, config.MAX_TRAIN_PAIRS_PER_CLASS, rng)
    train_neg_pairs = _subsample(train_neg_pairs, config.MAX_TRAIN_PAIRS_PER_CLASS, rng)
    test_pos_pairs = _subsample(test_pos_pairs, config.MAX_TEST_PAIRS_PER_CLASS, rng)
    test_neg_pairs = _subsample(test_neg_pairs, config.MAX_TEST_PAIRS_PER_CLASS, rng)

    print(f"Train pairs (after cap): {len(train_pos_pairs):,} positive, "
          f"{len(train_neg_pairs):,} negative")
    print(f"Test pairs (after cap):  {len(test_pos_pairs):,} positive, "
          f"{len(test_neg_pairs):,} negative")

    # Restrict encoding to the spectra the subsampled pairs actually reference,
    # and remap the pair indices onto that compact order. Without this we'd
    # encode all ~10M spectra (huge RAM + .npz), most of which no pair touches.
    train_metadata, (train_pos_pairs, train_neg_pairs) = _compact_referenced_spectra(
        train_metadata, [train_pos_pairs, train_neg_pairs])
    test_metadata, (test_pos_pairs, test_neg_pairs) = _compact_referenced_spectra(
        test_metadata, [test_pos_pairs, test_neg_pairs])
    print(f"Referenced spectra to encode: {len(train_metadata):,} train, "
          f"{len(test_metadata):,} test (only these are read into the .npz).")

    # -----------------------------
    # Pre-encode MGF → .npz (one pass; cached on disk)
    # -----------------------------
    # NOTE: the .npz now holds exactly the spectra referenced by the current
    # subsampled pairs, so it's tied to MAX_*_PAIRS_PER_CLASS and SUBSAMPLE_SEED.
    # Delete data/*_spectra.npz if you change those before rerunning.
    print("Pre-encoding spectra from MGF (one pass, one-time cost)...")
    preencode_mgf_to_npzs(config.MGF_PATH, [
        (train_metadata, config.TRAIN_NPZ_PATH),
        (test_metadata, config.TEST_NPZ_PATH),
    ])

    # -----------------------------
    # Datasets + DataLoaders
    # -----------------------------
    train_dataset = SpectrumPairDataset(
        config.TRAIN_NPZ_PATH, train_pos_pairs, train_neg_pairs,
        max_peaks=config.MAX_PEAKS)
    test_dataset = SpectrumPairDataset(
        config.TEST_NPZ_PATH, test_pos_pairs, test_neg_pairs,
        max_peaks=config.MAX_PEAKS)
    print(f"Train dataset: {len(train_dataset):,} pairs")
    print(f"Test dataset:  {len(test_dataset):,} pairs")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=collate_pairs, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_pairs, num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # -----------------------------
    # Model
    # -----------------------------
    encoder = CustomSpectrumTransformerEncoder(
        d_model=config.DIM_MODEL,
        nhead=config.N_HEAD,
        dim_feedforward=config.DIM_FEEDFORWARD,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder = encoder.to(device)

    optimizer = optim.Adam(
        encoder.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = ContrastiveLoss(margin=config.MARGIN)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    # Convenience locals so the CSV/checkpoint blocks below stay readable.
    batch_size = config.BATCH_SIZE
    dim_model = config.DIM_MODEL
    n_head = config.N_HEAD
    dim_feedforward = config.DIM_FEEDFORWARD
    n_layers = config.N_LAYERS
    dropout = config.DROPOUT

    # -----------------------------
    # Logging setup
    # -----------------------------
    log_batch_level = config.LOG_BATCH_LEVEL
    write_header = not os.path.exists(log_csv_path)

    with open(log_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "epoch", "batch", "split", "loss", "n_pairs",
                "batch_size", "max_peaks", "dim_model", "n_layers", "n_head",
                "dim_feedforward", "dropout", "lr", "margin",
                # Loss-independent, model-agnostic separation metrics (see
                # metrics.py). Populated on *_epoch rows only; comparable to the
                # CNN when the same metric is run on its distances.
                "auc", "fnr_at_fdr", "fdr_threshold", "eer",
            ],
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()

        # -----------------------------
        # Training loop
        # -----------------------------
        max_epochs = config.MAX_EPOCHS
        early_stop_patience = config.EARLY_STOPPING_PATIENCE
        global_step = 0
        best_test_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = os.path.join(run_dir, 'best_model.pt')
        final_model_path = os.path.join(run_dir, 'final_model.pt')
        plots_dir = os.path.join(run_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        for epoch in range(1, max_epochs + 1):
            encoder.train()
            epoch_train_loss = 0.0
            n_train_batches = 0
            train_pos_dist_parts: List[np.ndarray] = []
            train_neg_dist_parts: List[np.ndarray] = []

            t0 = time.time()
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", unit="batch")

            for batch_idx, batch in enumerate(batch_pbar, start=1):
                (mz1, int1, pepmass1, charge1), (mz2, int2, pepmass2, charge2), labels = batch

                mz1 = mz1.to(device)
                int1 = int1.to(device)
                pepmass1 = pepmass1.to(device)
                charge1 = charge1.to(device)

                mz2 = mz2.to(device)
                int2 = int2.to(device)
                pepmass2 = pepmass2.to(device)
                charge2 = charge2.to(device)

                labels = labels.to(device)

                emb1_full, _ = encoder(mz1, int1, pepmass=pepmass1, charge=charge1)
                emb2_full, _ = encoder(mz2, int2, pepmass=pepmass2, charge=charge2)

                emb1 = emb1_full[:, 0, :]
                emb2 = emb2_full[:, 0, :]

                loss = criterion(emb1, emb2, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                epoch_train_loss += loss_value
                n_train_batches += 1
                global_step += 1

                with torch.no_grad():
                    dists_np = torch.nn.functional.pairwise_distance(emb1, emb2).cpu().numpy()
                    lbls_np = labels.cpu().numpy()
                    train_pos_dist_parts.append(dists_np[lbls_np == 1])
                    train_neg_dist_parts.append(dists_np[lbls_np == 0])

                batch_pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{epoch_train_loss/n_train_batches:.4f}',
                })

                if log_batch_level:
                    writer.writerow({
                        "timestamp": time.time(), "epoch": epoch, "batch": batch_idx,
                        "split": "train", "loss": loss_value, "n_pairs": "",
                        "batch_size": batch_size, "max_peaks": config.MAX_PEAKS,
                        "dim_model": dim_model,
                        "n_layers": n_layers, "n_head": n_head,
                        "dim_feedforward": dim_feedforward, "dropout": dropout,
                        "lr": optimizer.param_groups[0]['lr'],
                        "margin": criterion.margin,
                    })

            avg_train_loss = epoch_train_loss / max(n_train_batches, 1)

            train_pos_dist = (
                np.concatenate(train_pos_dist_parts) if train_pos_dist_parts else np.array([]))
            train_neg_dist = (
                np.concatenate(train_neg_dist_parts) if train_neg_dist_parts else np.array([]))
            plot_distance_distributions(
                train_pos_dist, train_neg_dist,
                os.path.join(plots_dir, f'train_epoch_{epoch:03d}.png'),
                title=f'Train — epoch {epoch} (loss={avg_train_loss:.4f})',
            )

            avg_test_loss, test_pos_dist, test_neg_dist = evaluate_contrastive(
                encoder, test_loader, criterion, device)
            plot_distance_distributions(
                test_pos_dist, test_neg_dist,
                os.path.join(plots_dir, f'test_epoch_{epoch:03d}.png'),
                title=f'Test — epoch {epoch} (loss={avg_test_loss:.4f})',
            )

            # Loss-independent separation metrics on the distances already
            # collected above. These — not the loss — are what to compare
            # against the CNN (run metrics.ranking_metrics on its distances too).
            train_metrics = ranking_metrics(train_pos_dist, train_neg_dist)
            test_metrics = ranking_metrics(test_pos_dist, test_neg_dist)

            writer.writerow({
                "timestamp": time.time(), "epoch": epoch, "batch": "",
                "split": "train_epoch", "loss": avg_train_loss,
                "n_pairs": len(train_pos_pairs) + len(train_neg_pairs),
                "batch_size": batch_size, "max_peaks": config.MAX_PEAKS,
                "dim_model": dim_model,
                "n_layers": n_layers, "n_head": n_head,
                "dim_feedforward": dim_feedforward, "dropout": dropout,
                "lr": optimizer.param_groups[0]['lr'], "margin": criterion.margin,
                "auc": train_metrics['auc'],
                "fnr_at_fdr": train_metrics['fnr_at_fdr'],
                "fdr_threshold": train_metrics['fdr_threshold'],
                "eer": train_metrics['eer'],
            })
            writer.writerow({
                "timestamp": time.time(), "epoch": epoch, "batch": "",
                "split": "test_epoch", "loss": avg_test_loss,
                "n_pairs": len(test_pos_pairs) + len(test_neg_pairs),
                "batch_size": batch_size, "max_peaks": config.MAX_PEAKS,
                "dim_model": dim_model,
                "n_layers": n_layers, "n_head": n_head,
                "dim_feedforward": dim_feedforward, "dropout": dropout,
                "lr": optimizer.param_groups[0]['lr'], "margin": criterion.margin,
                "auc": test_metrics['auc'],
                "fnr_at_fdr": test_metrics['fnr_at_fdr'],
                "fdr_threshold": test_metrics['fdr_threshold'],
                "eer": test_metrics['eer'],
            })
            f.flush()

            # Refresh the loss-curve plot from the CSV. Overwrites the file each
            # epoch so opening loss_curves.png acts as a live training monitor.
            plot_loss_curves(
                log_csv_path,
                os.path.join(run_dir, 'loss_curves.png'),
                title=f'Loss (epoch {epoch}/{max_epochs})',
            )

            scheduler.step(avg_test_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'test_loss': avg_test_loss,
                    'hyperparameters': {
                        'dim_model': dim_model, 'n_layers': n_layers,
                        'n_head': n_head, 'dim_feedforward': dim_feedforward,
                        'dropout': dropout, 'margin': criterion.margin,
                    },
                }, best_model_path)
                print(f"✓ Saved best model (test_loss={avg_test_loss:.6f}) to {best_model_path}")
            else:
                epochs_without_improvement += 1

            dt = time.time() - t0
            print(
                f"Epoch {epoch}/{max_epochs} | "
                f"train_loss={avg_train_loss:.6f} | "
                f"test_loss={avg_test_loss:.6f} | "
                f"test_auc={test_metrics['auc']:.4f} | "
                f"test_fnr@1%fdr={test_metrics['fnr_at_fdr']:.4f} | "
                f"test_eer={test_metrics['eer']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"time={dt:.1f}s | "
                f"no_improve={epochs_without_improvement}"
            )

            # Early stopping: bail once test_loss has stopped improving.
            if (early_stop_patience is not None
                    and epochs_without_improvement >= early_stop_patience):
                print(
                    f"\n⏹ Early stopping triggered — no test_loss improvement "
                    f"for {epochs_without_improvement} consecutive epoch(s) "
                    f"(patience={early_stop_patience})."
                )
                break

    torch.save({
        'epoch': epoch,  # last epoch actually trained (may be < max_epochs after early stop)
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_test_loss': best_test_loss,
        'hyperparameters': {
            'dim_model': dim_model, 'n_layers': n_layers,
            'n_head': n_head, 'dim_feedforward': dim_feedforward,
            'dropout': dropout, 'margin': criterion.margin,
        },
    }, final_model_path)

    print(f"\nTraining complete after {epoch}/{max_epochs} epoch(s).")
    print(f"Loss log written to: {log_csv_path}")
    print(f"Best model (test_loss={best_test_loss:.6f}) saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
