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
    """Compute mean validation loss and collect positive/negative embedded distances.

    Returns (avg_loss, pos_distances, neg_distances).
    """
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    pos_dist_parts: List[np.ndarray] = []
    neg_dist_parts: List[np.ndarray] = []

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validation", unit="batch")

        for batch in val_pbar:
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

            val_pbar.set_postfix({'val_loss': f'{total_loss/n_batches:.4f}'})

    avg_loss = total_loss / max(n_batches, 1)
    pos_dist = np.concatenate(pos_dist_parts) if pos_dist_parts else np.array([])
    neg_dist = np.concatenate(neg_dist_parts) if neg_dist_parts else np.array([])
    return avg_loss, pos_dist, neg_dist


def train_model(run_dir: str):
    """Train one model and write all artifacts (log, loss CSV, checkpoints, plots) into `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)

    log_csv_path = os.path.join(run_dir, 'loss_log.csv')

    # -----------------------------
    # Load metadata and pairs
    # -----------------------------
    print("Loading metadata files...")
    train_metadata = pd.read_parquet(config.TRAIN_METADATA_FILE)
    test_metadata = pd.read_parquet(config.TEST_METADATA_FILE)

    print(f"Train metadata: {len(train_metadata):,} rows")
    print(f"Test metadata: {len(test_metadata):,} rows")

    if 'scan' not in train_metadata.columns:
        raise ValueError(
            "Metadata must have a 'scan' column to link pairs to MGF positions."
        )

    train_pos_pairs = np.load(config.TRAIN_POS_PAIRS_FILE).astype(int)
    train_neg_pairs = np.load(config.TRAIN_NEG_PAIRS_FILE).astype(int)
    val_pos_pairs = np.load(config.VAL_POS_PAIRS_FILE).astype(int)
    val_neg_pairs = np.load(config.VAL_NEG_PAIRS_FILE).astype(int)

    def _subsample(pairs: np.ndarray, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
        if cap is None or len(pairs) <= cap:
            return pairs
        idx = rng.choice(len(pairs), cap, replace=False)
        return pairs[idx]

    rng = np.random.default_rng(seed=config.SUBSAMPLE_SEED)
    train_pos_pairs = _subsample(train_pos_pairs, config.MAX_TRAIN_PAIRS_PER_CLASS, rng)
    train_neg_pairs = _subsample(train_neg_pairs, config.MAX_TRAIN_PAIRS_PER_CLASS, rng)
    val_pos_pairs = _subsample(val_pos_pairs, config.MAX_VAL_PAIRS_PER_CLASS, rng)
    val_neg_pairs = _subsample(val_neg_pairs, config.MAX_VAL_PAIRS_PER_CLASS, rng)

    print(f"Train pairs (after cap): {len(train_pos_pairs):,} positive, "
          f"{len(train_neg_pairs):,} negative")
    print(f"Val pairs (after cap):   {len(val_pos_pairs):,} positive, "
          f"{len(val_neg_pairs):,} negative")

    # -----------------------------
    # Pre-encode MGF → .npz (one pass; cached on disk)
    # -----------------------------
    print("Pre-encoding spectra from MGF (one pass, one-time cost)...")
    preencode_mgf_to_npzs(config.MGF_PATH, [
        (train_metadata, config.TRAIN_NPZ_PATH),
        (test_metadata, config.VAL_NPZ_PATH),
    ])

    # -----------------------------
    # Datasets + DataLoaders
    # -----------------------------
    train_dataset = SpectrumPairDataset(
        config.TRAIN_NPZ_PATH, train_pos_pairs, train_neg_pairs,
        max_peaks=config.MAX_PEAKS)
    val_dataset = SpectrumPairDataset(
        config.VAL_NPZ_PATH, val_pos_pairs, val_neg_pairs,
        max_peaks=config.MAX_PEAKS)
    print(f"Train dataset: {len(train_dataset):,} pairs")
    print(f"Val dataset:   {len(val_dataset):,} pairs")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=collate_pairs, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
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
                "batch_size", "dim_model", "n_layers", "n_head",
                "dim_feedforward", "dropout", "lr", "margin",
            ],
        )
        if write_header:
            writer.writeheader()

        # -----------------------------
        # Training loop
        # -----------------------------
        n_epochs = config.N_EPOCHS
        global_step = 0
        best_val_loss = float('inf')
        best_model_path = os.path.join(run_dir, 'best_model.pt')
        final_model_path = os.path.join(run_dir, 'final_model.pt')
        plots_dir = os.path.join(run_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            encoder.train()
            epoch_train_loss = 0.0
            n_train_batches = 0
            train_pos_dist_parts: List[np.ndarray] = []
            train_neg_dist_parts: List[np.ndarray] = []

            t0 = time.time()
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", unit="batch")

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
                        "batch_size": batch_size, "dim_model": dim_model,
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

            avg_val_loss, val_pos_dist, val_neg_dist = evaluate_contrastive(
                encoder, val_loader, criterion, device)
            plot_distance_distributions(
                val_pos_dist, val_neg_dist,
                os.path.join(plots_dir, f'val_epoch_{epoch:03d}.png'),
                title=f'Validation — epoch {epoch} (loss={avg_val_loss:.4f})',
            )

            writer.writerow({
                "timestamp": time.time(), "epoch": epoch, "batch": "",
                "split": "train_epoch", "loss": avg_train_loss,
                "n_pairs": len(train_pos_pairs) + len(train_neg_pairs),
                "batch_size": batch_size, "dim_model": dim_model,
                "n_layers": n_layers, "n_head": n_head,
                "dim_feedforward": dim_feedforward, "dropout": dropout,
                "lr": optimizer.param_groups[0]['lr'], "margin": criterion.margin,
            })
            writer.writerow({
                "timestamp": time.time(), "epoch": epoch, "batch": "",
                "split": "val_epoch", "loss": avg_val_loss,
                "n_pairs": len(val_pos_pairs) + len(val_neg_pairs),
                "batch_size": batch_size, "dim_model": dim_model,
                "n_layers": n_layers, "n_head": n_head,
                "dim_feedforward": dim_feedforward, "dropout": dropout,
                "lr": optimizer.param_groups[0]['lr'], "margin": criterion.margin,
            })
            f.flush()

            # Refresh the loss-curve plot from the CSV. Overwrites the file each
            # epoch so opening loss_curves.png acts as a live training monitor.
            plot_loss_curves(
                log_csv_path,
                os.path.join(run_dir, 'loss_curves.png'),
                title=f'Loss (epoch {epoch}/{n_epochs})',
            )

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'hyperparameters': {
                        'dim_model': dim_model, 'n_layers': n_layers,
                        'n_head': n_head, 'dim_feedforward': dim_feedforward,
                        'dropout': dropout, 'margin': criterion.margin,
                    },
                }, best_model_path)
                print(f"✓ Saved best model (val_loss={avg_val_loss:.6f}) to {best_model_path}")

            dt = time.time() - t0
            print(
                f"Epoch {epoch}/{n_epochs} | "
                f"train_loss={avg_train_loss:.6f} | "
                f"val_loss={avg_val_loss:.6f} | "
                f"lr={current_lr:.2e} | "
                f"time={dt:.1f}s"
            )

    torch.save({
        'epoch': n_epochs,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'hyperparameters': {
            'dim_model': dim_model, 'n_layers': n_layers,
            'n_head': n_head, 'dim_feedforward': dim_feedforward,
            'dropout': dropout, 'margin': criterion.margin,
        },
    }, final_model_path)

    print(f"\nTraining complete!")
    print(f"Loss log written to: {log_csv_path}")
    print(f"Best model (val_loss={best_val_loss:.6f}) saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
