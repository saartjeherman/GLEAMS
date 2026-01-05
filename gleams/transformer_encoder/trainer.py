"""Training functions for the spectrum encoder model."""

import os
import sys
import csv
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from .models import CustomSpectrumTransformerEncoder, ContrastiveLoss
from .dataset import SpectraDataset
from .dataloader import create_dataloader
from .evaluator import evaluate_contrastive
from . import config


def train_model(args):
    """
    Train the spectrum encoder model with contrastive learning.
    
    Args:
        args: Parsed command-line arguments containing:
            - Data paths (metadata, pairs, MGF file)
            - Model hyperparameters
            - Training configuration
            - Output paths
    """
    # -----------------------------
    # Paths from command-line arguments
    # -----------------------------
    train_metadata_file = args.train_metadata
    test_metadata_file = args.test_metadata
    
    filenames_pairs_pos = [
        args.train_pairs_pos,
        args.test_pairs_pos
    ]
    filenames_pairs_neg = [
        args.train_pairs_neg,
        args.test_pairs_neg
    ]

    mgf_path = args.mgf_file
    log_csv_path = args.log_csv

    # -----------------------------
    # Load metadata and pairs
    # -----------------------------
    print("Loading metadata files...")
    train_metadata = pd.read_parquet(train_metadata_file)
    test_metadata = pd.read_parquet(test_metadata_file)
    
    print(f"Train metadata: {len(train_metadata):,} rows")
    print(f"Test metadata: {len(test_metadata):,} rows")
    
    # Check if metadata has 'scan' column (required for proper linking)
    if config.REQUIRED_METADATA_COLUMN not in train_metadata.columns:
        raise ValueError(
            f"Metadata must have a '{config.REQUIRED_METADATA_COLUMN}' column to link pairs to MGF positions!\n"
            "Run the fixed metadata extraction code in read.ipynb to create metadata with scan numbers."
        )
    
    # Load pairs (these are indices into the metadata DataFrames)
    train_pos_pairs = np.load(filenames_pairs_pos[0]).astype(int)
    val_pos_pairs   = np.load(filenames_pairs_pos[1]).astype(int)
    train_neg_pairs = np.load(filenames_pairs_neg[0]).astype(int)
    val_neg_pairs   = np.load(filenames_pairs_neg[1]).astype(int)
    
    print(f"Train pairs: {len(train_pos_pairs):,} positive, {len(train_neg_pairs):,} negative")
    print(f"Val pairs: {len(val_pos_pairs):,} positive, {len(val_neg_pairs):,} negative")

    # -----------------------------
    # Model setup
    # -----------------------------
    dim_model = args.dim_model
    n_head = args.n_head
    dim_feedforward = args.dim_feedforward
    n_layers = args.n_layers
    dropout = args.dropout

    encoder = CustomSpectrumTransformerEncoder(
        d_model=dim_model,
        nhead=n_head,
        dim_feedforward=dim_feedforward,
        n_layers=n_layers,
        dropout=dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder = encoder.to(device)

    optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = ContrastiveLoss(margin=args.margin)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.SCHEDULER_FACTOR, 
        patience=args.scheduler_patience, 
        verbose=config.SCHEDULER_VERBOSE
    )

    # -----------------------------
    # Build dataset
    # -----------------------------
    train_indices = np.unique(np.concatenate([train_pos_pairs.reshape(-1), train_neg_pairs.reshape(-1)]))
    val_indices   = np.unique(np.concatenate([val_pos_pairs.reshape(-1),   val_neg_pairs.reshape(-1)]))
    all_indices   = np.unique(np.concatenate([train_indices, val_indices]))
    
    print(f"Unique metadata row indices needed (train): {len(train_indices):,}")
    print(f"Unique metadata row indices needed (val):   {len(val_indices):,}")
    print(f"Unique metadata row indices needed (total): {len(all_indices):,}")
    
    # Convert metadata row indices to MGF scan numbers
    all_metadata_rows = train_metadata.iloc[all_indices]
    mgf_scan_numbers = all_metadata_rows['scan'].values
    
    print(f"MGF scan number range: {mgf_scan_numbers.min():,} to {mgf_scan_numbers.max():,}")
    print(f"\nLoading {len(mgf_scan_numbers):,} spectra from MGF file by scan number...")
    
    base_dataset = SpectraDataset(mgf_path, scan_nrs=mgf_scan_numbers)
    print("✓ Dataset loading complete!")
    print(f"⚠️  Max peaks per spectrum: {base_dataset.max_peaks:,} (affects GPU memory usage)\n")

    # -----------------------------
    # Create index mappings for pair remapping
    # -----------------------------
    print("Creating index mapping...")
    
    scan_to_dataset_idx = {int(scan): i for i, scan in enumerate(base_dataset.identifiers)}
    
    metadata_idx_to_dataset_idx = {}
    for meta_row_idx in all_indices:
        scan_num = int(train_metadata.iloc[meta_row_idx]['scan'])
        dataset_idx = scan_to_dataset_idx.get(scan_num)
        if dataset_idx is not None:
            metadata_idx_to_dataset_idx[meta_row_idx] = dataset_idx
    
    print(f"✓ Mapped {len(metadata_idx_to_dataset_idx):,} metadata rows to dataset indices")
    
    def remap_pairs(pairs: np.ndarray, desc: str = "Remapping") -> List[Tuple[int, int]]:
        """Remap pairs from metadata row indices to dataset indices."""
        out = []
        missing = 0
        for a, b in tqdm(pairs, desc=desc, unit="pairs", disable=len(pairs) > 1000000, file=sys.__stdout__):
            ia = metadata_idx_to_dataset_idx.get(int(a))
            ib = metadata_idx_to_dataset_idx.get(int(b))
            if ia is None or ib is None:
                missing += 1
                continue
            out.append((ia, ib))
        if missing:
            print(f"  ⚠️  Warning: {missing} pairs had missing spectra and were dropped.")
        return out

    print("\nRemapping pair indices to dataset indices...")
    train_pos = remap_pairs(train_pos_pairs, "Train positive pairs")
    train_neg = remap_pairs(train_neg_pairs, "Train negative pairs")
    val_pos   = remap_pairs(val_pos_pairs, "Val positive pairs")
    val_neg   = remap_pairs(val_neg_pairs, "Val negative pairs")
    print("✓ Pair remapping complete!\n")

    # -----------------------------
    # DataLoaders
    # -----------------------------
    batch_size = args.batch_size

    # Optional: limit dataset size for debugging
    if args.max_train_pairs:
        train_pos = train_pos[:args.max_train_pairs]
        train_neg = train_neg[:args.max_train_pairs]
    if args.max_val_pairs:
        val_pos = val_pos[:args.max_val_pairs]
        val_neg = val_neg[:args.max_val_pairs]

    train_loader = create_dataloader(base_dataset, train_pos, train_neg, batch_size=batch_size, shuffle=config.SHUFFLE_TRAIN)
    val_loader   = create_dataloader(base_dataset, val_pos,   val_neg,   batch_size=batch_size, shuffle=config.SHUFFLE_VAL)

    # -----------------------------
    # Training loop
    # -----------------------------
    log_batch_level = config.LOG_BATCH_LEVEL

    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    write_header = not os.path.exists(log_csv_path)

    with open(log_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config.LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()

        n_epochs = args.n_epochs
        global_step = 0
        best_val_loss = float('inf')
        best_model_path = args.best_model_path
        final_model_path = args.final_model_path
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            encoder.train()
            epoch_train_loss = 0.0
            n_train_batches = 0

            t0 = time.time()
            
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", unit="batch", file=sys.__stdout__)

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
                
                batch_pbar.set_postfix({'loss': f'{loss_value:.4f}', 'avg_loss': f'{epoch_train_loss/n_train_batches:.4f}'})

                if log_batch_level:
                    writer.writerow({
                        "timestamp": time.time(),
                        "epoch": epoch,
                        "batch": batch_idx,
                        "split": "train",
                        "loss": loss_value,
                        "n_pairs": "",
                        "batch_size": batch_size,
                        "dim_model": dim_model,
                        "n_layers": n_layers,
                        "n_head": n_head,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                        "lr": optimizer.param_groups[0]['lr'],
                        "margin": criterion.margin
                    })

            avg_train_loss = epoch_train_loss / max(n_train_batches, 1)

            # Validation
            avg_val_loss = evaluate_contrastive(encoder, val_loader, criterion, device)

            # Write epoch-level rows
            writer.writerow({
                "timestamp": time.time(),
                "epoch": epoch,
                "batch": "",
                "split": "train_epoch",
                "loss": avg_train_loss,
                "n_pairs": len(train_pos) + len(train_neg),
                "batch_size": batch_size,
                "dim_model": dim_model,
                "n_layers": n_layers,
                "n_head": n_head,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "lr": optimizer.param_groups[0]['lr'],
                "margin": criterion.margin
            })
            
            writer.writerow({
                "timestamp": time.time(),
                "epoch": epoch,
                "batch": "",
                "split": "val_epoch",
                "loss": avg_val_loss,
                "n_pairs": len(val_pos) + len(val_neg),
                "batch_size": batch_size,
                "dim_model": dim_model,
                "n_layers": n_layers,
                "n_head": n_head,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "lr": optimizer.param_groups[0]['lr'],
                "margin": criterion.margin
            })
            f.flush()

            # Step scheduler
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save best model checkpoint
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
                        'dim_model': dim_model,
                        'n_layers': n_layers,
                        'n_head': n_head,
                        'dim_feedforward': dim_feedforward,
                        'dropout': dropout,
                        'margin': criterion.margin,
                    }
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

    # Save final model
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'hyperparameters': {
            'dim_model': dim_model,
            'n_layers': n_layers,
            'n_head': n_head,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'margin': criterion.margin,
        }
    }, final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Loss log written to: {log_csv_path}")
    print(f"Best model (val_loss={best_val_loss:.6f}) saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
