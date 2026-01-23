"""Training functions for the spectrum encoder model."""

import os
import sys
import csv
import time
from datetime import datetime
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from .models import CustomSpectrumTransformerEncoder, ContrastiveLoss
from .dataset import SpectraDataset
from .dataloader import create_dataloader
from .evaluator import evaluate_contrastive
from .scheduler import CosineWarmupScheduler
from .visualization import (
    compute_embedding_distances,
    plot_embedding_distance_distribution,
    plot_roc_curve
)
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
    
    try:
        train_metadata = pd.read_parquet(train_metadata_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Train metadata file not found: {train_metadata_file}")
    except Exception as e:
        raise RuntimeError(f"Error loading train metadata: {e}") from e
    
    try:
        test_metadata = pd.read_parquet(test_metadata_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test metadata file not found: {test_metadata_file}")
    except Exception as e:
        raise RuntimeError(f"Error loading test metadata: {e}") from e
    
    print(f"Train metadata: {len(train_metadata):,} rows")
    print(f"Test metadata: {len(test_metadata):,} rows")
    
    # Check if metadata has 'scan' column (required for proper linking)
    if config.REQUIRED_METADATA_COLUMN not in train_metadata.columns:
        raise ValueError(
            f"Metadata must have a '{config.REQUIRED_METADATA_COLUMN}' column to link pairs to MGF positions!\n"
            "Run the fixed metadata extraction code in read.ipynb to create metadata with scan numbers."
        )
    
    # Load pairs (these are indices into the metadata DataFrames)
    try:
        train_pos_pairs = np.load(filenames_pairs_pos[0]).astype(int)
    except FileNotFoundError:
        raise FileNotFoundError(f"Train positive pairs file not found: {filenames_pairs_pos[0]}")
    except Exception as e:
        raise RuntimeError(f"Error loading train positive pairs: {e}") from e
    
    try:
        val_pos_pairs = np.load(filenames_pairs_pos[1]).astype(int)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test positive pairs file not found: {filenames_pairs_pos[1]}")
    except Exception as e:
        raise RuntimeError(f"Error loading test positive pairs: {e}") from e
    
    try:
        train_neg_pairs = np.load(filenames_pairs_neg[0]).astype(int)
    except FileNotFoundError:
        raise FileNotFoundError(f"Train negative pairs file not found: {filenames_pairs_neg[0]}")
    except Exception as e:
        raise RuntimeError(f"Error loading train negative pairs: {e}") from e
    
    try:
        val_neg_pairs = np.load(filenames_pairs_neg[1]).astype(int)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test negative pairs file not found: {filenames_pairs_neg[1]}")
    except Exception as e:
        raise RuntimeError(f"Error loading test negative pairs: {e}") from e
    
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
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    encoder = encoder.to(device)

    optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = ContrastiveLoss(margin=args.margin, label_certainty=args.label_certainty)
    
    # VERIFY: Print the actual loss parameters being used
    print(f"\n⚠️  IMPORTANT: Using contrastive loss with:")
    print(f"   Margin: {criterion.margin} (config default: {config.CONTRASTIVE_MARGIN})")
    print(f"   Label certainty: {criterion.label_certainty} (config default: {config.LOSS_LABEL_CERTAINTY})\n")
    
    # Learning rate scheduler with warmup
    # CosineWarmupScheduler steps per batch, not per epoch
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        cosine_schedule_period_iters=args.cosine_schedule_iters,
    )
    
    print(f"📈 Learning rate schedule:")
    print(f"   Peak LR: {args.learning_rate}")
    print(f"   Warmup iterations: {args.warmup_iters} batches")
    print(f"   Cosine decay period: {args.cosine_schedule_iters} batches")
    print(f"   Note: Scheduler steps after each batch, not epoch\n")

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
    
    try:
        base_dataset = SpectraDataset(mgf_path, scan_nrs=mgf_scan_numbers)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"MGF file not found: {mgf_path}") from e
    except RuntimeError as e:
        raise RuntimeError(f"Error loading dataset from MGF file: {e}") from e
    
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
        # Show progress bar if original stdout is a terminal (not redirected/piped)
        # Write to original stdout to bypass TeeOutput logging
        show_progress = hasattr(sys.__stdout__, 'isatty') and sys.__stdout__.isatty()
        for a, b in tqdm(pairs, desc=desc, unit="pairs", 
                        disable=(not show_progress or len(pairs) > 1000000), 
                        mininterval=0.5, dynamic_ncols=True, file=sys.__stdout__):
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

    try:
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
                
                # Store last batch for visualization
                last_batch_data = None

                t0 = time.time()
                
                # Show progress bar if original stdout is a terminal (not redirected/piped)
                # Write to original stdout to bypass TeeOutput logging
                show_progress = hasattr(sys.__stdout__, 'isatty') and sys.__stdout__.isatty()
                batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", unit="batch", 
                                disable=not show_progress, mininterval=0.5, dynamic_ncols=True, file=sys.__stdout__)

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

                    # Use global token (best performance so far)
                    emb1 = emb1_full[:, 0, :]
                    emb2 = emb2_full[:, 0, :]
                    
                    # Debug: Check embedding statistics on first batch
                    if batch_idx == 1 and epoch == 1:
                        print(f"\n🔍 Embedding diagnostics (first batch):")
                        print(f"   Embedding shape: {emb1.shape}")
                        print(f"   Emb1 - mean: {emb1.mean():.4f}, std: {emb1.std():.4f}, "
                              f"norm: {torch.norm(emb1, dim=1).mean():.4f}")
                        print(f"   Emb2 - mean: {emb2.mean():.4f}, std: {emb2.std():.4f}, "
                              f"norm: {torch.norm(emb2, dim=1).mean():.4f}")
                        print(f"   Full output shape: {emb1_full.shape}")
                        print(f"   Using GLOBAL TOKEN (best performance so far)")
                        print(f"   Issue: Model not learning discriminative features from peaks")
                        
                        # Check cosine similarity for positive vs negative pairs
                        pos_mask = labels == 1
                        neg_mask = labels == 0
                        
                        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                            pos_emb1 = emb1[pos_mask]
                            pos_emb2 = emb2[pos_mask]
                            neg_emb1 = emb1[neg_mask]
                            neg_emb2 = emb2[neg_mask]
                            
                            pos_cosine = torch.nn.functional.cosine_similarity(pos_emb1, pos_emb2).mean()
                            neg_cosine = torch.nn.functional.cosine_similarity(neg_emb1, neg_emb2).mean()
                            
                            print(f"\n   Cosine similarity (should differ for pos/neg):")
                            print(f"   Positive pairs: {pos_cosine:.4f} (should be HIGH ~1.0)")
                            print(f"   Negative pairs: {neg_cosine:.4f} (should be LOW ~0.0)")
                            print(f"   Difference: {(pos_cosine - neg_cosine):.4f} (>0.1 is good)")
                        
                        # Check if embeddings are diverse (not collapsed to single point)
                        emb_all = torch.cat([emb1, emb2], dim=0)
                        pairwise_cosine = torch.mm(
                            torch.nn.functional.normalize(emb_all, dim=1),
                            torch.nn.functional.normalize(emb_all, dim=1).T
                        )
                        # Exclude diagonal (self-similarity)
                        off_diagonal = pairwise_cosine[~torch.eye(len(emb_all), dtype=bool, device=device)]
                        
                        print(f"\n   Embedding diversity check:")
                        print(f"   Mean pairwise cosine similarity: {off_diagonal.mean():.4f}")
                        print(f"   (Should be < 0.5 for diverse embeddings, ~1.0 means collapsed)\n")

                    loss = criterion(emb1, emb2, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Debug: Check gradients on first batch
                    if batch_idx == 1 and epoch == 1:
                        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), float('inf'))
                        print(f"   Gradient norm: {grad_norm:.4f}")
                        # Check if embeddings have gradients
                        if emb1.grad is not None:
                            print(f"   Emb1 grad norm: {emb1.grad.norm():.4f}")
                        print()
                    
                    optimizer.step()
                    
                    # Step the scheduler after each batch (warmup + cosine decay)
                    scheduler.step()

                    loss_value = float(loss.item())
                    epoch_train_loss += loss_value
                    n_train_batches += 1
                    global_step += 1
                    
                    # Save last batch data for visualization (detach from graph)
                    last_batch_data = {
                        'emb1': emb1.detach().cpu(),
                        'emb2': emb2.detach().cpu(),
                        'labels': labels.detach().cpu()
                    }
                    
                    # Track embedding distances every 100 batches to debug learning
                    if batch_idx % 100 == 0:
                        with torch.no_grad():
                            # Use raw embeddings (no normalization, matching loss function)
                            distances = torch.nn.functional.pairwise_distance(emb1, emb2)
                            
                            # Also track embedding norms to detect collapse
                            emb1_norm = torch.norm(emb1, dim=1).mean().item()
                            emb2_norm = torch.norm(emb2, dim=1).mean().item()
                            
                            pos_mask = labels == 1
                            neg_mask = labels == 0
                            if pos_mask.any():
                                pos_dist = distances[pos_mask].mean().item()
                            else:
                                pos_dist = float('nan')
                            if neg_mask.any():
                                neg_dist = distances[neg_mask].mean().item()
                            else:
                                neg_dist = float('nan')
                    
                    batch_pbar.set_postfix({
                        'loss': f'{loss_value:.4f}', 
                        'avg_loss': f'{epoch_train_loss/n_train_batches:.4f}',
                        'pos_d': f'{pos_dist:.2f}' if batch_idx % 100 == 0 and not np.isnan(pos_dist) else '',
                        'neg_d': f'{neg_dist:.2f}' if batch_idx % 100 == 0 and not np.isnan(neg_dist) else '',
                        'norm': f'{emb1_norm:.1f}' if batch_idx % 100 == 0 else ''
                    })

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
                
                # Print training loss for comparison
                print(f"  Training: avg_loss={avg_train_loss:.4f} over {n_train_batches} batches")

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

                # Current learning rate (scheduler steps after each batch, not here)
                current_lr = optimizer.param_groups[0]['lr']
                
                # ========================================
                # Create visualization plots for this epoch
                # ========================================
                if last_batch_data is not None:
                    try:
                        # Create plots directory
                        plots_dir = os.path.join(os.path.dirname(best_model_path), '..', 'results', 'training_plots')
                        os.makedirs(plots_dir, exist_ok=True)
                        
                        # Generate timestamp for unique filenames
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        print(f"\n  📊 Creating visualization plots for epoch {epoch} (last training batch)...")
                        
                        # Compute embedding distances on last training batch
                        with torch.no_grad():
                            emb1_batch = last_batch_data['emb1']
                            emb2_batch = last_batch_data['emb2']
                            labels_batch = last_batch_data['labels']
                            
                            # Compute distances
                            distances = torch.nn.functional.pairwise_distance(emb1_batch, emb2_batch).numpy()
                            labels_np = labels_batch.numpy()
                            
                            # Split into positive and negative
                            positive_distances = distances[labels_np == 1]
                            negative_distances = distances[labels_np == 0]
                        
                        # Create distance distribution plot
                        fig1, ax1 = plot_embedding_distance_distribution(
                            positive_distances,
                            negative_distances,
                            fdr=0.01,
                            save_path=os.path.join(plots_dir, f'epoch_{epoch:03d}_{timestamp_str}_distances.png'),
                            title=f'Epoch {epoch}/{n_epochs} - Last Training Batch\n'
                                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}\n'
                                  f'Batch: {len(positive_distances)} pos pairs, {len(negative_distances)} neg pairs',
                            figsize=(12, 7)
                        )
                        plt.close(fig1)
                        
                        # Create ROC curve
                        fig2, ax2, auc_score = plot_roc_curve(
                            positive_distances,
                            negative_distances,
                            save_path=os.path.join(plots_dir, f'epoch_{epoch:03d}_{timestamp_str}_roc.png')
                        )
                        plt.close(fig2)
                        
                        print(f"  ✓ Plots saved to {plots_dir}")
                        print(f"    - AUC: {auc_score:.4f}")
                        print(f"    - Positive distance mean: {np.mean(positive_distances):.4f}")
                        print(f"    - Negative distance mean: {np.mean(negative_distances):.4f}")
                        
                    except Exception as e:
                        print(f"  ⚠️  Warning: Failed to create visualization plots: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Save best model checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    try:
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
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save best model: {e}")
                
                dt = time.time() - t0
                print(
                    f"Epoch {epoch}/{n_epochs} | "
                    f"train_loss={avg_train_loss:.6f} | "
                    f"val_loss={avg_val_loss:.6f} | "
                    f"lr={current_lr:.2e} | "
                    f"time={dt:.1f}s"
                )

    except IOError as e:
        raise IOError(f"Error writing to log file {log_csv_path}: {e}") from e
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user. Saving current model...")
        raise
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        raise

    # Save final model
    try:
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
    except Exception as e:
        print(f"⚠️  Warning: Failed to save final model: {e}")
        print(f"    Best model is still saved at: {best_model_path}")
    
    print(f"\nTraining complete!")
    print(f"Loss log written to: {log_csv_path}")
    print(f"Best model (val_loss={best_val_loss:.6f}) saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")