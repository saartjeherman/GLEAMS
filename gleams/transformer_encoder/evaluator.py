"""Evaluation functions for model validation."""

import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_contrastive(
    encoder,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """
    Compute mean validation loss over a dataloader.
    
    Args:
        encoder: Spectrum encoder model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        Average validation loss
    """
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    
    # Track statistics for debugging high validation loss
    max_loss = 0.0
    min_loss = float('inf')
    loss_values = []
    distances = []
    labels_all = []
    
    # Verify model is on correct device
    model_device = next(encoder.parameters()).device
    if model_device.type != device.type or (device.type == 'cuda' and model_device.index != device.index):
        print(f"⚠️  Warning: Model is on {model_device} but device argument is {device}")
    
    # Show progress bar if original stdout is a terminal (not redirected/piped)
    # Write to original stdout to bypass TeeOutput logging
    show_progress = hasattr(sys.__stdout__, 'isatty') and sys.__stdout__.isatty()

    try:
        with torch.no_grad():
            # Add progress bar for validation
            val_pbar = tqdm(dataloader, desc="Validation", unit="batch", 
                           disable=not show_progress, mininterval=0.5, dynamic_ncols=True, file=sys.__stdout__)
            
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

                # Calculate distances for analysis (no normalization, matching original CNN)
                euclidean_distance = torch.nn.functional.pairwise_distance(emb1, emb2)
                distances.extend(euclidean_distance.cpu().numpy().tolist())
                labels_all.extend(labels.cpu().numpy().tolist())

                loss = criterion(emb1, emb2, labels)
                loss_value = float(loss.item())
                total_loss += loss_value
                n_batches += 1
                
                # Track min/max for debugging
                max_loss = max(max_loss, loss_value)
                min_loss = min(min_loss, loss_value)
                loss_values.append(loss_value)
                
                # Update progress bar with running average
                val_pbar.set_postfix({'val_loss': f'{total_loss/n_batches:.4f}'})

        avg_loss = total_loss / max(n_batches, 1)
        
        # Print statistics to help debug high validation loss
        import numpy as np
        loss_array = np.array(loss_values)
        dist_array = np.array(distances)
        labels_array = np.array(labels_all)
        
        # Separate distances by label
        pos_distances = dist_array[labels_array == 1]
        neg_distances = dist_array[labels_array == 0]
        
        print(f"\n  Validation loss statistics:")
        print(f"    Mean: {avg_loss:.4f}")
        print(f"    Median: {np.median(loss_array):.4f}")
        print(f"    Std: {np.std(loss_array):.4f}")
        print(f"    Min: {min_loss:.4f}")
        print(f"    Max: {max_loss:.4f}")
        print(f"    95th percentile: {np.percentile(loss_array, 95):.4f}")
        
        print(f"\n  Label distribution:")
        print(f"    Positive pairs: {len(pos_distances):,} ({len(pos_distances)/len(labels_array)*100:.1f}%)")
        print(f"    Negative pairs: {len(neg_distances):,} ({len(neg_distances)/len(labels_array)*100:.1f}%)")
        
        print(f"\n  Embedding distances:")
        print(f"    Positive pairs - Mean: {np.mean(pos_distances):.4f}, Median: {np.median(pos_distances):.4f}")
        print(f"    Negative pairs - Mean: {np.mean(neg_distances):.4f}, Median: {np.median(neg_distances):.4f}")
        print(f"    Expected: Positive distances should be LOW, Negative distances should be HIGH (>margin={criterion.margin})")
        
        return avg_loss
    
    except Exception as e:
        raise RuntimeError(f"Error during validation: {e}") from e
