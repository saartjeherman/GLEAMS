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

    with torch.no_grad():
        # Add progress bar for validation
        val_pbar = tqdm(dataloader, desc="Validation", unit="batch", file=sys.__stdout__)
        
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
            loss_value = float(loss.item())
            total_loss += loss_value
            n_batches += 1
            
            # Update progress bar with running average
            val_pbar.set_postfix({'val_loss': f'{total_loss/n_batches:.4f}'})

    return total_loss / max(n_batches, 1)
