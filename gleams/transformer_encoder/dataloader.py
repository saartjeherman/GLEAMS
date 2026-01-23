"""DataLoader utilities for creating training and validation data loaders."""

from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

from .dataset import SpectrumPairDataset
from . import config


def create_dataloader(
    base_dataset: Dataset,
    pos_pairs: List[Tuple[int, int]],
    neg_pairs: List[Tuple[int, int]],
    batch_size: int = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a base dataset and pre-defined positive
    and negative index pairs.
    
    Args:
        base_dataset: Base spectrum dataset
        pos_pairs: List of (i, j) index pairs for similar spectra
        neg_pairs: List of (i, j) index pairs for dissimilar spectra
        batch_size: Batch size for training (defaults to config.BATCH_SIZE)
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader for training or validation
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    pair_dataset = SpectrumPairDataset(
        base_dataset,
        pos_pairs,
        neg_pairs
    )
    
    print(f"Creating DataLoader with {len(pair_dataset):,} total pairs, batch_size={batch_size}")
    print(f"Expected batches per epoch: {len(pair_dataset) // batch_size}")

    dataloader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return dataloader