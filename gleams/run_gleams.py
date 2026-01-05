import os
import logging
import tempfile
from typing import List
import polars as pl
import depthcharge as dc
from depthcharge.transformers import SpectrumTransformerEncoder
from depthcharge.encoders import FloatEncoder
#from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder
#import depthcharge.masses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import re
from numba.typed import List as NumbaList

from nn import nn_torch as nn
from metadata import metadata
from feature import feature, encoder
#from nn import nn
import config
from ms_io import mgf_io
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import List, Tuple
import csv
import time
from tqdm import tqdm



precursor_encoding = {
    'num_bits_mz': config.num_bits_precursor_mz,
    'mz_min': config.precursor_mz_min,
    'mz_max': config.precursor_mz_max,
    'num_bits_mass': config.num_bits_precursor_mass,
    'mass_min': config.precursor_mass_min,
    'mass_max': config.precursor_mass_max,
    'charge_max': config.precursor_charge_max
}

fragment_encoding = {
    'min_mz': config.fragment_mz_min,
    'max_mz': config.fragment_mz_max,
    'bin_size': config.bin_size
}

reference_encoding = {
    'filename': config.ref_spectra_filename,
    'preprocessing': {
        'mz_min': config.fragment_mz_min,
        'mz_max': config.fragment_mz_max,
        'min_peaks': config.min_peaks,
        'min_mz_range': config.min_mz_range,
        'remove_precursor_tolerance': config.remove_precursor_tolerance,
        'min_intensity': config.min_intensity,
        'max_peaks_used': config.max_peaks_used,
        'scaling': config.scaling
    },
    'fragment_mz_tol': config.fragment_mz_tol,
    'num_ref_spectra': config.num_ref_spectra
}

def run_gleams_embed(peak_files: List[str], embed_name: str = 'GLEAMS_embed') -> None:
    """
    Run the 'gleams embed' command programmatically.

    Parameters:
    - peak_files: List of input peak files (e.g., mzML, MGF).
    - embed_name: Name for the output files.
    """
    logger = logging.getLogger('gleams')
    logger.info('Starting GLEAMS embedding...')

    if not peak_files:
        raise ValueError('No input peak files specified.')

    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    metadata_filename = os.path.join(temp_dir, f'{embed_name}.parquet')
    embed_dir = os.path.join(temp_dir, 'embed')
    os.mkdir(embed_dir)

    # Create a metadata file with the file names
    metadata = pd.DataFrame({'filename': peak_files})
    metadata['dataset'] = 'GLEAMS'
    metadata.to_parquet(metadata_filename, index=False)

    # Prepare embedding configuration
    

    embedder_config = {
        'num_precursor_features': config.num_precursor_features,
        'num_fragment_features': config.num_fragment_features,
        'num_ref_spectra_features': config.num_ref_spectra,
        'lr': config.lr
    }

    # Run the embedding
    model_filename = '/home/saartje/Desktop/Research/GLEAMS/data/gleams.pth' #config.model_filename  
    nn.embed(
        metadata_filename,
        model_filename,
        #config.model_filename,
        f'{embed_name}.npy',
        embed_dir,
        precursor_encoding,
        fragment_encoding,
        reference_encoding,
        embedder_config,
        config.batch_size,
        config.charges
    )

    logger.info('GLEAMS embedding completed successfully.')




class SpectraDataset(Dataset):
    def __init__(self, mgf_path, scan_nrs=None):
        """Load spectra dataset with optimized memory usage and progress monitoring.
        
        Args:
            mgf_path: Path to MGF file
            scan_nrs: Optional list of scan numbers to load (if None, loads all)
        """
        print(f"Loading spectra from {mgf_path}...")
        
        # Warn about compressed file performance
        if mgf_path.endswith(('.xz', '.gz', '.bz2')):
            print(f"⚠️  WARNING: Compressed file detected (.{mgf_path.split('.')[-1]})")
            print("   Decompression + filtering is VERY slow for large files.")
            print("   Consider decompressing first: unxz file.mgf.xz")
            if scan_nrs is not None:
                print(f"   Must scan through entire file to find {len(scan_nrs):,} spectra...\n")
        
        # Pre-allocate lists for efficient storage
        mz_list = []
        intensity_list = []
        precursor_mz_list = []
        precursor_charge_list = []
        identifier_list = []
        
        # Single-pass loading with progress bar
        max_peaks = 0
        spectra_generator = mgf_io.get_spectra(mgf_path, scan_nrs)
        
        # If scan_nrs is provided, we know the total count to LOAD (not scan)
        total = len(scan_nrs) if scan_nrs is not None else None
        
        import time as time_module
        start_time = time_module.time()
        
        # Track loaded scans to detect missing ones
        requested_scans = set(scan_nrs) if scan_nrs is not None else None
        loaded_count = 0
        
        for spectrum in tqdm(spectra_generator, desc="Loading spectra", total=total, unit="spectra"):
            # Extract only necessary data
            mz_list.append(spectrum.mz)
            intensity_list.append(spectrum.intensity)
            precursor_mz_list.append(spectrum.precursor_mz)
            precursor_charge_list.append(spectrum.precursor_charge)
            identifier_list.append(int(spectrum.identifier))
            
            # Update max_peaks in single pass
            max_peaks = max(max_peaks, len(spectrum.mz))
            
            loaded_count += 1
            if requested_scans is not None:
                requested_scans.discard(int(spectrum.identifier))
        
        load_time = time_module.time() - start_time
        
        # Warn about missing scans
        if requested_scans is not None and len(requested_scans) > 0:
            print(f"⚠️  WARNING: {len(requested_scans):,} requested scans not found in MGF file")
            print(f"   Loaded {loaded_count:,} out of {len(scan_nrs):,} requested spectra")
        
        # Store as efficient numpy arrays or lists
        self.mz_data = mz_list
        self.intensity_data = intensity_list
        self.precursor_mz_data = np.array(precursor_mz_list, dtype=np.float32)
        self.precursor_charge_data = np.array(precursor_charge_list, dtype=np.float32)
        self.identifiers = identifier_list  # Needed for pair remapping
        
        self.n_spectra = len(mz_list)
        self.max_peaks = max_peaks
        
        print(f"✓ Loaded {self.n_spectra:,} spectra with max {self.max_peaks} peaks in {load_time/60:.1f} min")

    def __len__(self):
        return self.n_spectra

    def __getitem__(self, idx):
        # Pre-allocate zero-padded arrays
        mz = np.zeros(self.max_peaks, dtype=np.float32)
        intensity = np.zeros(self.max_peaks, dtype=np.float32)
        
        # Fill with actual data
        mz_vals = self.mz_data[idx]
        intensity_vals = self.intensity_data[idx]
        
        mz[:len(mz_vals)] = mz_vals
        intensity[:len(intensity_vals)] = intensity_vals

        pepmass = np.array([self.precursor_mz_data[idx]], dtype=np.float32)
        charge = np.array([self.precursor_charge_data[idx]], dtype=np.float32)

        return torch.tensor(mz), torch.tensor(intensity), torch.tensor(pepmass), torch.tensor(charge)


# Define a simple contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 1 if similar, 0 if not
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

class SpectrumPairDataset(Dataset):
    def __init__(self, dataset, pos_indices, neg_indices):
        self.dataset = dataset
        self.pairs = [(i, j, 1) for i, j in pos_indices] + [(i, j, 0) for i, j in neg_indices]
        print(f"Total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        spec1 = self.dataset[i]  # now returns (mz1, int1, pepmass1, charge1)
        spec2 = self.dataset[j]  # now returns (mz2, int2, pepmass2, charge2)

        return spec1, spec2, torch.tensor(label, dtype=torch.float32)
    


def create_dataloader(
    base_dataset: Dataset,
    pos_pairs: List[Tuple[int, int]],
    neg_pairs: List[Tuple[int, int]],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a base dataset and pre-defined positive
    and negative index pairs.
    """
    pair_dataset = SpectrumPairDataset(
        base_dataset,
        pos_pairs,
        neg_pairs
    )
    
    print(f"Creating DataLoader with {len(pair_dataset):,} total pairs, batch_size={batch_size}")
    print(f"Expected batches per epoch: {len(pair_dataset) // batch_size}")

    #dataloader = DataLoader(
    #    pair_dataset,
    #    batch_size=batch_size,
    #    shuffle=shuffle
    #)
    dataloader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      
        pin_memory=True     # ← Add this for CUDA
    )

    return dataloader


class CustomSpectrumTransformerEncoder(SpectrumTransformerEncoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
        #dim_intensity: Optional[int] = None,
        #peak_encoder: Optional[Any] = None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            #dim_intensity=dim_intensity,
            #peak_encoder=peak_encoder
        )
        # Create float encoder as part of the model so it moves to device with model
        self.float_encoder = FloatEncoder(d_model=d_model)
        
    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Custom implementation of the global token hook.
        Modify this function to encode additional information
        into the global spectrum token.

        Example input (from `kwargs` or `args`):
        - kwargs["precursor_mass"]
        - kwargs["charge"]
        - args[0], etc.
        """

        # Example: Just return a learned linear projection of precursor mass if provided
        precursor_mass = kwargs.get("pepmass", None)
        charge = kwargs.get("charge", None)

        # Error handling for missing precursor mass or charge
        if precursor_mass is None:
            raise ValueError("Missing precursor mass: precursor_mass is required but is None.")

        if charge is None:
            raise ValueError("Missing charge: charge is required but is None.")

        # Encode precursor mass and charge using the model's float encoder
        precursor_mass_encoded = self.float_encoder(precursor_mass.float())  # Ensure it's a 2D tensor
        charge_encoded = self.float_encoder(charge.float())  # Ensure it's a 2D tensor

        # Combine the two encodings
        combined_encoding = precursor_mass_encoded + charge_encoded

        # Ensure it's the correct shape: (batch_size, d_model)
        combined_encoding = combined_encoding.squeeze(1)

        return combined_encoding

    

def evaluate_contrastive(
    encoder,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """Compute mean validation loss over a dataloader."""
    encoder.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        # Add progress bar for validation
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
            loss_value = float(loss.item())
            total_loss += loss_value
            n_batches += 1
            
            # Update progress bar with running average
            val_pbar.set_postfix({'val_loss': f'{total_loss/n_batches:.4f}'})

    return total_loss / max(n_batches, 1)


def train_model(op_kwargs):
    # -----------------------------
    # Paths
    # -----------------------------
    train_metadata_file = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata.parquet'
    test_metadata_file = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata.parquet'
    
    filenames_pairs_pos = [
        '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata_pairs_pos_2.npy',
        '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata_pairs_pos_2.npy'
    ]
    filenames_pairs_neg = [
        '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata_pairs_neg_2.npy',
        '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata_pairs_neg_2.npy'
    ]

    mgf_path = '/home/saartje/Desktop/Research/GLEAMS/data/massivekb_82c0124b.mgf'  # Replace with your actual MGF file path
    log_csv_path = '/home/saartje/Desktop/Research/GLEAMS/data/loss_log.csv'

    # -----------------------------
    # Load metadata and pairs
    # -----------------------------
    # IMPORTANT: Pairs contain ROW INDICES from the metadata files
    print("Loading metadata files...")
    train_metadata = pd.read_parquet(train_metadata_file)
    test_metadata = pd.read_parquet(test_metadata_file)
    
    print(f"Train metadata: {len(train_metadata):,} rows")
    print(f"Test metadata: {len(test_metadata):,} rows")
    
    # Check if metadata has 'scan' column (required for proper linking)
    if 'scan' not in train_metadata.columns:
        raise ValueError(
            "Metadata must have a 'scan' column to link pairs to MGF positions!\n"
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
    # Model hyperparameters
    # -----------------------------
    dim_model: int = 256 #512
    n_head: int = 4 #4
    dim_feedforward: int = 512 #1024
    n_layers: int = 2 #4
    #dropout: float = 0.1 #0.0
    dropout: float = 0.3  # From 0.1

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

    #optimizer = optim.Adam(encoder.parameters(), lr=1e-4) #lr=1e-4)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = ContrastiveLoss(margin=2.0) #1.0)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # -----------------------------
    # Build ONE base dataset used by both train and val
    # -----------------------------
    # Get unique metadata row indices needed from pairs
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
    
    # The dataset loaded spectra by MGF scan numbers
    # We need to map: metadata_row_index -> dataset_index
    # base_dataset.identifiers contains the MGF scan numbers that were loaded
    scan_to_dataset_idx = {int(scan): i for i, scan in enumerate(base_dataset.identifiers)}
    
    # Create the full mapping: metadata_row_idx -> dataset_idx
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
        for a, b in tqdm(pairs, desc=desc, unit="pairs", disable=len(pairs) > 1000000):
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
    batch_size = 64  # Reduced to fit in GPU memory

    # Use only first 50K pairs for speed test
    train_pos = train_pos[:50000]
    train_neg = train_neg[:50000]

    # Use only first 10K pairs for speed test
    val_pos = val_pos[:10000]
    val_neg = val_neg[:10000]


    train_loader = create_dataloader(base_dataset, train_pos, train_neg, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(base_dataset, val_pos,   val_neg,   batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Logging setup
    # -----------------------------
    log_batch_level = False  # set True if you want per-batch rows in CSV too

    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    write_header = not os.path.exists(log_csv_path)

    with open(log_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "epoch",
                "batch",
                "split",
                "loss",
                "n_pairs",
                "batch_size",
                "dim_model",
                "n_layers",
                "n_head",
                "dim_feedforward",
                "dropout",
                "lr",
                "margin",
            ],
        )
        if write_header:
            writer.writeheader()

        # -----------------------------
        # Training loop
        # -----------------------------
        n_epochs = 10  # adjust
        global_step = 0
        best_val_loss = float('inf')
        best_model_path = 'GLEAMS/models/best_model.pt'
        final_model_path = 'GLEAMS/models/final_model.pt'
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            encoder.train()
            epoch_train_loss = 0.0
            n_train_batches = 0

            t0 = time.time()
            
            # Progress bar for batches
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
                
                # Update progress bar with current loss
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

            # Write epoch-level rows (recommended for visualization)
            writer.writerow({
                "timestamp": time.time(),
                "epoch": epoch,
                "batch": "",         # blank for epoch-level
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

            # Step the learning rate scheduler based on validation loss
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


    
    
    #nn.train_nn(filename_model, 
    #            filenames_pairs_pos, 
    #            filenames_pairs_neg, 
    #            embedder_config=config.embedder_config,
    #            batch_size=config.batch_size,
    #            num_epochs=2,        #config.num_epochs,
    #            steps_per_epoch=config.steps_per_epoch,
    #            max_num_pairs_train=config.max_num_pairs_train,
    #            max_num_pairs_val=config.max_num_pairs_val)


    
    
def run_preprocessing():
    """Regenerate positive and negative pairs from the updated metadata files."""
    train_file = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata.parquet'
    test_file = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata.parquet'
    
    print("Regenerating pairs for train metadata...")
    metadata.generate_pairs_positive(train_file, config.charges)
    metadata.generate_pairs_negative(train_file, config.charges, config.pair_mz_tolerance, 
                                     config.negative_pair_fragment_tolerance, 
                                     config.negative_pair_matching_fragments_threshold)
    
    print("\nRegenerating pairs for test metadata...")
    metadata.generate_pairs_positive(test_file, config.charges)
    metadata.generate_pairs_negative(test_file, config.charges, config.pair_mz_tolerance, 
                                     config.negative_pair_fragment_tolerance, 
                                     config.negative_pair_matching_fragments_threshold)
    
    print("\n✓ Pair regeneration complete!")


    #peaks_dir = '/home/saartje/Desktop/Research/GLEAMS/data/peaks'
    #metadata.download_massivekb_peaks(reduced_filename_data, peaks_dir)

    #output_file_feature = '/home/saartje/Desktop/Research/GLEAMS/data/feature_output.npz'
    #feature.convert_peaks_to_features(output_file, output_file_feature, precursor_encoding, 
                                        #fragment_encoding, reference_encoding, filter_scans=True)

    #read the parquet file
    #metadata_data = pd.read_parquet(output_file)
    #print(metadata_data.head())
    #print(metadata_data.columns)

    
    

if __name__ == "__main__":
    import sys
    
    # Setup logging to both console and file
    log_file = '/home/saartje/Desktop/Research/GLEAMS/data/training.log'
    
    # Open log file for appending
    log_fileobj = open(log_file, 'a')
    
    # Create a custom class that writes to both stdout and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Redirect stdout to both console and file
    sys.stdout = TeeOutput(sys.__stdout__, log_fileobj)
    sys.stderr = TeeOutput(sys.__stderr__, log_fileobj)
    
    print("="*80)
    print(f"Starting new training run - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Example usage
    #peak_files = ["/home/saartje/Desktop/Research/GLEAMS/gleams_reference_spectra_modified.mgf"]  # Replace with your actual file path(s)
    #run_gleams_embed(peak_files, embed_name="GLEAMS_embed")
    #run_preprocessing()  # Regenerate pairs from new metadata
    
    try:
        train_model(None)
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fileobj.close()


    #print(df.head())






