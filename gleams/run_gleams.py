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

from metadata import metadata
from feature import feature, encoder
#from nn import nn
import config
from ms_io import mgf_io
from spectrum_pairs import (
    SpectrumPairDataset, collate_pairs, preencode_mgf_to_npzs,
)
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


def train_model(run_dir: str):
    """Train one model and write all artifacts (log, loss CSV, checkpoints) into `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)

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
    train_npz_path = '/home/saartje/Desktop/Research/GLEAMS/data/train_spectra.npz'
    val_npz_path   = '/home/saartje/Desktop/Research/GLEAMS/data/test_spectra.npz'
    log_csv_path = os.path.join(run_dir, 'loss_log.csv')

    batch_size = 128
    num_workers = 2

    # Cap pairs per (split, polarity) for fast smoke-testing. Set to None to use all.
    max_train_pairs_per_class = 20000
    max_val_pairs_per_class   = 4000

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

    def _subsample(pairs: np.ndarray, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
        if cap is None or len(pairs) <= cap:
            return pairs
        idx = rng.choice(len(pairs), cap, replace=False)
        return pairs[idx]

    rng = np.random.default_rng(seed=0)
    train_pos_pairs = _subsample(train_pos_pairs, max_train_pairs_per_class, rng)
    train_neg_pairs = _subsample(train_neg_pairs, max_train_pairs_per_class, rng)
    val_pos_pairs   = _subsample(val_pos_pairs,   max_val_pairs_per_class,   rng)
    val_neg_pairs   = _subsample(val_neg_pairs,   max_val_pairs_per_class,   rng)

    print(f"Train pairs (after cap): {len(train_pos_pairs):,} positive, {len(train_neg_pairs):,} negative")
    print(f"Val pairs (after cap):   {len(val_pos_pairs):,} positive, {len(val_neg_pairs):,} negative")

    # -----------------------------
    # Pre-encode MGF spectra → packed .npz (one row per metadata row).
    # Skips silently if the .npz already exists; pass overwrite=True to rebuild.
    # -----------------------------
    print("Pre-encoding spectra from MGF (one pass, one-time cost)...")
    preencode_mgf_to_npzs(mgf_path, [
        (train_metadata, train_npz_path),
        (test_metadata,  val_npz_path),
    ])

    # -----------------------------
    # Datasets + DataLoaders. Pairs index directly into the packed arrays.
    # -----------------------------
    # Cap each spectrum to its top-K most intense peaks. Bounds attention cost
    # so one outlier doesn't pad the whole batch to thousands of tokens.
    max_peaks = 150
    train_dataset = SpectrumPairDataset(
        train_npz_path, train_pos_pairs, train_neg_pairs, max_peaks=max_peaks)
    val_dataset = SpectrumPairDataset(
        val_npz_path,   val_pos_pairs,   val_neg_pairs,   max_peaks=max_peaks)
    print(f"Train dataset: {len(train_dataset):,} pairs")
    print(f"Val dataset:   {len(val_dataset):,} pairs")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pairs, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pairs, num_workers=num_workers, pin_memory=True,
    )

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
    # Logging setup
    # -----------------------------
    log_batch_level = False  # set True if you want per-batch rows in CSV too

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
        best_model_path = os.path.join(run_dir, 'best_model.pt')
        final_model_path = os.path.join(run_dir, 'final_model.pt')

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
                "n_pairs": len(train_pos_pairs) + len(train_neg_pairs),
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
                "n_pairs": len(val_pos_pairs) + len(val_neg_pairs),
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

    
    

class _Tee:
    """Fan stdout/stderr writes out to multiple file-like objects."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    import sys
    import traceback

    results_root = '/home/saartje/Desktop/Research/GLEAMS/results'
    run_id = time.strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'training.log')

    with open(log_path, 'a') as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

        banner = "=" * 80
        print(banner)
        print(f"Starting new training run - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Run directory: {run_dir}")
        print(banner)

        try:
            train_model(run_dir)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__






