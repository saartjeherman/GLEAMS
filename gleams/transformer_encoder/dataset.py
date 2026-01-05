"""Dataset classes for loading and processing mass spectrometry data."""

import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from tqdm import tqdm

from ms_io import mgf_io
from . import config


class SpectraDataset(Dataset):
    """Dataset for loading mass spectra from MGF files."""
    
    def __init__(self, mgf_path: str, scan_nrs: Optional[List[int]] = None):
        """
        Load spectra dataset with optimized memory usage and progress monitoring.
        
        Args:
            mgf_path: Path to MGF file
            scan_nrs: Optional list of scan numbers to load (if None, loads all)
        """
        print(f"Loading spectra from {mgf_path}...")
        
        # Warn about compressed file performance
        if mgf_path.endswith(config.COMPRESSED_EXTENSIONS):
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
        
        for spectrum in tqdm(spectra_generator, desc="Loading spectra", total=total, unit="spectra", file=sys.__stdout__):
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

    def __len__(self) -> int:
        return self.n_spectra

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a spectrum by index.
        
        Args:
            idx: Index of spectrum
            
        Returns:
            Tuple of (mz, intensity, precursor_mz, charge) tensors
        """
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


class SpectrumPairDataset(Dataset):
    """Dataset for spectrum pairs with similarity labels."""
    
    def __init__(self, dataset: SpectraDataset, pos_indices: List[Tuple[int, int]], 
                 neg_indices: List[Tuple[int, int]]):
        """
        Create pair dataset from base dataset.
        
        Args:
            dataset: Base SpectraDataset
            pos_indices: List of (i, j) index pairs for similar spectra
            neg_indices: List of (i, j) index pairs for dissimilar spectra
        """
        self.dataset = dataset
        self.pairs = [(i, j, 1) for i, j in pos_indices] + [(i, j, 0) for i, j in neg_indices]
        print(f"Total pairs: {len(self.pairs):,}")

    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        """
        Get a spectrum pair with label.
        
        Args:
            idx: Index of pair
            
        Returns:
            Tuple of (spec1, spec2, label)
        """
        i, j, label = self.pairs[idx]
        spec1 = self.dataset[i]
        spec2 = self.dataset[j]

        return spec1, spec2, torch.tensor(label, dtype=torch.float32)
