"""GLEAMS training script - main entry point for spectrum encoder training."""

import os
import sys
import time
from typing import List

import pandas as pd

from nn import nn_torch as nn
from metadata import metadata
import config  # Old config with dependencies



# Configuration dictionaries for legacy functions
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
    import logging
    import tempfile
    
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
    metadata_df = pd.DataFrame({'filename': peak_files})
    metadata_df['dataset'] = 'GLEAMS'
    metadata_df.to_parquet(metadata_filename, index=False)

    embedder_config = {
        'num_precursor_features': config.num_precursor_features,
        'num_fragment_features': config.num_fragment_features,
        'num_ref_spectra_features': config.num_ref_spectra,
        'lr': config.lr
    }

    # Run the embedding
    model_filename = 'GLEAMS/data/gleams.pth'
    nn.embed(
        metadata_filename,
        model_filename,
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


# Import refactored transformer encoder module
from transformer_encoder import (
    parse_args,
    train_model,
)
from transformer_encoder import config  # New modular config



def run_preprocessing():
    """Regenerate positive and negative pairs from the updated metadata files."""
    train_file = 'GLEAMS/data/train_metadata.parquet'
    test_file = 'GLEAMS/data/test_metadata.parquet'
    
    print("Regenerating pairs for train metadata...")
    metadata.generate_pairs_positive(train_file, config.charges)
    metadata.generate_pairs_negative(
        train_file, config.charges, config.pair_mz_tolerance, 
        config.negative_pair_fragment_tolerance, 
        config.negative_pair_matching_fragments_threshold
    )
    
    print("\nRegenerating pairs for test metadata...")
    metadata.generate_pairs_positive(test_file, config.charges)
    metadata.generate_pairs_negative(
        test_file, config.charges, config.pair_mz_tolerance, 
        config.negative_pair_fragment_tolerance, 
        config.negative_pair_matching_fragments_threshold
    )
    
    print("\n✓ Pair regeneration complete!")


class TeeOutput:
    """Utility class to write output to both stdout and a file."""
    
    def __init__(self, *files):
        self.files = files
        
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
            
    def flush(self):
        for f in self.files:
            f.flush()
    
    def isatty(self):
        """Return False to indicate this is not a TTY (disables tqdm progress bars)."""
        return False


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging to both console and file
    log_file = args.log_txt
    
    try:
        log_fileobj = open(log_file, 'a')
    except IOError as e:
        print(f"Warning: Could not open log file {log_file}: {e}")
        print("Continuing without file logging...")
        log_fileobj = None
    
    # Redirect stdout to both console and file
    if log_fileobj:
        sys.stdout = TeeOutput(sys.__stdout__, log_fileobj)
        sys.stderr = TeeOutput(sys.__stderr__, log_fileobj)
    
    print("="*80)
    print(f"Starting new training run - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command-line arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*80)
    
    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n\n❌ File not found: {e}")
        print("Please check that all required files exist and paths are correct.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n\n❌ Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore stdout/stderr and close log file
        if log_fileobj:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_fileobj.close()






