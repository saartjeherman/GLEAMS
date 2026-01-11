"""GLEAMS training script - main entry point for spectrum encoder training."""

import os
import sys
import time
from typing import List

import pandas as pd

from transformer_encoder import config  # New modular config



# Import refactored transformer encoder module
from transformer_encoder import (
    parse_args,
    train_model,
)




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






