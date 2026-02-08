"""Configuration file for transformer encoder training.

This file contains all default values for model architecture, training hyperparameters,
and file paths. These can be overridden via command-line arguments.
"""

from pathlib import Path


# =============================================================================
# Directory Paths
# =============================================================================

# Base directories
DATA_DIR = Path('GLEAMS/data')
MODEL_DIR = Path('GLEAMS/models')
RESULTS_DIR = Path('GLEAMS/results')

# Input data files
TRAIN_METADATA_FILE = 'train_metadata.parquet'
TEST_METADATA_FILE = 'test_metadata.parquet'
MGF_FILE = 'massivekb_82c0124b.mgf'

# Pair files (charge state 2)
TRAIN_PAIRS_POS_FILE = 'train_metadata_pairs_pos_2.npy'
TRAIN_PAIRS_NEG_FILE = 'train_metadata_pairs_neg_2.npy'
TEST_PAIRS_POS_FILE = 'test_metadata_pairs_pos_2.npy'
TEST_PAIRS_NEG_FILE = 'test_metadata_pairs_neg_2.npy'

# Output files
LOG_CSV_FILE = 'loss_log.csv'
LOG_TXT_FILE = 'training.log'
BEST_MODEL_FILE = 'best_model.pt'
FINAL_MODEL_FILE = 'final_model.pt'


# =============================================================================
# Model Architecture
# =============================================================================

# Transformer encoder dimensions
DIM_MODEL = 256              # Model embedding dimension (increased from 64 for more capacity)
N_HEAD = 4                   # Number of attention heads
DIM_FEEDFORWARD = 512        # Feedforward network dimension
N_LAYERS = 2                 # Number of transformer layers (increased from 2)
DROPOUT = 0.1                # Dropout rate for regularization


# =============================================================================
# Training Hyperparameters
# =============================================================================

# Optimization
LEARNING_RATE = 5e-5         # Initial learning rate (peak LR after warmup)
WEIGHT_DECAY = 1e-5          # L2 regularization weight
BATCH_SIZE = 64              # Physical batch size (fits in memory)
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over N steps (effective batch = 64*4=256)
N_EPOCHS = 5                # Number of training epochs
GRADIENT_CLIP_NORM = 1.0     # Clip gradients to prevent explosion (recommended: 0.5-2.0)

# Loss function
CONTRASTIVE_MARGIN = 0.5     # Margin for contrastive loss (reduced for normalized embeddings)
LOSS_LABEL_CERTAINTY = 1.0   # Confidence in labels (1.0 = fully certain, <1.0 for noisy labels)

# Learning rate scheduler (CosineWarmupScheduler)
# The scheduler uses linear warmup followed by cosine decay, which is critical for
# stable transformer training. During warmup, LR increases from 0 to LEARNING_RATE.
WARMUP_ITERS = 250           # Number of iterations (steps/batches) for LR warmup
                             # Typical values: 500-2000 steps
                             # Should be ~5-10% of total training steps
COSINE_SCHEDULE_ITERS = 2500 # Total iterations for cosine decay period
                             # Should approximately equal total_batches * N_EPOCHS
                             # Example: 200 batches/epoch * 5 epochs = 1000 steps
                             
# Legacy scheduler settings (kept for backward compatibility, not used with CosineWarmupScheduler)
SCHEDULER_PATIENCE = 2       # [LEGACY] Epochs to wait before reducing LR (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.5       # [LEGACY] Factor to reduce LR by (ReduceLROnPlateau)
SCHEDULER_VERBOSE = True     # Print LR updates


# =============================================================================
# DataLoader Settings
# =============================================================================

NUM_WORKERS = 8              # Number of data loading workers (0 = single-threaded)
                             # 4-8 workers enable parallel data loading while GPU processes batches
                             # Increase this if GPU utilization is low during training
                             # Reduce to 0 if you see memory issues or "too many open files" errors
PIN_MEMORY = True            # Pin memory for faster GPU transfer
SHUFFLE_TRAIN = True         # Shuffle training data
SHUFFLE_VAL = False          # Don't shuffle validation data


# =============================================================================
# Logging Settings
# =============================================================================

LOG_BATCH_LEVEL = False      # Log individual batches to CSV (creates large files)

# CSV fieldnames for logging
LOG_FIELDNAMES = [
    'timestamp',
    'epoch',
    'batch',
    'split',
    'loss',
    'n_pairs',
    'batch_size',
    'dim_model',
    'n_layers',
    'n_head',
    'dim_feedforward',
    'dropout',
    'lr',
    'margin',
]


# =============================================================================
# Debug Settings
# =============================================================================

# Limit dataset size for quick testing (None = use full dataset)
MAX_TRAIN_PAIRS = 50000      # Maximum training pairs to use
MAX_VAL_PAIRS = 10000        # Maximum validation pairs to use

# Set to None to use full datasets:
# MAX_TRAIN_PAIRS = None
# MAX_VAL_PAIRS = None


# =============================================================================
# Dataset Loading
# =============================================================================

# Required metadata column for linking pairs to MGF spectra
REQUIRED_METADATA_COLUMN = 'scan'

# Compressed file extensions to warn about
COMPRESSED_EXTENSIONS = ('.xz', '.gz', '.bz2')


# =============================================================================
# Helper Functions
# =============================================================================

def get_default_paths(data_dir: Path = None):
    """
    Get dictionary of default file paths.
    
    Args:
        data_dir: Base data directory (defaults to DATA_DIR)
        
    Returns:
        Dictionary with all file paths
    """
    from datetime import datetime
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Generate timestamped subdirectory for this training run
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp_str
    
    return {
        'train_metadata': str(data_dir / TRAIN_METADATA_FILE),
        'test_metadata': str(data_dir / TEST_METADATA_FILE),
        'mgf_file': str(data_dir / MGF_FILE),
        'train_pairs_pos': str(data_dir / TRAIN_PAIRS_POS_FILE),
        'train_pairs_neg': str(data_dir / TRAIN_PAIRS_NEG_FILE),
        'test_pairs_pos': str(data_dir / TEST_PAIRS_POS_FILE),
        'test_pairs_neg': str(data_dir / TEST_PAIRS_NEG_FILE),
        'log_csv': str(run_dir / LOG_CSV_FILE),
        'log_txt': str(run_dir / LOG_TXT_FILE),
        'best_model': str(run_dir / BEST_MODEL_FILE),
        'final_model': str(run_dir / FINAL_MODEL_FILE),
        'run_dir': str(run_dir),  # Add run directory path for other outputs
    }


def get_model_config():
    """Get model architecture configuration dictionary."""
    return {
        'd_model': DIM_MODEL,
        'nhead': N_HEAD,
        'dim_feedforward': DIM_FEEDFORWARD,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
    }


def get_training_config():
    """Get training hyperparameters configuration dictionary."""
    return {
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'margin': CONTRASTIVE_MARGIN,
        'warmup_iters': WARMUP_ITERS,
        'cosine_schedule_iters': COSINE_SCHEDULE_ITERS,
        'scheduler_patience': SCHEDULER_PATIENCE,  # Legacy, for backward compatibility
    }