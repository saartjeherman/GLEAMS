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
DIM_MODEL = 256              # Model embedding dimension
N_HEAD = 4                   # Number of attention heads
DIM_FEEDFORWARD = 512        # Feedforward network dimension
N_LAYERS = 2                 # Number of transformer layers
DROPOUT = 0.3                # Dropout rate for regularization


# =============================================================================
# Training Hyperparameters
# =============================================================================

# Optimization
LEARNING_RATE = 1e-4         # Initial learning rate
WEIGHT_DECAY = 1e-5          # L2 regularization weight
BATCH_SIZE = 64              # Training batch size
N_EPOCHS = 10                # Number of training epochs

# Loss function
CONTRASTIVE_MARGIN = 2.0     # Margin for contrastive loss

# Learning rate scheduler
SCHEDULER_PATIENCE = 2       # Epochs to wait before reducing LR
SCHEDULER_FACTOR = 0.5       # Factor to reduce LR by
SCHEDULER_VERBOSE = True     # Print LR reduction messages


# =============================================================================
# DataLoader Settings
# =============================================================================

NUM_WORKERS = 0              # Number of data loading workers (0 = single-threaded)
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
    if data_dir is None:
        data_dir = DATA_DIR
    
    return {
        'train_metadata': str(data_dir / TRAIN_METADATA_FILE),
        'test_metadata': str(data_dir / TEST_METADATA_FILE),
        'mgf_file': str(data_dir / MGF_FILE),
        'train_pairs_pos': str(data_dir / TRAIN_PAIRS_POS_FILE),
        'train_pairs_neg': str(data_dir / TRAIN_PAIRS_NEG_FILE),
        'test_pairs_pos': str(data_dir / TEST_PAIRS_POS_FILE),
        'test_pairs_neg': str(data_dir / TEST_PAIRS_NEG_FILE),
        'log_csv': str(data_dir / LOG_CSV_FILE),
        'log_txt': str(data_dir / LOG_TXT_FILE),
        'best_model': str(MODEL_DIR / BEST_MODEL_FILE),
        'final_model': str(MODEL_DIR / FINAL_MODEL_FILE),
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
        'scheduler_patience': SCHEDULER_PATIENCE,
    }
