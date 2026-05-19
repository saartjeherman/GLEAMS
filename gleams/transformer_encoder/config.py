"""Editable hyperparameters and paths for transformer training.

Tweak anything here and rerun `train.py` — no other file should need editing
for routine experiments. Each run writes its full set of hyperparameters into
the per-epoch CSV log and into the model checkpoint, so config drift is always
recoverable from the artifacts in `results/<timestamp>/`.
"""

# -----------------------------
# Input data paths
# -----------------------------
TRAIN_METADATA_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata.parquet'
TEST_METADATA_FILE  = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata.parquet'

TRAIN_POS_PAIRS_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata_pairs_pos_2.npy'
TRAIN_NEG_PAIRS_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata_pairs_neg_2.npy'
VAL_POS_PAIRS_FILE   = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata_pairs_pos_2.npy'
VAL_NEG_PAIRS_FILE   = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata_pairs_neg_2.npy'

MGF_PATH = '/home/saartje/Desktop/Research/GLEAMS/data/massivekb_82c0124b.mgf'

# -----------------------------
# Cached pre-encoded spectra (built on first run, reused after)
# -----------------------------
TRAIN_NPZ_PATH = '/home/saartje/Desktop/Research/GLEAMS/data/train_spectra.npz'
VAL_NPZ_PATH   = '/home/saartje/Desktop/Research/GLEAMS/data/test_spectra.npz'

# -----------------------------
# Output
# -----------------------------
RESULTS_ROOT = '/home/saartje/Desktop/Research/GLEAMS/results'

# -----------------------------
# Data loading
# -----------------------------
BATCH_SIZE = 128
NUM_WORKERS = 2

# Cap pairs per (split, polarity). Set to None to use the full pair lists.
MAX_TRAIN_PAIRS_PER_CLASS = 50000
MAX_VAL_PAIRS_PER_CLASS   = 10000

# Keep only the top-K most intense peaks per spectrum (bounds attention cost
# from outlier spectra). Set to None to keep all peaks.
MAX_PEAKS = 150

# RNG seed for reproducible pair subsampling.
SUBSAMPLE_SEED = 0

# -----------------------------
# Model architecture
# -----------------------------
DIM_MODEL = 256
N_HEAD = 4
DIM_FEEDFORWARD = 512
N_LAYERS = 4
DROPOUT = 0.1

# -----------------------------
# Optimization
# -----------------------------
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MARGIN = 2.0
N_EPOCHS = 10

# ReduceLROnPlateau: halve LR after `patience` epochs of no val improvement.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2

# -----------------------------
# Logging
# -----------------------------
# Set True to also write a CSV row per training batch (large file).
LOG_BATCH_LEVEL = False

# -----------------------------
# Metadata extraction (only used by prepare_data.py)
# -----------------------------
# Combined metadata before train/test split — an intermediate cache.
MASSIVEKB_METADATA_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/massivekb_with_scan.parquet'

# A (sequence, charge) key is only kept once exactly N_REPLICATES spectra have
# been observed for it. Acts as a quality filter — singletons are dropped.
N_REPLICATES = 10

# Stop after this many peptides have hit N_REPLICATES. Set to None to walk the
# entire MGF (no cap).
N_PEPTIDES_TARGET = 100_000

# I/O batching: flush the CSV buffer every N completed peptides.
FLUSH_EVERY = 5000

# -----------------------------
# Train/test split (only used by prepare_data.py)
# -----------------------------
TEST_SPLIT_RATIO = 0.2
TEST_SPLIT_RANDOM_STATE = 42

# -----------------------------
# Pair generation (only used by prepare_data.py)
# -----------------------------
# (min_charge, max_charge) inclusive. Pairs are generated per charge.
CHARGES = (2, 5)

# Precursor m/z window (in ppm) for a candidate negative pair.
PAIR_MZ_TOLERANCE = 10

# Fragment m/z tolerance (in Da) for declaring two theoretical b/y ions to overlap.
NEGATIVE_PAIR_FRAGMENT_TOLERANCE = 0.01

# Max ratio of overlapping fragments (vs. the shorter peptide's #ions) allowed
# for a negative pair — keeps obviously-too-similar pairs out of the negatives.
NEGATIVE_PAIR_MATCHING_FRAGMENTS_THRESHOLD = 0.25
