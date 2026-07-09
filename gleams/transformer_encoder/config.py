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
VAL_METADATA_FILE   = '/home/saartje/Desktop/Research/GLEAMS/data/val_metadata.parquet'
TEST_METADATA_FILE  = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata.parquet'

# Pair files, one set per split. Training uses train (fit) + val (early stopping);
# test is held out for final evaluation. (Charge 2 wired here; other charges are
# generated too — see CHARGES — and can be swapped in as needed.)
TRAIN_POS_PAIRS_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata_pairs_pos_2.npy'
TRAIN_NEG_PAIRS_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata_pairs_neg_2.npy'
VAL_POS_PAIRS_FILE   = '/home/saartje/Desktop/Research/GLEAMS/data/val_metadata_pairs_pos_2.npy'
VAL_NEG_PAIRS_FILE   = '/home/saartje/Desktop/Research/GLEAMS/data/val_metadata_pairs_neg_2.npy'
TEST_POS_PAIRS_FILE  = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata_pairs_pos_2.npy'
TEST_NEG_PAIRS_FILE  = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata_pairs_neg_2.npy'

MGF_PATH = '/home/saartje/Desktop/Research/GLEAMS/data/massivekb_82c0124b.mgf'

# -----------------------------
# Cached pre-encoded spectra (built on first run, reused after)
# -----------------------------
TRAIN_NPZ_PATH = '/home/saartje/Desktop/Research/GLEAMS/data/train_spectra.npz'
VAL_NPZ_PATH   = '/home/saartje/Desktop/Research/GLEAMS/data/val_spectra.npz'
TEST_NPZ_PATH  = '/home/saartje/Desktop/Research/GLEAMS/data/test_spectra.npz'

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
# Training fits on train and monitors on test; the val split is a later holdout.
MAX_TRAIN_PAIRS_PER_CLASS = 100000
MAX_TEST_PAIRS_PER_CLASS  = 20000

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

# Hard ceiling on training length. Training also stops early — see below.
MAX_EPOCHS = 30

# Early stopping: stop training after this many consecutive epochs without an
# improvement in val_loss. Set to None to disable (always train MAX_EPOCHS).
# A value slightly above LR_SCHEDULER_PATIENCE gives the LR reduction at least
# one or two epochs to help before we give up.
EARLY_STOPPING_PATIENCE = 5

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
MASSIVEKB_METADATA_FILE = '/home/saartje/Desktop/Research/GLEAMS/data/massivekb.parquet'

# Variable replicate collection. Each (sequence, charge) group keeps between
# MIN_REPLICATES and MAX_REPLICATES spectra:
#   - MIN_REPLICATES: groups with fewer than this are dropped (singletons carry
#     no positive pair, so MIN must be >= 2 for pair training to work).
#   - MAX_REPLICATES: hard cap per group, so one ultra-common peptide can't
#     dominate the set or explode its C(n,2) positive-pair count.
MIN_REPLICATES = 2
MAX_REPLICATES = 10

# Stop walking the MGF once this many *kept* spectra have accumulated (spectra
# belonging to groups that have reached MIN_REPLICATES). Set to None to walk the
# entire MGF (no cap). Groups still buffering below MIN at stop-time are dropped.
SPECTRA_TARGET = 10_000_000

# I/O batching: flush the CSV buffer every N completed peptides.
FLUSH_EVERY = 5000

# -----------------------------
# Train/val/test split (only used by prepare_data.py)
# -----------------------------
# Peptide-level 80/10/10 split, matching the original GLEAMS model. Train is
# whatever is left after carving off the val and test fractions of the peptides.
VAL_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.1
SPLIT_RANDOM_STATE = 42

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
