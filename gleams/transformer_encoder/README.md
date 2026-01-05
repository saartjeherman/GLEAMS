# Transformer Encoder Module

This module contains the refactored components for training the GLEAMS spectrum encoder with contrastive learning.

## Module Structure

```
transformer_encoder/
├── __init__.py           # Package exports
├── config.py             # Configuration constants
├── models.py             # Neural network models
├── dataset.py            # Dataset classes
├── dataloader.py         # DataLoader utilities  
├── evaluator.py          # Model evaluation
├── trainer.py            # Training logic
├── cli.py                # Command-line interface
└── README.md             # Documentation
```

## Components

### config.py
Central configuration file containing all default values:
- **Directory paths**: `DATA_DIR`, `MODEL_DIR`, file names
- **Model architecture**: `DIM_MODEL`, `N_HEAD`, `DIM_FEEDFORWARD`, `N_LAYERS`, `DROPOUT`
- **Training hyperparameters**: `LEARNING_RATE`, `WEIGHT_DECAY`, `BATCH_SIZE`, `N_EPOCHS`, `CONTRASTIVE_MARGIN`
- **Scheduler settings**: `SCHEDULER_PATIENCE`, `SCHEDULER_FACTOR`
- **DataLoader settings**: `NUM_WORKERS`, `PIN_MEMORY`, shuffle flags
- **Logging settings**: `LOG_BATCH_LEVEL`, `LOG_FIELDNAMES`
- **Debug settings**: `MAX_TRAIN_PAIRS`, `MAX_VAL_PAIRS`
- **Helper functions**: `get_default_paths()`, `get_model_config()`, `get_training_config()`

### models.py
- **CustomSpectrumTransformerEncoder**: Transformer encoder with precursor mass and charge encoding
- **ContrastiveLoss**: Contrastive loss function for spectrum pair similarity

### dataset.py
- **SpectraDataset**: Loads mass spectra from MGF files with optional scan filtering
- **SpectrumPairDataset**: Creates paired dataset with similarity labels

### dataloader.py
- **create_dataloader()**: Factory function for creating PyTorch DataLoaders

### evaluator.py
- **evaluate_contrastive()**: Computes validation loss with progress tracking

### trainer.py
- **train_model()**: Main training loop with:
  - Model checkpointing (saves best and final models)
  - Learning rate scheduling
  - CSV + text logging
  - Progress bars
  - Automatic pair remapping (metadata indices → dataset indices)

### cli.py
- **parse_args()**: Argument parser with sensible defaults for:
  - Data paths
  - Model architecture
  - Training hyperparameters
  - Debug options

## Configuration

All default values are centralized in `config.py`. You can:

**1. Edit defaults directly** in `config.py`:
```python
# Change default values
DIM_MODEL = 512          # Instead of 256
BATCH_SIZE = 128         # Instead of 64
MAX_TRAIN_PAIRS = None   # Use full dataset instead of 50K
```

**2. Override via command-line** (without changing code):
```bash
python run_gleams.py --dim-model 512 --batch-size 128
```

**3. Import in code**:
```python
from transformer_encoder import config

print(config.BATCH_SIZE)  # 64
print(config.DIM_MODEL)   # 256

# Get all paths
paths = config.get_default_paths()
print(paths['train_metadata'])  # GLEAMS/data/train_metadata.parquet
```

## Usage

### Basic Training
```bash
python run_gleams.py
```

### Custom Configuration
```bash
python run_gleams.py \
  --data-dir /path/to/data \
  --batch-size 128 \
  --n-epochs 20 \
  --learning-rate 5e-5 \
  --dropout 0.4
```

### Debug Mode (Limited Dataset)
```bash
python run_gleams.py \
  --max-train-pairs 10000 \
  --max-val-pairs 2000 \
  --n-epochs 3
```

## Importing in Code

```python
from transformer_encoder import (
    CustomSpectrumTransformerEncoder,
    ContrastiveLoss,
    SpectraDataset,
    create_dataloader,
    train_model,
    evaluate_contrastive,
    parse_args,
    config
)

# Use config defaults
encoder = CustomSpectrumTransformerEncoder(
    d_model=config.DIM_MODEL,
    nhead=config.N_HEAD,
    dim_feedforward=config.DIM_FEEDFORWARD,
    n_layers=config.N_LAYERS,
    dropout=config.DROPOUT
)

# Or use helper functions
model_config = config.get_model_config()
encoder = CustomSpectrumTransformerEncoder(**model_config)

# Run full training pipeline
args = parse_args()
train_model(args)
```

## Model Outputs

- **Best model checkpoint**: Saved when validation loss improves (`GLEAMS/models/best_model.pt`)
- **Final model**: Saved at end of training (`GLEAMS/models/final_model.pt`)
- **Loss log CSV**: Epoch-level metrics (`GLEAMS/data/loss_log.csv`)
- **Training log**: Full console output (`GLEAMS/data/training.log`)

## Dependencies

- PyTorch
- depthcharge
- pandas
- numpy
- tqdm
- pyteomics (via ms_io module)
