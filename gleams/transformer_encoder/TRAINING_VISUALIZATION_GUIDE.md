# Training Visualization - Automatic Plot Generation

## Overview

The training script now automatically generates visualization plots at the end of **every epoch**. These plots help you monitor model performance during training and are saved with timestamps to prevent overwriting across multiple runs.

## What Gets Generated

At the end of each epoch, two plots are created:

### 1. **Distance Distribution Plot**
- Shows histograms of embedding distances for positive and negative pairs
- Includes FDR threshold line (1% false discovery rate)
- Displays train loss, validation loss, and current learning rate in title
- Filename: `epoch_XXX_YYYYMMDD_HHMMSS_distances.png`

### 2. **ROC Curve**
- Receiver Operating Characteristic curve
- Shows model's ability to distinguish positive from negative pairs
- Includes AUC (Area Under Curve) score
- Filename: `epoch_XXX_YYYYMMDD_HHMMSS_roc.png`

## File Organization

### Plot Directory
All training plots are saved to:
```
GLEAMS/results/training_plots/
```

This directory is automatically created if it doesn't exist.

### Filename Format
```
epoch_<epoch>_<timestamp>_<type>.png
```

**Example filenames:**
- `epoch_001_20260123_143052_distances.png` (Epoch 1, Jan 23 2026, 2:30:52 PM)
- `epoch_001_20260123_143052_roc.png`
- `epoch_005_20260123_152045_distances.png` (Epoch 5, Jan 23 2026, 3:20:45 PM)

### Timestamp Format
- `YYYYMMDD_HHMMSS` (Year-Month-Day Hour:Minute:Second)
- Example: `20260123_143052` = January 23, 2026 at 14:30:52

This ensures:
- ✅ Files are never overwritten between runs
- ✅ Easy to identify when each run happened
- ✅ Files sort chronologically by name
- ✅ Multiple runs can coexist in the same directory

## What's Visualized

### During Training
The plots are generated using:
- **Validation set**: Ensures unbiased evaluation
- **5,000 random pairs**: Faster computation (configurable)
- **Current epoch's model state**: Shows real-time performance

### Console Output
After each epoch, you'll see:
```
  📊 Creating visualization plots for epoch 3...
  ✓ Plots saved to GLEAMS/results/training_plots
    - AUC: 0.8523
    - Positive distance mean: 12.3456
    - Negative distance mean: 15.7890
```

This gives you immediate feedback on:
- **AUC score**: Higher is better (1.0 = perfect, 0.5 = random)
- **Distance separation**: Negative pairs should have higher distances than positive pairs

## Configuration

### Adjusting Number of Pairs
In [trainer.py](trainer.py#L479), you can modify:

```python
max_pairs=5000  # Change this value
```

- **Lower value** (e.g., 1000): Faster plotting, less representative
- **Higher value** (e.g., 10000): Slower plotting, more accurate statistics
- **None**: Use all pairs (slowest, most accurate)

### Adjusting FDR Threshold
In [trainer.py](trainer.py#L487), modify:

```python
fdr=0.01,  # 1% false discovery rate
```

Common values:
- `0.01` = 1% FDR (standard)
- `0.05` = 5% FDR (more lenient)
- `0.001` = 0.1% FDR (more stringent)

## Interpreting the Plots

### Distance Distribution Plot

**Good model indicators:**
- 📈 Clear separation between positive (purple) and negative (blue) distributions
- 📉 Positive pairs have **lower** distances than negative pairs
- 🎯 Minimal overlap between the two distributions

**Problem indicators:**
- ⚠️ Heavy overlap between distributions → Model not learning well
- ⚠️ Similar means for both distributions → Embedding collapse
- ⚠️ No change across epochs → Training stalled

### ROC Curve

**AUC Score interpretation:**
- **0.95-1.0**: Excellent discrimination
- **0.85-0.95**: Good discrimination
- **0.75-0.85**: Fair discrimination
- **0.50-0.75**: Poor discrimination
- **~0.50**: Random guessing (model not learning)

**What to expect:**
- Early epochs (1-3): AUC around 0.6-0.7
- Mid training (4-7): AUC around 0.7-0.85
- Late training (8-10): AUC around 0.85-0.95

## Example Training Session

### Run 1 (Morning training)
```
results/training_plots/
  epoch_001_20260123_090512_distances.png
  epoch_001_20260123_090512_roc.png
  epoch_002_20260123_091234_distances.png
  epoch_002_20260123_091234_roc.png
  ...
  epoch_010_20260123_104523_distances.png
  epoch_010_20260123_104523_roc.png
```

### Run 2 (Afternoon training with different hyperparameters)
```
results/training_plots/
  ... (morning files still here)
  epoch_001_20260123_143052_distances.png
  epoch_001_20260123_143052_roc.png
  ...
```

**No conflicts!** Each run has unique timestamps.

## Troubleshooting

### Plots not appearing
1. Check console for error messages
2. Verify `results/training_plots/` directory exists
3. Check disk space

### "Failed to create visualization plots" warning
- Training continues normally
- Plots are optional (won't break training)
- Check the error traceback for details

### Out of memory errors
- Reduce `max_pairs` from 5000 to 1000
- This uses less GPU memory during plot generation

### Plots look wrong/collapsed
- Check if model is learning (compare with loss curves)
- See recommendations in the notebook's diagnostic cells
- Consider adjusting learning rate or warmup

## Comparing Multiple Runs

To compare different training runs:

1. **By timestamp**: Group plots by date/time
   - `ls results/training_plots/epoch_*_202601231*` (morning run)
   - `ls results/training_plots/epoch_*_202601231[45]*` (afternoon run)

2. **By epoch**: Compare same epoch across runs
   - `ls results/training_plots/epoch_005_*_distances.png`

3. **Create a comparison notebook**: Load and display plots side-by-side

## Tips

### Organizing plots for multiple experiments
Consider creating subdirectories:
```bash
results/training_plots/
  run_baseline_20260123/
  run_higher_lr_20260123/
  run_more_layers_20260124/
```

You can modify the `plots_dir` variable in `trainer.py` to use a run-specific subdirectory.

### Monitoring training remotely
If training on a server:
```bash
# Copy plots to local machine
scp -r server:~/GLEAMS/results/training_plots/ ./local_plots/

# Or use rsync for incremental updates
rsync -av --progress server:~/GLEAMS/results/training_plots/ ./local_plots/
```

### Creating a training animation
After training completes:
```bash
cd results/training_plots
# Create GIF of distance plots across epochs (requires imagemagick)
convert -delay 100 epoch_*_distances.png training_progress.gif
```

## Disabling Visualization

If you want to disable automatic plotting (for faster training):

In [trainer.py](trainer.py#L463), comment out or remove this section:
```python
# ========================================
# Create visualization plots for this epoch
# ========================================
try:
    ...  # entire visualization block
except Exception as e:
    ...
```

Or wrap it in a condition:
```python
if args.create_plots:  # Add this argument to CLI
    # ... visualization code ...
```

## Related Files

- [visualization.py](visualization.py) - Plotting functions
- [visualize_embeddings.ipynb](visualize_embeddings.ipynb) - Detailed analysis notebook
- [trainer.py](trainer.py) - Main training script with visualization
- [config.py](config.py) - Training configuration

## Questions?

The visualization code is designed to be:
- ✅ Non-intrusive (failures won't stop training)
- ✅ Informative (immediate feedback on model quality)
- ✅ Organized (timestamped, won't overwrite)
- ✅ Efficient (samples pairs for speed)

If you encounter issues, the full error traceback will be printed to help debug.
