# Learning Rate Warmup Scheduler - Implementation Guide

## Overview

This implementation adds a **CosineWarmupScheduler** to the transformer training pipeline. This scheduler is essential for stable transformer training and combines:

1. **Linear Warmup**: Gradually increases learning rate from 0 to peak LR
2. **Cosine Decay**: Smoothly decreases learning rate following a cosine curve

## Why Warmup?

Transformers are very sensitive to large gradient updates early in training. Without warmup:
- Gradients are noisy and random at the beginning
- Training can be unstable or fail to converge
- The model may not learn effectively

With warmup, you start from a very low learning rate and slowly increase it during the first several hundred steps. This stabilizes training significantly.

## Implementation Details

### Files Modified/Created

1. **`scheduler.py`** (NEW)
   - Contains the `CosineWarmupScheduler` class
   - Based on the Casanovo implementation
   - Steps per batch, not per epoch

2. **`config.py`** (MODIFIED)
   - Added `WARMUP_ITERS = 500` (warmup iterations)
   - Added `COSINE_SCHEDULE_ITERS = 5000` (total cosine period)
   - Kept legacy scheduler settings for backward compatibility

3. **`trainer.py`** (MODIFIED)
   - Replaced `ReduceLROnPlateau` with `CosineWarmupScheduler`
   - Scheduler now steps after each batch (not after validation)
   - Added informative print statements about LR schedule

4. **`cli.py`** (MODIFIED)
   - Added `--warmup-iters` argument
   - Added `--cosine-schedule-iters` argument

## Configuration Parameters

### `WARMUP_ITERS` (default: 500)

The number of training steps (batches) for linear warmup.

- **Typical values**: 500-2000 steps
- **Rule of thumb**: 5-10% of total training steps
- **Calculation**: 
  ```python
  total_steps = (num_training_pairs / batch_size) * num_epochs
  warmup_iters = total_steps * 0.05  # 5% warmup
  ```

**Example:**
- 50,000 training pairs
- Batch size 256
- 10 epochs
- Steps per epoch: 50,000 / 256 ≈ 195
- Total steps: 195 * 10 = 1,950
- Warmup: 1,950 * 0.05 ≈ 100-200 steps

### `COSINE_SCHEDULE_ITERS` (default: 5000)

The number of iterations for the full cosine decay period.

- **Should approximately equal total training steps**
- **Calculation**:
  ```python
  cosine_schedule_iters = (num_training_pairs / batch_size) * num_epochs
  ```

**Example:**
- Using the same numbers: ≈ 1,950 steps

## Usage Examples

### Basic Usage (with defaults)

```bash
python -m gleams.transformer_encoder.cli
```

This will use the default values:
- `warmup_iters=500`
- `cosine_schedule_iters=5000`
- `learning_rate=1e-4`

### Custom Warmup Settings

```bash
python -m gleams.transformer_encoder.cli \
    --warmup-iters 1000 \
    --cosine-schedule-iters 10000 \
    --learning-rate 5e-4
```

### Adjust Based on Your Dataset

Calculate appropriate values:

```python
# Your dataset parameters
num_train_pairs = 50000  # or count from your data
batch_size = 256
num_epochs = 10

# Calculate steps
steps_per_epoch = num_train_pairs // batch_size
total_steps = steps_per_epoch * num_epochs

# Set warmup to 5-10% of total steps
warmup_steps = int(total_steps * 0.05)

# Cosine period should match total training
cosine_period = total_steps

print(f"--warmup-iters {warmup_steps} --cosine-schedule-iters {cosine_period}")
```

## Learning Rate Schedule Behavior

The learning rate follows this pattern:

```
LR
│
│     ╱────╲
│   ╱        ╲
│ ╱            ╲___
│╱                  ╲
└──────────────────────── Steps
  ^        ^
  │        └─ Cosine decay starts after warmup
  └─ Linear warmup phase
```

### Mathematical Formula

```python
# During warmup (step <= warmup_iters):
lr_factor = (step / warmup_iters) * 0.5 * (1 + cos(π * step / cosine_period))

# After warmup:
lr_factor = 0.5 * (1 + cos(π * step / cosine_period))

# Actual learning rate:
lr = base_lr * lr_factor
```

## Monitoring the Learning Rate

The training script prints the current learning rate and schedule parameters:

```
📈 Learning rate schedule:
   Peak LR: 0.0001
   Warmup iterations: 500 batches
   Cosine decay period: 5000 batches
   Note: Scheduler steps after each batch, not epoch
```

During training, you can monitor the learning rate in:
- Console output (current LR printed each epoch)
- CSV log file (`loss_log.csv` includes `lr` column)

## Differences from Previous Scheduler

| Aspect | Old (ReduceLROnPlateau) | New (CosineWarmupScheduler) |
|--------|-------------------------|------------------------------|
| **Stepping** | After each epoch (on validation loss) | After each batch |
| **Warmup** | None | Linear warmup phase |
| **Decay** | Only when validation plateaus | Smooth cosine decay |
| **Predictability** | Reactive to validation | Deterministic schedule |
| **Transformer-friendly** | Not specifically designed | Designed for transformers |

## Tips for Tuning

### If training is unstable early on:
- **Increase** `warmup_iters` (try 1000-2000)
- **Decrease** initial `learning_rate` (try 5e-5)

### If training plateaus too early:
- **Increase** `cosine_schedule_iters` beyond total steps
- This will slow down the LR decay

### If convergence is too slow:
- **Decrease** `warmup_iters` (try 200-300)
- **Increase** `learning_rate` (try 2e-4)

### For very large datasets:
- Keep warmup as a percentage (5-10%) of total steps
- Adjust `cosine_schedule_iters` to match total training steps

## Backward Compatibility

The old scheduler parameters are still in the config for backward compatibility but are not used:

- `SCHEDULER_PATIENCE` - marked as [LEGACY]
- `SCHEDULER_FACTOR` - marked as [LEGACY]

If you need to revert to the old scheduler, you can modify `trainer.py` to use `ReduceLROnPlateau` again.

## References

- **Casanovo**: Original implementation source
- **"Attention is All You Need"** (Vaswani et al., 2017): Introduced warmup for transformers
- Common practice in modern transformer training (BERT, GPT, etc.)

## Questions?

If you see unexpected behavior:
1. Check the console output for LR schedule parameters
2. Verify `warmup_iters` and `cosine_schedule_iters` are appropriate for your dataset size
3. Monitor the `lr` column in `loss_log.csv` to see actual learning rate over time
4. Consider adjusting warmup duration if training is unstable
