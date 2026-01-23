# Transformer Model Training Recommendations

## Problem Diagnosed
**Embedding Collapse** - Model outputs nearly identical embeddings for all inputs.
- Cosine similarity difference: 0.001 (should be 0.1-0.3)
- Positive and negative pair distances: identical distributions
- Model satisfied loss but learned trivial solution

## Recommended Changes for Retraining

### 1. Model Architecture
```python
# In config.py or command line args:
dim_model = 256          # Keep this
n_layers = 4             # INCREASE from 2 to 4 or 6
n_head = 4               # Keep this
dim_feedforward = 512    # Keep this
dropout = 0.1            # Keep this
```

### 2. Loss Function
```python
# Change margin:
margin = 0.5             # REDUCE from 1.0 to 0.5 or 0.3
```

### 3. Training Hyperparameters
```python
# Reduce initial learning rate:
learning_rate = 5e-5     # REDUCE from 1e-4 to 5e-5 or 1e-5

# Add gradient clipping:
max_grad_norm = 1.0      # Clip gradients to prevent instability

# Consider warmup:
warmup_steps = 1000      # Gradual LR increase at start
```

### 4. Training Command Example
```bash
python -m gleams.transformer_encoder.cli \
    --train_metadata data/train_metadata.parquet \
    --test_metadata data/test_metadata.parquet \
    --train_pairs_pos data/train_metadata_pairs_pos_2.npy \
    --train_pairs_neg data/train_metadata_pairs_neg_2.npy \
    --test_pairs_pos data/test_metadata_pairs_pos_2.npy \
    --test_pairs_neg data/test_metadata_pairs_neg_2.npy \
    --mgf_file data/massivekb_82c0124b.mgf \
    --dim_model 256 \
    --n_layers 6 \
    --n_head 4 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --batch_size 64 \
    --n_epochs 20 \
    --lr 5e-5 \
    --margin 0.3 \
    --output_dir models_v2/
```

### 5. Alternative: Cosine Similarity Loss
Consider switching from Euclidean distance to cosine similarity:
```python
# In loss function:
# Instead of: distance = pairwise_distance(emb1, emb2)
# Use: cosine_dist = 1 - cosine_similarity(emb1, emb2)
```

### 6. Monitoring During Training
Watch for these signs of collapse:
- Validation loss stops improving but train loss continues decreasing
- All embedding norms become identical
- Cosine similarity between random pairs approaches 1.0

### 7. Early Stopping
Use early stopping based on validation metrics:
```python
patience = 5  # Stop if no improvement for 5 epochs
```

## Expected Good Results
After retraining with these settings, you should see:
- Positive pair distances: mean ~0.2-0.4
- Negative pair distances: mean ~0.6-0.9
- Clear separation in distance distributions
- AUC > 0.85
- Cosine similarity difference > 0.1

## Quick Validation Test
Before full retraining, test with:
- Smaller dataset (10k pairs)
- 2-3 epochs
- Check if embedding norms and cosine similarities diversify
