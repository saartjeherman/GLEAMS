# Transformer-Encoder Contrastive Spectrum Embedder

Trains a [depthcharge](https://github.com/wfondrie/depthcharge) `SpectrumTransformerEncoder` in a Siamese setup to embed MS/MS spectra such that spectra from the same peptide land close together in latent space, and spectra from different peptides land far apart.

This is a re-implementation of the GLEAMS idea (Bittremieux et al., 2022) with two differences:

1. The backbone is a **transformer over variable-length peak sequences** instead of a CNN over fixed-size binned vectors.
2. The loss is the same Hadsell contrastive loss, but operates on the transformer's global token output rather than a final dense layer.

---

## Folder layout

| File | Role |
|---|---|
| `config.py` | All editable hyperparameters and paths. The only file you should normally need to edit. |
| `model.py` | `ContrastiveLoss` + `CustomSpectrumTransformerEncoder` (depthcharge subclass with precursor m/z + charge baked into the global token). |
| `spectrum_pairs.py` | MGF → packed `.npz` pre-encoder, `SpectrumPairDataset`, and the `collate_pairs` batch padder. |
| `prepare_data.py` | Three-stage data-prep script: (1) walk the MGF to build a combined metadata parquet, (2) train/test split it, (3) generate positive + negative pair `.npy` files. Each stage caches its output and skips if it already exists. |
| `training.py` | `train_model(run_dir)` — builds data loaders, model, optimizer; runs the training loop; saves checkpoints, CSV loss log, and plots. |
| `visualization.py` | All plotting helpers: `plot_distance_distributions` (per-epoch KDE of pair distances) and `plot_loss_curves` (train + val loss vs. epoch). |
| `train.py` | Entry point. Creates `results/<timestamp>/`, tees stdout to a log file, calls `train_model`. |
| `__init__.py` | Empty — marks the folder as a Python package. |

---

## Inputs you need on disk

All paths are configured in `config.py`. Defaults assume:

```
data/
├── train_metadata.parquet            # columns: scan, rtinseconds, sequence, mz, charge
├── test_metadata.parquet             # same schema; used as validation
├── train_metadata_pairs_pos_2.npy    # shape (N, 2) uint32 — positive pair row indices
├── train_metadata_pairs_neg_2.npy    # shape (N, 2) uint32 — negative pair row indices
├── test_metadata_pairs_pos_2.npy
├── test_metadata_pairs_neg_2.npy
└── massivekb_82c0124b.mgf            # the bundled MGF with all spectra
```

**Metadata** is generated upstream (see `read.ipynb`) by iterating the MGF with `pyteomics.mgf.read(..., use_index=False)` and saving the **enumerate position** (0, 1, 2, …) as the `scan` column. This is critical — see "How pairs map to spectra" below.

**Pair files** are `(N, 2)` arrays of row indices into the corresponding metadata file. A row of `[42, 117]` in the positives means "metadata row 42 and metadata row 117 came from the same peptide." Negative pair files have the same structure but for spectra of different peptides.

---

## End-to-end pipeline

```
config.py
   │
   ▼                                                                              
train.py ───────► train_model(run_dir)
                     │
                     │ 1. Load train + val metadata (parquet)
                     │ 2. Load pair .npy files, subsample to MAX_*_PAIRS_PER_CLASS
                     │ 3. preencode_mgf_to_npzs(mgf, [(train_md, train.npz),
                     │                                  (test_md,  val.npz)])
                     │     ─ ONE pass through the MGF
                     │     ─ for each spectrum at enumerate position P,
                     │       check if any metadata row has scan==P;
                     │       if so, store its peaks in row P's slot
                     │     ─ stops once P > max(metadata['scan'])
                     │     ─ skips entirely if both .npz already exist
                     │ 4. Build SpectrumPairDataset(npz, pos_pairs, neg_pairs):
                     │     ─ mmap the .npz
                     │     ─ stack pos+neg pairs with labels 1/0
                     │     ─ drop any pair touching a row not in the MGF
                     │ 5. DataLoader with collate_pairs (pads to longest in batch)
                     │ 6. Build CustomSpectrumTransformerEncoder + ContrastiveLoss
                     │ 7. For each epoch:
                     │     ─ train batches, collect distances per pair polarity
                     │     ─ save plots/train_epoch_NNN.png
                     │     ─ run evaluate_contrastive on val set
                     │     ─ save plots/val_epoch_NNN.png
                     │     ─ write per-epoch row to loss_log.csv
                     │     ─ ReduceLROnPlateau step
                     │     ─ save best_model.pt if val_loss improved
                     ▼
                  save final_model.pt
```

---

## How pairs map to spectra (the key trick)

This is the subtlest part of the pipeline.

### The matching identifier is **enumerate position**, not scan number

The bundled MGF concatenates spectra from many source files. Each source file's scan numbers reset from 1, so `scan:1856` in the MGF title is **not unique** — it appears many times across the bundle.

So upstream metadata extraction (`read.ipynb`) does this:

```python
with mgf.read(mgf_path, use_index=False) as reader:
    for scan_idx, spectrum in enumerate(reader):
        # scan_idx is 0, 1, 2, ... across the whole bundle
        row['scan'] = scan_idx
```

…and `preencode_mgf_to_npzs` matches the same way:

```python
with mgf.read(mgf_path, use_index=False) as reader:
    for position, spec in enumerate(reader):
        slots = scan_to_slots.get(position)  # metadata row(s) with scan==position
        if slots:
            store spec at those row indices
```

Both iterate the MGF with the same call, so `position` in the encoder equals `scan_idx` in the metadata, byte-for-byte.

### Pair files reference row indices, not scans

A pair file row `[42, 117]` means "metadata row 42 paired with metadata row 117." So the lookup chain is:

```
pair row index 42
   │
   ▼
metadata.iloc[42]['scan']  →  e.g. 1,234,567   (enumerate position in MGF)
   │
   ▼
.npz row 42 (peaks for that spectrum)
```

The `.npz` is built so that **row i of the packed arrays corresponds to row i of the metadata DataFrame**. Pairs index directly into the `.npz` without going through the scan number at all at training time.

---

## The `.npz` format

`spectrum_pairs.py` packs variable-length peak arrays CSR-style:

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `mz_flat` | float32 | (total peaks,) | All m/z values concatenated |
| `int_flat` | float32 | (total peaks,) | All intensities concatenated |
| `offsets` | int64 | (N+1,) | Row `i`'s peaks live at `mz_flat[offsets[i]:offsets[i+1]]` |
| `pepmass` | float32 | (N,) | Precursor m/z per row |
| `charge` | int16 | (N,) | Precursor charge per row |

`N = len(metadata)`. Rows whose scan wasn't found in the MGF get length 0; `SpectrumPairDataset` filters out any pairs that touch such rows.

---

## How a batch is built

`SpectrumPairDataset.__getitem__(idx)`:

1. Look up `pair = self.pairs[idx]` → two row indices `(r1, r2)`.
2. For each row: slice `mz_flat[offsets[r]:offsets[r+1]]`, `int_flat[...]`, scalar `pepmass[r]`, scalar `charge[r]`.
3. If `len(mz) > MAX_PEAKS`: keep the `MAX_PEAKS` highest-intensity peaks (then re-sort by m/z so downstream sorted-m/z assumptions hold).
4. Return `((mz1, int1, pep1, chg1), (mz2, int2, pep2, chg2), label)`.

`collate_pairs(batch)`:

1. Find the max peak count in the batch.
2. Pad every spectrum's `mz`/`intensity` to that length with zeros.
3. Stack `pepmass` and `charge` to `(batch, 1)` (depthcharge's `FloatEncoder` expects a feature dim).
4. Return `((mz_pad, int_pad, pepmass, charge), …, labels)`.

The padding zeros are the natural mask for the transformer — depthcharge ignores them during attention.

### Why cap peaks (`MAX_PEAKS`)?

Self-attention is O(L²). One spectrum with 2000 peaks in a batch of 128 forces the whole batch's attention matrix to 2000×2000, killing throughput. Capping at 150 (well above the median of ~200) keeps the worst-case batch bounded without losing significant information — the top-150 most intense peaks carry almost all the spectrum's identity signal.

### Why preprocess (`preencode_mgf_to_npzs`) instead of streaming from MGF?

- MGF parsing is pure Python and slow (~2-5k spectra/sec).
- A 130 GB MGF takes ~10 min to walk once; doing that every epoch would be infeasible.
- The `.npz` is ~150 MB for 800k spectra and mmaps in milliseconds.
- Cost is **one-time**: the `.npz` is cached; `preencode_mgf_to_npzs` skips silently if it already exists.

---

## The model

`CustomSpectrumTransformerEncoder` is a thin subclass of `depthcharge.transformers.SpectrumTransformerEncoder`. It treats each `(m/z, intensity)` peak as a token, fed through a transformer with `N_LAYERS` self-attention layers. The `global_token_hook` injects per-spectrum scalars (precursor m/z and charge) into the model's special global token by passing them through a learned `FloatEncoder`.

The output `emb_full[:, 0, :]` is the global token's embedding — a single `d_model`-dim vector per spectrum.

### Contrastive loss

```python
L = label * d²  +  (1 - label) * max(0, margin - d)²
```

- `label = 1` (positive pair) → loss = `d²`, so the model is pushed to minimize the distance.
- `label = 0` (negative pair) → loss is zero if `d ≥ margin`; otherwise it pushes `d` up toward `margin`.

If you raise `MARGIN`, the model is asked to spread negatives further. If your val plot shows negatives still clustered around the current margin, training longer or raising the margin can help.

---

## How to run

```bash
cd /home/saartje/Desktop/Research/GLEAMS/gleams/transformer_encoder

# 1. One-time (or after changing data-prep params): build metadata + pair files.
python prepare_data.py

# 2. Train. First run also pre-encodes the MGF to .npz (~7 min for 130 GB).
python train.py
```

`train.py`:

1. Creates `results/<YYYY-MM-DD_HH-MM-SS>/`.
2. Tees all stdout/stderr to `<run_dir>/training.log`.
3. Calls `train_model(run_dir)`.

To change hyperparameters: edit `config.py` and rerun. Each run is its own directory so different sweeps never collide.

### What `prepare_data.py` does (three cached stages)

| Stage | Output | Re-runs only when … |
|---|---|---|
| `build_metadata` | `MASSIVEKB_METADATA_FILE` (one combined parquet) | the parquet is missing, or you flip `OVERWRITE = True` |
| `split_metadata` | `TRAIN_METADATA_FILE`, `TEST_METADATA_FILE` | either split parquet is missing |
| `prepare_pairs` | `<basename>_pairs_{pos,neg}_<charge>.npy` (4 files × charges) | any of the pair files for a charge is missing |

So a normal rerun is cheap — it sees the cached parquets and `.npy`s and prints "skipping" for each stage. To rebuild a stage, delete its output or set `OVERWRITE = True` at the top of the file.

Knobs that affect each stage live in `config.py`:

| Stage | Relevant config keys |
|---|---|
| `build_metadata` | `MGF_PATH`, `MASSIVEKB_METADATA_FILE`, `N_REPLICATES`, `N_PEPTIDES_TARGET` (set to `None` for the whole MGF), `FLUSH_EVERY` |
| `split_metadata` | `MASSIVEKB_METADATA_FILE`, `TRAIN_METADATA_FILE`, `TEST_METADATA_FILE`, `TEST_SPLIT_RATIO`, `TEST_SPLIT_RANDOM_STATE` |
| `prepare_pairs` | `TRAIN_METADATA_FILE`, `TEST_METADATA_FILE`, `CHARGES`, `PAIR_MZ_TOLERANCE`, `NEGATIVE_PAIR_FRAGMENT_TOLERANCE`, `NEGATIVE_PAIR_MATCHING_FRAGMENTS_THRESHOLD` |

---

## Output structure

```
results/2026-05-18_14-30-22/
├── training.log              # full console output + tracebacks
├── loss_log.csv              # one row per epoch (train + val) with all hyperparams
├── loss_curves.png           # train + val loss vs. epoch (refreshed every epoch)
├── best_model.pt             # checkpoint with lowest val_loss seen
├── final_model.pt            # checkpoint at end of training
└── plots/
    ├── train_epoch_001.png   # KDE of positive vs negative distances on train data
    ├── val_epoch_001.png     # same, on validation data
    ├── train_epoch_002.png
    ├── val_epoch_002.png
    └── ...
```

`loss_curves.png` is overwritten after every epoch, so opening it in an image viewer that refreshes (e.g. `eog` on Linux) gives you a live training monitor without having to parse the CSV. To re-plot any past run from outside training, just call `visualization.plot_loss_curves(<path/to/loss_log.csv>, <out.png>)`.

### Reading the distance plots

These mirror Extended Data Fig. 3 of the GLEAMS paper. As training progresses you want:

- Positive (pink) peak migrating toward 0.
- Negative (blue) peak moving up to or past `MARGIN`.
- The crossover point between the two curves to shift right (less overlap).

If the curves look identical to each other → model isn't learning.
If train looks great but val doesn't → overfitting; increase data or dropout.

### Reading `loss_log.csv`

Each epoch writes two rows (`split=train_epoch`, `split=val_epoch`). All hyperparameters are stamped on every row, so the CSV is fully self-describing — you can grep across many runs to compare without consulting the config that was used at training time.

### Reading checkpoints

```python
import torch
ckpt = torch.load('results/.../best_model.pt')
ckpt['model_state_dict']     # weights
ckpt['hyperparameters']      # dict matching the arch used
ckpt['val_loss']             # best val_loss when saved
```

To resume: reconstruct the model with `ckpt['hyperparameters']`, then `model.load_state_dict(ckpt['model_state_dict'])`.

---

## When to rebuild the `.npz`

The `.npz` is keyed by metadata row, **not** by which pairs reference it. So:

| Change | Rebuild `.npz`? |
|---|---|
| `MAX_TRAIN_PAIRS_PER_CLASS`, `MAX_VAL_PAIRS_PER_CLASS` | No |
| `MAX_PEAKS` (applied lazily at `__getitem__`) | No |
| Model hyperparameters | No |
| Optimizer / margin / epochs | No |
| Different metadata file (different scans) | Yes |
| Different MGF | Yes |
| Modified `preencode_mgf_to_npzs` itself | Yes |

To force a rebuild: `rm data/train_spectra.npz data/test_spectra.npz` then rerun. `preencode_mgf_to_npzs` regenerates whatever's missing.

---

## Hyperparameter cheat sheet (`config.py`)

| Knob | Default | When to change |
|---|---|---|
| `MAX_TRAIN_PAIRS_PER_CLASS` | 50,000 | Highest-impact lever; bump to 200k+ or `None` once smoke test works |
| `MAX_PEAKS` | 150 | Lower → faster batches; higher → more spectrum info (quadratic cost) |
| `BATCH_SIZE` | 128 | Lower if you OOM with bigger `MAX_PEAKS` or `DIM_MODEL` |
| `DIM_MODEL` | 256 | Increase (e.g. 384, 512) once data is large enough to support capacity |
| `N_LAYERS` | 4 | 2-4 typical; deeper helps with more data |
| `DROPOUT` | 0.1 | Lower with more data; higher if val_loss > train_loss diverges |
| `LEARNING_RATE` | 1e-4 | Lower if loss is unstable; otherwise leave |
| `MARGIN` | 2.0 | Raise if negatives cluster well below it after long training |
| `N_EPOCHS` | 10 | 20-50 for serious runs; watch the per-epoch plots |
| `LR_SCHEDULER_PATIENCE` | 2 | Epochs of no val improvement before LR is halved |

---

## Quick sanity checks if something looks off

| Symptom | Likely cause |
|---|---|
| `[preencode] matched X/N rows; N - X missing` is large | Metadata `scan` column wasn't generated with `enumerate()` over the same MGF, or you have a different MGF |
| `[preencode] matched X/N where X > N` | Old bug — metadata scans aren't unique; rebuild metadata |
| Epoch takes hours on GPU | `MAX_PEAKS` is too high or unset; one outlier spectrum is blowing up batches |
| `Using device: cpu` | CUDA not visible to PyTorch — model will train but very slowly |
| val_loss starts rising while train_loss falls | Overfitting; increase data or `DROPOUT` |
| Both curves overlap after many epochs | Model isn't learning; check data integrity (matched rows, scan alignment) |
