# Transformer-Encoder Contrastive Spectrum Embedder

Trains a [depthcharge](https://github.com/wfondrie/depthcharge) `SpectrumTransformerEncoder`
in a Siamese setup to embed MS/MS spectra so that spectra of the **same peptide**
land close together in latent space and spectra of **different peptides** land far
apart. The learned distance can then be used for spectrum clustering / search, the
same downstream task GLEAMS targets.

This is a re-implementation of the GLEAMS idea (Bittremieux et al., *Nature
Communications* 2022) with two deliberate differences:

1. The backbone is a **transformer over variable-length peak sequences** instead
   of a CNN over a fixed-size binned vector.
2. The embedding is read off the transformer's **global spectrum token** rather
   than a final dense layer, but it is still trained with the same Hadsell
   contrastive loss.

The folder also contains a full **CNN-vs-transformer holdout comparison harness**
so the two embedders can be scored head-to-head on identical pairs with
model-agnostic separation metrics (this is the core empirical result for the
report — see [Comparison harness](#the-cnn-vs-transformer-comparison-harness)).

---

## Table of contents

- [Two Python environments](#two-python-environments)
- [Requirements](#requirements)
- [The dataset](#the-dataset)
- [Folder layout](#folder-layout)
- [Key data concepts](#key-data-concepts-read-this-first)
- [End-to-end pipeline](#end-to-end-pipeline)
- [Stage 1 — data preparation (`prepare_data.py`)](#stage-1--data-preparation-prepare_datapy)
- [Stage 2 — the streaming data path (`spectrum_pairs.py`)](#stage-2--the-streaming-data-path-spectrum_pairspy)
- [The model (`model.py`)](#the-model-modelpy)
- [Stage 3 — training (`training.py`, `train.py`)](#stage-3--training-trainingpy-trainpy)
- [Evaluation metrics (`metrics.py`)](#evaluation-metrics-metricspy)
- [Visualization (`visualization.py`)](#visualization-visualizationpy)
- [The CNN-vs-transformer comparison harness](#the-cnn-vs-transformer-comparison-harness)
- [Inspecting the data (`inspect_spectra.py`)](#inspecting-the-data-inspect_spectrapy)
- [How to run (quick reference)](#how-to-run-quick-reference)
- [Output structure](#output-structure)
- [Hyperparameter cheat sheet](#hyperparameter-cheat-sheet-configpy)
- [Results so far](#results-so-far)
- [Possible improvements](#possible-improvements)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Two Python environments

The project spans **two conda environments** because the transformer and the
original CNN have incompatible deep-learning stacks:

| Env | Stack | Used for |
|---|---|---|
| `base` | PyTorch (CPU-only, `2.9.1+cpu`) + depthcharge | Transformer code, editing, CPU smoke tests |
| `gleams` | PyTorch `2.3.1+cu121` **(CUDA)** + depthcharge + TensorFlow 2.17 | GPU transformer training **and** the CNN — everything that needs speed |

**Run all real training and the comparison in the `gleams` env.** The `base`
env's torch is CPU-only (~75× slower per batch), whereas the `gleams` env already
carries a CUDA torch build that drives the RTX 3080 Ti *and* the TensorFlow stack
the CNN needs, so a single env runs both sides of the comparison on the same GPU.

```bash
# GPU training / comparison — use the gleams interpreter explicitly:
/home/saartje/miniconda3/envs/gleams/bin/python train.py
```

`training.py` auto-selects `cuda` when it is visible, so nothing in the code needs
changing — only the interpreter you launch with.

---

## Requirements

Pinned versions of the two environments actually used (checkpoints load across
them with no key mismatch, since the depthcharge versions are adjacent):

| Env | Python stack | Key packages |
|---|---|---|
| `gleams` (GPU + CNN) | `torch 2.3.1+cu121`, `depthcharge 0.4.8`, TensorFlow 2.17 | `pyteomics`, `spectrum_utils`, `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow`, `tqdm` |
| `base` (CPU torch) | `torch 2.9.1+cpu`, `depthcharge 0.4.9` | same scientific stack |

Both share the common scientific Python stack (NumPy / pandas / SciPy /
matplotlib / pyarrow / tqdm). The CNN side additionally needs the **`gleams`
package itself** (this folder lives inside the `gleams/` tree and reuses its
`feature`, `metadata`, `rndm`, and `config` modules) and `spectrum_utils` — note
the ≥ 0.4 normalization behaviour change handled in `compare_cnn.py`
(see [CNN side](#cnn-side-compare_cnnpy-tf-gleams-env)).

Hardware used: a single **NVIDIA RTX 3080 Ti (12 GB)**; ~20 CPU cores (drives
`NUM_WORKERS`); the box runs under WSL2, which is why streaming (rather than
loading peaks into RAM) matters — see [Stage 2](#stage-2--the-streaming-data-path-spectrum_pairspy).

---

## The dataset

Spectra come from the **MassIVE-KB** peptide spectral library, bundled as one
concatenated MGF, `massivekb_82c0124b.mgf` (**≈ 129 GB**). The original GLEAMS CNN
weights (`gleams_82c0124b.hdf5`, 26 MB) come from the same `82c0124b` build, so
CNN and transformer are compared on data drawn from the identical source.

After `build_metadata` (keeping 2–10 replicates per `(sequence, charge)` group),
the combined metadata (`massivekb.parquet`) contains:

| | |
|---|---|
| Spectra | **9,732,868** |
| Unique peptide sequences | **1,083,765** |
| `(sequence, charge)` groups | **1,467,924** (0 singletons — the `MIN_REPLICATES` filter) |
| Charge states | 2–5 |

**Spectra per charge:** charge 2 → 4,664,310 · charge 3 → 3,739,386 · charge 4 →
1,111,896 · charge 5 → 217,276. Charge 2 dominates, which is exactly why the
training pair caps are applied **per charge** — otherwise charge 2 would swamp the
rarer charges.

**Replicate cap is active.** The `MAX_REPLICATES = 10` cap bites hard: **663,721**
groups sit at exactly 10 replicates (vs. 308,713 at the minimum of 2), so without
the cap a minority of very common peptides would contribute a hugely
disproportionate share of positive pairs (`C(n, 2)` grows quadratically).

**Pair scale (charges 2–5 combined).** Positive pairs are all same-peptide
`C(n, 2)` combinations; negatives are precursor-confusable but
fragment-dissimilar pairs. The counts are very lopsided toward negatives:

| Split | Positive pairs | Negative pairs |
|---|---|---|
| train (~80%) | ~29.0 M | ~978 M |
| val / test (~10% each) | ~7.2 M each | ~61 M each |

On disk the charge-2 **train negatives alone are 4.8 GB** — this is the concrete
reason `_load_pairs_all_charges` memory-maps the negative files and subsamples
only the rows it keeps, and why training never loads all pairs at once. (Exact
per-split, per-charge counts are printed by
[`inspect_spectra.py`](#inspecting-the-data-inspect_spectrapy).)

### `data/` directory contents

```
data/
├── massivekb_82c0124b.mgf              # ≈129 GB — the bundled source spectra
├── massivekb_82c0124b.mgf.beginions.npy# 233 MB — cached byte-offset index (built on first run)
├── massivekb.parquet                   # 195 MB — combined metadata (pre-split)
├── {train,val,test}_metadata.parquet   # columns: scan, rtinseconds, sequence, mz, charge
├── {train,val,test}_metadata_pairs_{pos,neg}_{2,3,4,5}.npy   # (N, 2) row-index pairs
└── gleams_82c0124b.hdf5                # 26 MB — original GLEAMS CNN weights (comparison)
```

The `{train,val,test}_spectra.npz` paths in `config.py` are the legacy
pre-encoded caches; the streaming trainer no longer writes them (only the
comparison's transformer side pre-encodes, into its own run folder).

---

## Folder layout

| File | Role |
|---|---|
| `config.py` | All editable hyperparameters and paths. The only file you normally edit. |
| `prepare_data.py` | Three-stage, cached data-prep: build metadata → peptide-level split → generate pairs. |
| `spectrum_pairs.py` | MGF → tensors. Byte-offset index, the **streaming** dataset used for training, a legacy pre-encoded-`.npz` dataset used by the comparison, and the batch collate/pad. |
| `model.py` | `ContrastiveLoss` + `CustomSpectrumTransformerEncoder` (depthcharge subclass that bakes precursor m/z + charge into the global token). |
| `metrics.py` | Model-agnostic separation metrics (AUC, FNR@FDR, EER) — the numbers that are actually comparable across models. Pure NumPy. |
| `training.py` | `train_model(run_dir)` — builds streaming loaders, model, optimizer; runs the loop; writes checkpoints, CSV log, plots. |
| `visualization.py` | Plot helpers: per-epoch distance KDEs + the live loss-curve plot. |
| `train.py` | Entry point. Creates `results/<timestamp>/`, tees stdout to a log, calls `train_model`. |
| `compare_common.py` | Shared, pure-NumPy plumbing for the CNN-vs-transformer comparison (pair selection, distance I/O, report/plot). Imports **neither** torch nor TF nor `config`, so it loads in both envs. |
| `compare_transformer.py` | Transformer side of the comparison (torch env): embed the holdout spectra, write per-pair distances. |
| `compare_cnn.py` | CNN side of the comparison (TF `gleams` env): rebuild the original GLEAMS embedder, embed the same spectra, write per-pair distances. |
| `compare_report.py` | Score both models on the pairs both could embed; write `comparison.csv` + `comparison.png`. |
| `run_comparison.sh` | One command that runs all three comparison steps into one run folder. |
| `inspect_spectra.py` | Diagnostic report on the metadata / grouping / replicates / split / pair counts. Writes `inspect_spectra.log`. |
| `__init__.py` | Empty — marks the folder as a Python package. |

---

## Key data concepts (read this first)

Three ideas underpin the whole pipeline. Getting these wrong silently corrupts
training, so they are worth stating up front.

### 1. A spectrum's identity is its **enumerate position** in the MGF

The bundled MGF (`massivekb_82c0124b.mgf`) concatenates spectra from many source
files. Each source file's scan numbers restart from 1, so the `scan:1856` in an
MGF `TITLE` is **not unique** across the bundle, and there is no `SCANS=` field.

So every part of the pipeline keys a spectrum by its **position in the enumerate
order** of `pyteomics.mgf.read(..., use_index=False)`:

```python
with mgf.read(mgf_path, use_index=False) as reader:
    for scan_idx, spectrum in enumerate(reader):
        row['scan'] = scan_idx        # 0, 1, 2, ... across the WHOLE bundle
```

`prepare_data.py` stores this position as the metadata `scan` column, and the
byte-offset index (below) is keyed on the same position. Everything lines up
because everything counts spectra the same way.

### 2. Pair files reference **metadata row indices**, not scans

A positive/negative pair file is an `(N, 2)` array. A row `[42, 117]` means
"metadata row 42 pairs with metadata row 117." The lookup chain at training time
is:

```
pair row index 42  →  metadata.iloc[42]['scan']  →  enumerate position in MGF  →  peaks
```

Because the `scan` column *is* the MGF position, the dataset can go straight from
a pair's row index to the spectrum's bytes.

### 3. The split is **peptide-level**, three-way (train / val / test)

`split_metadata` partitions on the **I→L-normalized peptide sequence** so that
every spectrum of a given peptide (all charges, all replicates) lands in exactly
one split. Ratios default to **80 / 10 / 10** by peptide.

**Why peptide-level and not row-level?** A naive row-level `train_test_split`
lets different replicates of the *same* peptide fall on both sides. The model
then "recognises" a peptide it memorised in training when it reappears in
validation, inflating the val/test metrics — classic leakage. Splitting by
peptide (the same identity the pair generators use, hence the I→L normalization
so isoleucine/leucine twins can't be separated) removes that leakage. Because
replicate counts are similar across peptides, the *spectrum* fractions land near
80/10/10 rather than exactly. `inspect_spectra.py` quantifies exactly how much
the old row-level split leaked.

**How the three splits are used** (this matches the operative code):

| Split | File | Role |
|---|---|---|
| **train** | `train_metadata.parquet` | Fits the model. |
| **test** | `test_metadata.parquet` | **In-training monitor**: per-epoch loss, LR schedule, early stopping, epoch plots. |
| **val** | `val_metadata.parquet` | **Final holdout** — untouched during training, used only by the comparison harness. |

(The `test`/`val` naming is a little counter-intuitive: `training.py` monitors on
`test` and leaves `val` for the head-to-head comparison. The important invariant
is that the comparison's holdout never influenced training.)

---

## End-to-end pipeline

```
config.py ── paths + hyperparameters for everything below
   │
   ▼
prepare_data.py            (once, cached)
   ├─ build_metadata   → massivekb.parquet          (walk MGF, keep 2–10 replicates/peptide)
   ├─ split_metadata   → {train,val,test}_metadata.parquet   (peptide-level 80/10/10)
   └─ prepare_pairs    → *_pairs_{pos,neg}_{charge}.npy       (gleams pair generators)
   │
   ▼
train.py → train_model(run_dir)     (results/<timestamp>/)
   ├─ build_mgf_offset_index         → massivekb...mgf.beginions.npy   (once, cached)
   ├─ StreamingSpectrumPairDataset   (train + test, peaks read from MGF on demand)
   ├─ CustomSpectrumTransformerEncoder + ContrastiveLoss
   └─ per epoch: train → plot → evaluate(test) → plot → metrics → CSV → checkpoint → LR step → early stop
   │
   ▼
run_comparison.sh                    (comparisons/<timestamp>/)
   ├─ compare_cnn.py          → cnn_distances.npz          (CNN embeds the val holdout)
   ├─ compare_transformer.py  → transformer_distances.npz  (transformer embeds the SAME pairs)
   └─ compare_report.py       → comparison.csv + comparison.png   (score on common pairs)
```

---

## Stage 1 — data preparation (`prepare_data.py`)

Three stages, each of which **skips itself if its output already exists** (set
`OVERWRITE = True` at the top of the file to force a full rebuild). A normal
rerun is cheap: it prints "skipping" for every stage.

### `build_metadata` — walk the MGF, collect per-spectrum metadata

Streams the whole MGF once and, for every spectrum, extracts
`scan, rtinseconds, sequence, pepmass, charge` (`pepmass` is renamed to `mz` in
the saved parquet). It keeps between `MIN_REPLICATES` and `MAX_REPLICATES` spectra
per `(sequence, charge)` group:

- **`MIN_REPLICATES = 2`**: a peptide seen only once contributes **no positive
  pair** (you need ≥2 spectra to form a same-peptide pair), so singletons are
  dropped as a quality filter.
- **`MAX_REPLICATES = 10`**: caps each group so one ultra-common peptide can't
  dominate the dataset or explode its `C(n, 2)` positive-pair count.

`SPECTRA_TARGET` stops the walk once that many *committed* spectra (spectra in
groups that already crossed `MIN_REPLICATES`) have accumulated; set it to `None`
to process the whole MGF. Output is buffered to CSV and flushed every
`FLUSH_EVERY` peptides so memory doesn't grow with popular peptides, then
converted to parquet.

### `split_metadata` — peptide-level 80/10/10

See [Key concept 3](#3-the-split-is-peptide-level-three-way-train--val--test).
Asserts the splits are disjoint and lose no rows, and prints per-split spectrum
and sequence counts.

### `prepare_pairs` — positive + negative pairs per charge

Delegates to the original GLEAMS generators (`gleams.metadata.generate_pairs_*`)
for the train, val, and test parquets, over every charge in
`CHARGES = (2, 5)` inclusive:

- **Positive pairs**: all `C(n, 2)` same-`(sequence, charge)` combinations.
- **Negative pairs**: candidate different-peptide pairs whose precursors fall
  within `PAIR_MZ_TOLERANCE` ppm (so they are *confusable* — hard negatives), but
  whose theoretical b/y fragment overlap stays under
  `NEGATIVE_PAIR_MATCHING_FRAGMENTS_THRESHOLD` (so genuinely near-identical
  peptides aren't mislabelled negative). `NEGATIVE_PAIR_FRAGMENT_TOLERANCE` sets
  the Da window for declaring two theoretical ions "the same."

Reusing GLEAMS's own pair generators (rather than writing new ones) means the
transformer trains and is evaluated on **exactly the pairs the CNN would see** —
essential for a fair comparison.

Output files: `<split>_metadata_pairs_{pos,neg}_<charge>.npy`.

---

## Stage 2 — the streaming data path (`spectrum_pairs.py`)

Training must read peaks for millions of pairs across many epochs. There are two
implementations here; **training uses the streaming one**, the comparison uses the
legacy pre-encoded one.

### The byte-offset index (`build_mgf_offset_index`)

Scans the raw MGF bytes once for every `BEGIN IONS` marker and records its byte
offset, indexed by spectrum enumerate position. Cached next to the MGF as
`<mgf>.beginions.npy`. This lets any later run **seek straight to a spectrum's
bytes** instead of parsing the whole multi-hundred-GB file. The scan is
I/O-bound (no spectrum parsing), chunked at 64 MB, and carefully carries a few
bytes between chunks so a marker split across a read boundary is still caught.

### `StreamingSpectrumPairDataset` — the training dataset

Reads each spectrum's peaks **on demand** by seeking to its byte offset. Only the
pair list, the per-row scalars (`scan`, `pepmass`, `charge`), and the offset index
(~244 MB for the full MassIVE MGF) live in RAM.

**Why streaming instead of pre-encoding everything into one `.npz`?** The old
approach packed every referenced spectrum's peaks into an in-RAM array before
training. At the scale we now train (millions of pairs, all charges) that built
tens of GB of peak arrays and OOM-killed the WSL VM. Streaming makes memory
proportional to the *pair list*, not the peak data, so training scales past that
ceiling.

**The pid-aware file handle (a subtle correctness point).** Under a
`DataLoader` with `num_workers > 0`, the dataset is forked into worker processes.
A file handle opened *before* the fork would be shared across workers, and their
concurrent `seek()`s would clobber a single shared file offset — returning garbage
bytes that span spectrum boundaries. So `_handle()` reopens the file whenever the
pid changes, giving every process its own independent offset. This is why `_fh`
**must** start as `None`.

### `SpectrumPairDataset` + `preencode_mgf_to_npzs` — the legacy path

`preencode_mgf_to_npzs` seeks the referenced spectra (via the same offset index)
and packs them CSR-style into a `.npz`; `SpectrumPairDataset` mmaps that file.
This whole-cache-in-RAM approach is fine at the comparison's scale (tens of
thousands of spectra) and is what `compare_transformer.py` uses, so it is kept.

`.npz` format:

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `mz_flat` | float32 | (total peaks,) | All m/z values concatenated |
| `int_flat` | float32 | (total peaks,) | All intensities concatenated |
| `offsets` | int64 | (N+1,) | Row `i`'s peaks are `mz_flat[offsets[i]:offsets[i+1]]` |
| `pepmass` | float32 | (N,) | Precursor m/z per row |
| `charge` | int16 | (N,) | Precursor charge per row |

Rows whose scan wasn't found get length 0; the dataset drops any pair touching
such a row.

### Peak capping (`MAX_PEAKS`) — why

Both datasets keep only the `MAX_PEAKS` (default 150) most intense peaks per
spectrum, then re-sort them by m/z (so any downstream sorted-m/z assumption
holds). Self-attention is **O(L²)** in sequence length: one 2000-peak outlier in
a batch forces the whole batch's attention matrix to 2000×2000 and destroys
throughput. Capping bounds the worst case, and the most intense peaks carry
almost all of a spectrum's identity, so little signal is lost.

### `collate_pairs` — batching

Pads every spectrum in a batch to the longest peak count with zeros (the natural
attention mask — depthcharge ignores padded positions), and stacks the precursor
`pepmass`/`charge` scalars to shape `(batch, 1)` because depthcharge's
`FloatEncoder` expects a feature dimension.

---

## The model (`model.py`)

### Architecture at a glance

The model is trained **Siamese**: the *same* encoder (shared weights) embeds both
spectra of a pair, and the contrastive loss acts on the distance between the two
embeddings.

```
        Spectrum A                                     Spectrum B
 ┌───────────────────────────┐               ┌───────────────────────────┐
 │ peaks:  (m/z, intensity)  │               │ peaks:  (m/z, intensity)  │
 │ precursor m/z, charge     │               │ precursor m/z, charge     │
 └───────────────────────────┘               └───────────────────────────┘
              │                                            │
   ┌──────────┴───────────┐                     (identical tower,
   │  per-peak embedding   │                      shared weights)
   │  (depthcharge peak    │                            │
   │   m/z + intensity     │                            │
   │   encoders) → tokens  │                            │
   └──────────┬───────────┘                            │
              │  ┌───────────────────────────┐          │
  precursor → │  │ global token              │          │
  m/z, charge │  │  = FloatEncoder(pep m/z)  │          │
     ─────────┼─►│  + FloatEncoder(charge)   │          │
              │  └───────────────────────────┘          │
              ▼                                          ▼
   ┌──────────────────────┐                  ┌──────────────────────┐
   │ [global | peak tokens]│                  │ [global | peak tokens]│
   │  → N_LAYERS transformer│                 │  → N_LAYERS transformer│
   │    encoder layers      │                 │    encoder layers      │
   │  (self-attention over  │                 │  (self-attention over  │
   │   peaks; pad = mask)   │                 │   peaks; pad = mask)   │
   └──────────┬───────────┘                  └──────────┬───────────┘
              │  take global token                       │
              ▼  emb_full[:, 0, :]                        ▼
        emb_A  ∈ ℝ^DIM_MODEL                       emb_B ∈ ℝ^DIM_MODEL
              └──────────────┐          ┌──────────────────┘
                             ▼          ▼
                       d = ‖emb_A − emb_B‖₂   (Euclidean)
                             │
                             ▼
             ContrastiveLoss(d, label, margin=MARGIN)
        label=1 (same peptide) → pull together   |   label=0 (different) → push apart to MARGIN
```

Defaults: `DIM_MODEL=256`, `N_HEAD=4`, `N_LAYERS=4`, `DIM_FEEDFORWARD=512`,
`DROPOUT=0.1`. Each peak is a token — there is no positional encoding over peaks
(a spectrum is an unordered set), and the padded positions from `collate_pairs`
act as the attention mask. At inference/evaluation you run **one** tower to get a
spectrum's embedding; the Siamese pairing only exists to compute the training
loss.

### `CustomSpectrumTransformerEncoder`

A thin subclass of depthcharge's `SpectrumTransformerEncoder`. Each
`(m/z, intensity)` peak is one token; tokens pass through `N_LAYERS` self-attention
layers of width `DIM_MODEL` with `N_HEAD` heads and a `DIM_FEEDFORWARD` MLP.

The subclass overrides `global_token_hook` to inject **precursor m/z and charge**
into the model's special global token: both scalars go through a learned
`FloatEncoder` (sinusoidal-style float embedding) and are summed into the global
token's initial embedding. The precursor is highly discriminative between
different peptides, so feeding it in explicitly — rather than hoping the model
infers it from fragment peaks — is a strong prior. The CNN's `PrecursorEncoder`
does the analogous thing, so this keeps the two models informationally comparable.

The output embedding for a spectrum is the global token: `emb_full[:, 0, :]`, a
single `DIM_MODEL`-dimensional vector.

### `ContrastiveLoss` (Hadsell)

```python
L = label · d²  +  (1 − label) · max(0, margin − d)²      #  d = ‖emb1 − emb2‖₂
```

- **Positive pair (`label = 1`)** → loss is `d²`, pulling the two embeddings
  together.
- **Negative pair (`label = 0`)** → loss is zero once `d ≥ margin`; otherwise it
  pushes them apart toward `MARGIN`.

This is the same loss family GLEAMS uses, which is why the swap is "transformer
backbone, same training objective."

---

## Stage 3 — training (`training.py`, `train.py`)

`train.py` mints `results/<YYYY-MM-DD_HH-MM-SS>/`, tees all stdout/stderr to
`training.log` (so tracebacks are captured even under a crash), and calls
`train_model(run_dir)`.

`train_model` does:

1. **Load metadata + pairs.** Loads the `train` and `test` (monitor) parquets.
   `_load_pairs_all_charges` loads `*_pairs_{pol}_{charge}.npy` for every charge
   in `CHARGES` and concatenates them, applying the cap
   (`MAX_TRAIN_PAIRS_PER_CLASS` / `MAX_TEST_PAIRS_PER_CLASS`) **per charge** so
   the low charges aren't swamped by the far more numerous charge-2 pairs. Big
   negative files are memory-mapped so only the sampled rows are materialised.
2. **Build the offset index** (cached) and two `StreamingSpectrumPairDataset`s,
   then `DataLoader`s with `collate_pairs`, `NUM_WORKERS` workers, pinned memory,
   and persistent workers.
3. **Build model + optimizer.** Adam (`LEARNING_RATE`, `WEIGHT_DECAY`),
   `ContrastiveLoss(MARGIN)`, `ReduceLROnPlateau` scheduler. Auto-selects CUDA.
4. **Epoch loop** (up to `MAX_EPOCHS`):
   - Train all batches; collect per-pair distances split by polarity.
   - Save `plots/train_epoch_NNN.png` (distance KDE).
   - `evaluate_contrastive` on the test/monitor loader → `plots/test_epoch_NNN.png`.
   - Compute the [separation metrics](#evaluation-metrics-metricspy) on both
     splits' distances.
   - Write one `train_epoch` and one `test_epoch` row to `loss_log.csv`, each
     stamped with **all hyperparameters** + the metrics (the CSV is fully
     self-describing, so runs can be compared without knowing the config used).
   - Refresh `loss_curves.png`.
   - `scheduler.step(test_loss)` — halve LR after `LR_SCHEDULER_PATIENCE` epochs
     with no monitor improvement.
   - Save `best_model.pt` whenever the monitor loss improves.
   - **Early stopping**: break after `EARLY_STOPPING_PATIENCE` consecutive epochs
     with no improvement (set to `None` to always train `MAX_EPOCHS`). It is set
     slightly above `LR_SCHEDULER_PATIENCE` so the LR cut gets a chance to help
     before training gives up.
5. Save `final_model.pt`.

### Checkpoint contents

```python
ckpt = torch.load('results/.../best_model.pt', weights_only=False)
ckpt['model_state_dict']    # weights
ckpt['hyperparameters']     # {dim_model, n_layers, n_head, dim_feedforward, dropout, margin}
ckpt['test_loss']           # monitor loss when this checkpoint was saved (best)
ckpt['train_loss']          # train loss at that epoch
ckpt['optimizer_state_dict'], ckpt['scheduler_state_dict'], ckpt['epoch']
```

To reload: rebuild the model from `ckpt['hyperparameters']`, then
`model.load_state_dict(ckpt['model_state_dict'])` — exactly what
`compare_transformer.py` does.

---

## Evaluation metrics (`metrics.py`)

**Why the loss is not a fair comparison metric.** The contrastive *loss* depends
on the loss formulation, the margin value, and the positive/negative balancing —
all of which differ between the CNN and this transformer. Comparing raw losses
would compare bookkeeping, not embedding quality. So the comparison scores only
the **separation** between positive and negative pair distances, which is directly
comparable between any two embedders evaluated on the same pairs. These metrics
also mirror the paper's own evaluation (Extended Data Fig. 3–4).

All functions take two 1-D arrays — `pos_dist` (same-peptide) and `neg_dist`
(different-peptide) — and are pure NumPy (no scikit-learn), so they import in both
envs. Same-peptide pairs should have *smaller* distances.

| Metric | What it measures | Better |
|---|---|---|
| `auc` | P(a random positive pair is closer than a random negative pair), via the Mann-Whitney U statistic — no threshold sweep, tie-averaged ranks. 0.5 = chance, 1.0 = perfect. | Higher |
| `fnr_at_fdr` | The **false-negative rate at ≤ `fdr_target` (1%) FDR**: sweep the distance threshold, take the largest cutoff whose FDR stays ≤ target, and report the fraction of positives missed. This is the paper's "1% FNR at 1% FDR" headline. | Lower |
| `fdr_threshold` | The distance cutoff chosen for the FDR target above. | (context) |
| `eer` | Equal error rate — the threshold-free rate where FPR = FNR. | Lower |

`ranking_metrics(pos_dist, neg_dist)` returns all of them in one dict, keyed for
CSV logging. Feed it the CNN's and the transformer's distances to get numbers you
can put side by side.

---

## Visualization (`visualization.py`)

Headless (`Agg` backend), so it is safe under stdout-teeing and cron.

- **`plot_distance_distributions`** — KDE of positive (pink) vs negative (blue)
  embedded distances, saved per epoch. **Reading it** (mirrors the paper's
  Extended Data Fig. 3): as training progresses you want the positive peak to
  migrate toward 0, the negative peak to move up to/past `MARGIN`, and the
  crossover between them to shift right (less overlap). Identical curves ⇒ the
  model isn't learning. Great on train but poor on test ⇒ overfitting.
- **`plot_loss_curves`** — train + test loss vs epoch, re-read from
  `loss_log.csv` and **overwritten every epoch**, so opening `loss_curves.png` in
  a viewer that auto-refreshes (e.g. `eog`) gives a live training monitor. A
  two-line hyperparameter subtitle is stamped underneath so the plot is
  self-documenting. Callable standalone to re-plot any past run:
  `plot_loss_curves('results/.../loss_log.csv', 'out.png')`.

---

## The CNN-vs-transformer comparison harness

This is the core empirical deliverable: score the transformer and the original
GLEAMS CNN **head-to-head** on the same holdout pairs, with the model-agnostic
metrics above.

### The protocol (`compare_common.py`)

`compare_common.py` is deliberately **pure NumPy + matplotlib** — it imports
neither torch, nor TensorFlow, nor `config` (the two packages each ship a
`config.py` and they collide). That lets it load unchanged in both the torch env
and the TF env.

1. Each comparison runs in one timestamped folder under
   `comparisons/<timestamp>/` (mirroring `results/`). All three steps share it.
2. **`select_pairs`** deterministically subsamples the same `val` positive/negative
   pairs for both evaluators. Same `(file, cap, seed)` ⇒ identical arrays in
   identical order (`DEFAULT_CAP = 25000` per class, `DEFAULT_SEED = 0`), so the
   two models' distance arrays line up **pair-for-pair**.
3. Each evaluator embeds the referenced spectra with its own model and writes an
   `.npz` of per-pair Euclidean distances aligned to the selection order (**NaN**
   where a spectrum was dropped), plus a `meta` record naming the
   model/checkpoint.
4. **`build_report`** loads every model's distances, keeps only pairs that are
   **finite for *all* models** (so every model is scored on the same subset),
   computes `ranking_metrics` for each, and writes `comparison.csv` +
   `comparison.png` (a KDE panel per model, each on its own x-scale since the two
   models produce distances on very different scales).

### Transformer side (`compare_transformer.py`, torch env)

Loads a checkpoint (default is a scoped `best_model.pt`; override with
`--checkpoint`), pre-encodes just the referenced holdout spectra with
`preencode_mgf_to_npzs` + `_compact_referenced_spectra` (so only the needed
spectra are seeked/encoded), embeds them, and writes `transformer_distances.npz`.

### CNN side (`compare_cnn.py`, TF `gleams` env)

Rebuilds the **original GLEAMS embedder** in-process (precursor + fragment VGG-style
CNN + reference-spectra arms) and `load_weights` from `gleams_82c0124b.hdf5`,
rather than `keras.models.load_model` (which is pathologically slow deserializing
the TF-2.2-era HDF5 on this TF 2.17 install). It encodes the same spectra with the
gleams feature encoders and writes `cnn_distances.npz`.

Two correctness details worth calling out for the report:

- **Reference-spectra normalization fix.** The `ReferenceSpectraEncoder` stores
  its reference spectra after preprocessing, relying on `spectrum_utils` 0.3.4
  (as in the paper) preprocessing **in place**. Under the installed 0.4.1,
  `preprocess` returns a *new* normalized object and leaves the original raw, so
  the stored references kept raw intensities and the reference dot-products blew
  up ~1000×. `compare_cnn._build_encoder` re-preprocesses the stored references to
  restore normalized intensities. **This bug was responsible for an earlier,
  wrong result** that made the CNN look far weaker than it is — fixing it is what
  produced the corrected numbers below.
- **Seed before encoder construction.** `rndm.set_seeds(42)` is called *before*
  building the encoder because the 500 reference spectra are selected in a
  seed-dependent order, and the feature vector is order-sensitive; this reproduces
  the exact feature layout the CNN was trained with.

### Running it (`run_comparison.sh`)

```bash
./run_comparison.sh [TRANSFORMER_CKPT] [CNN_HDF5]
```

Mints one run folder and runs, in the `gleams` env (which has both stacks + the
GPU): CNN eval → transformer eval → report. Outputs land in
`comparisons/<timestamp>/`: `comparison.csv` and `comparison.png`, each stamped
with which weights/checkpoint produced them.

---

## Inspecting the data (`inspect_spectra.py`)

A read-only diagnostic that reports, from the artifacts in `data/`:

1. Combined-metadata overview (columns, spectrum/sequence counts).
2. Grouping by `(sequence, charge)`; replicate-count distribution; spectra per
   charge.
3. **The leakage contrast**: it computes overlap for the *existing* split vs a
   fresh peptide-level split and shows how many sequences leak across train/val —
   the empirical justification for the peptide-level split.
4. Positive-pair counts computed directly from the metadata (`C(n, 2)` per
   `(sequence, charge)`, I→L normalized) vs the on-disk pair-file counts.

```bash
python inspect_spectra.py                # report → terminal + inspect_spectra.log
python inspect_spectra.py --write-split  # also materialize peptide-level split parquets
python inspect_spectra.py --no-log       # terminal only
```

---

## How to run (quick reference)

```bash
cd /home/saartje/Desktop/Research/GLEAMS/gleams/transformer_encoder

# 1. One-time (or after changing data-prep params): metadata + split + pairs.
python prepare_data.py

# 2. (optional) sanity-check the data + split leakage.
python inspect_spectra.py

# 3. Train on the GPU. First run also builds the MGF offset index (cached).
/home/saartje/miniconda3/envs/gleams/bin/python train.py

# 4. Compare the trained transformer against the original CNN.
./run_comparison.sh
```

Change hyperparameters by editing `config.py` and rerunning — each `train.py`
run gets its own `results/<timestamp>/` so sweeps never collide.

### When to rebuild cached artifacts

| Change | Rebuild offset index? | Rebuild pairs / metadata? |
|---|---|---|
| Pair caps, `MAX_PEAKS`, model/optimizer hyperparameters | No | No |
| `CHARGES`, replicate/split/negative-pair params | No | Yes (`OVERWRITE = True` or delete outputs) |
| A different MGF | Yes (delete `<mgf>.beginions.npy`) | Yes |

---

## Output structure

```
results/2026-05-19_22-44-29/
├── training.log            # full console output + tracebacks
├── loss_log.csv            # 2 rows/epoch (train_epoch, test_epoch); all hyperparams + metrics
├── loss_curves.png         # train + test loss vs epoch (refreshed every epoch)
├── best_model.pt           # checkpoint with the best monitor loss
├── final_model.pt          # checkpoint at end of training
└── plots/
    ├── train_epoch_001.png # KDE of positive vs negative distances (train)
    ├── test_epoch_001.png  # same, on the monitor split
    └── ...

comparisons/2026-07-09_.../
├── cnn_distances.npz            # CNN per-pair distances (+ meta)
├── transformer_distances.npz    # transformer per-pair distances (+ meta)
├── comparison.csv               # both models' AUC / FNR@FDR / EER on common pairs
└── comparison.png               # one distance-KDE panel per model
```

`loss_log.csv` columns: `timestamp, epoch, batch, split, loss, n_pairs,
batch_size, max_peaks, dim_model, n_layers, n_head, dim_feedforward, dropout, lr,
margin, auc, fnr_at_fdr, fdr_threshold, eer`. The `*_epoch` rows carry the metrics;
per-batch rows are only written when `LOG_BATCH_LEVEL = True`.

---

## Hyperparameter cheat sheet (`config.py`)

| Knob | Default | When to change |
|---|---|---|
| `MAX_TRAIN_PAIRS_PER_CLASS` | 100,000 | Per charge & polarity. Highest-impact lever; raise (or set `None`) once a smoke test works and the GPU is in use. |
| `MAX_TEST_PAIRS_PER_CLASS` | 10,000 | Monitor-set size; larger = smoother but slower per-epoch eval. |
| `MAX_PEAKS` | 150 | Lower → faster batches; higher → more spectrum info (quadratic attention cost). |
| `BATCH_SIZE` | 128 | Lower if you OOM with bigger `MAX_PEAKS`/`DIM_MODEL`; the 3080 Ti (12 GB) can likely go higher. |
| `NUM_WORKERS` | 8 | Streaming reads peaks per item, so more workers = more parallel disk seeks. Box has 20 cores. |
| `DIM_MODEL` | 256 | Increase (384, 512) once data is large enough to use the capacity. |
| `N_HEAD` / `N_LAYERS` / `DIM_FEEDFORWARD` | 4 / 4 / 512 | Deeper/wider helps with more data. |
| `DROPOUT` | 0.1 | Higher if test loss diverges above train; lower with more data. |
| `LEARNING_RATE` | 1e-4 | Lower if loss is unstable. |
| `WEIGHT_DECAY` | 1e-5 | L2 regularization. |
| `MARGIN` | 2.0 | Raise if negatives cluster well below it after long training. |
| `MAX_EPOCHS` | 30 | Hard ceiling; training usually stops earlier via early stopping. |
| `EARLY_STOPPING_PATIENCE` | 5 | Consecutive no-improvement epochs before stopping (`None` disables). |
| `LR_SCHEDULER_FACTOR` / `LR_SCHEDULER_PATIENCE` | 0.5 / 2 | LR is multiplied by factor after patience epochs of no monitor improvement. |
| `MIN_REPLICATES` / `MAX_REPLICATES` | 2 / 10 | Replicates kept per `(sequence, charge)` group (see Stage 1). |
| `SPECTRA_TARGET` | 10,000,000 | Stop the metadata walk after this many committed spectra; `None` = whole MGF. |
| `VAL_SPLIT_RATIO` / `TEST_SPLIT_RATIO` | 0.1 / 0.1 | Peptide-level split fractions; train is the remainder. |
| `CHARGES` | (2, 5) | Inclusive charge range for pair generation + training. |
| `PAIR_MZ_TOLERANCE` | 10 | ppm window for candidate (hard) negative pairs. |
| `NEGATIVE_PAIR_FRAGMENT_TOLERANCE` | 0.01 | Da window for two theoretical ions to "overlap." |
| `NEGATIVE_PAIR_MATCHING_FRAGMENTS_THRESHOLD` | 0.25 | Max fragment-overlap ratio allowed for a negative pair. |

---

## Results so far

On the charge-2 val holdout (25k pos + 25k neg pairs, ~24,995 common evaluable),
current transformer checkpoint `results/2026-07-12_01-06-35/best_model.pt` vs the
original CNN `gleams_82c0124b.hdf5`:

| metric | CNN (original) | Transformer | Better |
|---|---|---|---|
| AUC | **0.9993** | 0.9628 | higher |
| EER | **0.0123** | 0.1024 | lower |
| FNR@1%FDR | **0.0149** | 0.5705 | lower |
| FDR threshold | 0.5769 | 0.5400 | — |

(Common evaluable pairs: 24,996 positive / 24,994 negative.)

The original CNN still wins and reproduces the paper (AUC ≈ 0.99, ~1% FNR at
1% FDR). But the transformer has **closed much of the gap** thanks to the
streaming refactor + multi-charge data scaling described above — its AUC rose
from 0.9109 to 0.9628 and its EER roughly halved (0.170 → 0.102):

| metric | Transformer — earlier (`2026-05-19_22-44-29`) | Transformer — current (`2026-07-12_01-06-35`) |
|---|---|---|
| AUC | 0.9109 | **0.9628** |
| EER | 0.1700 | **0.1024** |
| FNR@1%FDR | 0.9458 | **0.5705** |
| FDR threshold | 0.227 | **0.5400** |

The remaining gap is concentrated in **FNR@1%FDR** (0.57 vs the CNN's 0.015): at
the strict 1%-FDR operating point the transformer still misses most true pairs,
i.e. its positive/negative distance distributions overlap in the tail even though
their bulk separation (AUC/EER) is now good. Notably the transformer's chosen FDR
threshold has climbed to 0.54 (from 0.23), close to the CNN's 0.58 — the embedding
scale is now paper-like. Deferred ideas that target the remaining tail overlap: an
L2-normalized projection head to fix the FNR@1%FDR calibration, sqrt-normalizing
peak intensities (the CNN feeds normalized intensities, the transformer currently
feeds raw), and in-batch / InfoNCE hard negatives.

> An even earlier result that appeared to show the transformer *winning* (CNN AUC
> ~0.83) was an artifact of the reference-normalization bug described in
> [CNN side](#cnn-side-compare_cnnpy-tf-gleams-env); the CNN's weights and
> architecture were verified against the upstream repo.

---

## Possible improvements

**Diagnosis first.** The *shape* of the remaining gap says where to aim: the
transformer's **AUC (0.963) and EER (0.102) are already close**, but
**FNR@1%FDR (0.57) is far worse than the CNN's (0.015)**. So bulk separation is
fine — the problem is the **tail / calibration**: a minority of same-peptide pairs
sit at large distances (hard positives) and/or confusable negatives sit at small
distances, dragging in the distance threshold that buys 1% FDR. The implication is
to target **preprocessing, embedding geometry, and the decision boundary** before
reaching for a bigger model.

The levers below are ordered by leverage-per-effort; #1 and #2 are cheap and worth
doing (and re-running `run_comparison.sh`) before anything else. Measure after each
so the effect is attributable.

### 1. Match the CNN's spectrum preprocessing *(cheap, high leverage)*

The clearest apples-to-oranges gap. `compare_cnn.py` runs every spectrum through
`gleams.feature.spectrum.preprocess` — **sqrt/rank intensity scaling**,
`min_intensity` denoising, **precursor-peak removal**, and a
`min_peaks`/`min_mz_range` validity filter. The transformer's
`StreamingSpectrumPairDataset._spectrum` feeds **raw intensities** and merely keeps
the top-`MAX_PEAKS` by intensity — no scaling, no denoising, no precursor removal.
Raw MS/MS intensities span orders of magnitude, so the peak encoder is dominated by
a few base peaks and attends to noise. Applying the same sqrt-scale + relative-
abundance normalization + precursor/noise removal (kept **consistent between the
streaming trainer and the comparison path** so evaluation stays fair) should
tighten the positive tail directly. *Effort: a few lines in `_spectrum`.*

### 2. L2-normalize the embedding (+ optional projection head) *(cheap, targets the tail)*

The embedding is currently the raw global token (`DIM_MODEL`-d, **unnormalized**)
with plain Euclidean distance. Unnormalized Euclidean couples *direction* (identity)
with *magnitude* (a per-spectrum nuisance that varies with peak count/intensity),
and that magnitude variance is a classic source of positive-tail spread.
L2-normalizing to the unit sphere — optionally through a small projection head
(e.g. 256→128) trained only for the contrastive loss — bounds distances to [0, 2]
and calibrates them, which is exactly what FNR@1%FDR depends on. Note this makes
`MARGIN = 2.0` meaningless, so retune the margin or switch to a cosine objective.
*Effort: small.*

### 3. A stronger contrastive objective *(more work, fixes the boundary)*

The Hadsell pairwise loss with pre-sampled negatives is weak at the boundary: most
sampled negatives are already beyond the margin, contribute **zero gradient**, and
never sharpen the region that sets the 1%-FDR threshold. Two upgrades:

- **In-batch InfoNCE / NT-Xent** — treat the other spectra in a batch as negatives
  so each anchor sees many temperature-scaled negatives per step.
- **Online hard / semi-hard negative mining** — within the batch, up-weight the
  negatives closest to the anchor.

The pair generator already produces *precursor-confusable* negatives (good), but
the current loss doesn't exploit batch structure to use them. *Effort: medium;
biggest expected payoff on FNR@1%FDR after #1–2.*

### 4. Confirm convergence / optimization before adding capacity *(medium)*

Data scaling is what moved AUC 0.91 → 0.96, so training may still be data/compute-
bound rather than capacity-bound. Before enlarging the model, read
`loss_curves.png` and the per-epoch `test_epoch` AUC/EER trend:

- Still improving at early-stop? → raise `MAX_TRAIN_PAIRS_PER_CLASS` (currently
  100k/charge of ~29M available train pairs) and/or `MAX_EPOCHS`.
- Test diverging from train? → it's overfitting tail noise, not a capacity limit.

Also consider swapping `ReduceLROnPlateau` for **LR warmup + cosine decay**
(transformers are sensitive to early-step LR; warmup often removes tail
instability). *Only then* try `DIM_MODEL` 256→384/512 or `N_LAYERS` 4→6.

### 5. Cheap architectural nits

- **Charge as an embedding, not a `FloatEncoder`.** Precursor m/z and charge
  currently share one `FloatEncoder` and are summed (`model.py`,
  `global_token_hook`). Charge is a small integer (2–5); a float encoder tuned for
  m/z ranges represents it poorly. Use a dedicated `nn.Embedding(max_charge+1,
  DIM_MODEL)` instead.
- Revisit `MAX_PEAKS = 150` once intensities are scaled/denoised — fewer, cleaner
  peaks may suffice.

### Suggested order & a diagnostic

Do **#1 + #2** together and re-run the comparison (each should visibly cut
FNR@1%FDR); then **#4** to confirm convergence; then **#3** if the strict-FDR gap
persists. A useful diagnostic for the report: histogram the **positive-pair
distances against peak count and charge** on the val set — it reveals whether the
tail comes from low-quality/low-peak spectra (→ preprocessing, #1) or genuine hard
positives (→ loss/mining, #3).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `[preencode] matched X/N rows` with large N−X missing | Metadata `scan` wasn't generated by `enumerate()` over this exact MGF, or the MGF differs. |
| `[dataset] dropping … pairs with out-of-range scans` (many) | Pair files reference a metadata file that doesn't match the split, or the offset index is stale — delete `<mgf>.beginions.npy` and retrain. |
| `Using device: cpu` during training | You launched with the CPU-only `base` interpreter — use the `gleams` env's python. |
| Epoch takes far longer than expected | `MAX_PEAKS` too high/unset, or `NUM_WORKERS` too low to keep the GPU fed. |
| Train KDE separates but test doesn't | Overfitting — add data or raise `DROPOUT`. |
| Both KDE curves overlap after many epochs | Model isn't learning — check scan alignment / pair integrity via `inspect_spectra.py`. |
| CNN distances ~1000× too large in the comparison | Reference-spectra normalization regression (`spectrum_utils` ≥ 0.4) — the fix is in `compare_cnn._build_encoder`; ensure it's applied. |
| `Distance arrays differ in length across models` | The CNN and transformer evaluators used different cap/seed — regenerate both into the same run folder. |

---

## References

- **GLEAMS.** Bittremieux, W., May, D. H., Bilmes, J., & Noble, W. S. (2022).
  *A learned embedding for efficient joint analysis of millions of mass spectra.*
  **Nature Communications**, 13, 4552. https://doi.org/10.1038/s41467-022-31936-7
  — the paper this work re-implements; source of the CNN architecture, the pair
  generation strategy, and the evaluation (Extended Data Fig. 3–4). Reference
  implementation: https://github.com/bittremieuxlab/GLEAMS
- **depthcharge.** Fondrie, W. E., *et al.* Deep-learning toolkit for mass
  spectrometry; provides `SpectrumTransformerEncoder` and the float/peak encoders
  the transformer is built on. https://github.com/wfondrie/depthcharge
- **Contrastive loss.** Hadsell, R., Chopra, S., & LeCun, Y. (2006).
  *Dimensionality reduction by learning an invariant mapping.* CVPR — the
  margin-based loss used here.
- **Dataset.** MassIVE-KB spectral library (build `82c0124b`).
  https://massive.ucsd.edu
```
