"""CNN side of the holdout comparison (run in the TensorFlow `gleams` env).

    /home/saartje/miniconda3/envs/gleams/bin/python compare_cnn.py [--run-dir DIR] [--model HDF5]

Embeds the same validation spectra referenced by the shared selected pairs with
the trained original CNN embedder, then writes per-pair Euclidean distances
(aligned to the selection order, NaN where a spectrum is dropped) into the
comparison run folder for `compare_report.py` to score against the transformer.
Pass the same --run-dir as the transformer step (the orchestrator does this).

Self-contained on purpose: it does NOT import torch or the transformer's
`config`/`training`. It reuses only the gleams feature encoders + the pure-NumPy
`compare_common`/`metrics` helpers. The 500 reference-spectra features are
order-sensitive, so `rndm.set_seeds(42)` is called *before* building the encoder
to reproduce the exact feature order the CNN was trained with.
"""

import argparse
import os
import sys

# gleams package first, so `import config`/`feature`/`ms_io`/`rndm` resolve to
# the gleams tree (its config carries the encoder settings). The transformer dir
# is only appended for the pure-NumPy compare_common/metrics modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLEAMS = os.path.dirname(_HERE)
sys.path.insert(0, _GLEAMS)
sys.path.append(_HERE)

import numpy as np
import pandas as pd
import scipy.sparse as ss
from spectrum_utils.spectrum import MsmsSpectrum

import rndm
rndm.set_seeds(42)  # MUST precede encoder construction (reference order).

import config
from feature import encoder
from feature import spectrum as gleams_spectrum

import compare_common

# ---- Data / model paths (this env has no transformer config) ----
DATA_DIR = '/home/saartje/Desktop/Research/GLEAMS/data'
VAL_METADATA_FILE = os.path.join(DATA_DIR, 'val_metadata.parquet')
VAL_POS_PAIRS_FILE = os.path.join(DATA_DIR, 'val_metadata_pairs_pos_2.npy')
VAL_NEG_PAIRS_FILE = os.path.join(DATA_DIR, 'val_metadata_pairs_neg_2.npy')
MGF_PATH = os.path.join(DATA_DIR, 'massivekb_82c0124b.mgf')
OFFSET_INDEX = MGF_PATH + '.beginions.npy'
DEFAULT_CNN_MODEL = os.path.join(DATA_DIR, 'gleams_82c0124b.hdf5')
REF_MGF = '/home/saartje/Desktop/Research/gleams_reference_spectra.mgf'

ENCODE_BATCH = 4096


def _build_encoder():
    """MultipleEncoder identical to gleams.py, but reading the local ref MGF."""
    precursor_encoding = dict(
        num_bits_mz=config.num_bits_precursor_mz,
        mz_min=config.precursor_mz_min, mz_max=config.precursor_mz_max,
        num_bits_mass=config.num_bits_precursor_mass,
        mass_min=config.precursor_mass_min, mass_max=config.precursor_mass_max,
        charge_max=config.precursor_charge_max)
    fragment_encoding = dict(
        min_mz=config.fragment_mz_min, max_mz=config.fragment_mz_max,
        bin_size=config.bin_size)
    preprocessing = dict(
        mz_min=config.fragment_mz_min, mz_max=config.fragment_mz_max,
        min_peaks=config.min_peaks, min_mz_range=config.min_mz_range,
        remove_precursor_tolerance=config.remove_precursor_tolerance,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used, scaling=config.scaling)
    reference_encoding = dict(
        filename=REF_MGF, preprocessing=preprocessing,
        fragment_mz_tol=config.fragment_mz_tol,
        num_ref_spectra=config.num_ref_spectra)
    enc = encoder.MultipleEncoder([
        encoder.PrecursorEncoder(**precursor_encoding),
        encoder.FragmentEncoder(**fragment_encoding),
        encoder.ReferenceSpectraEncoder(**reference_encoding),
    ])

    # spectrum_utils compatibility fix. ReferenceSpectraEncoder.__init__ stores
    # `spec` after `preprocess(spec).is_valid`, relying on preprocess mutating
    # the spectrum IN PLACE (spectrum_utils 0.3.4, as in the paper). In 0.4.1
    # preprocess returns a NEW normalized object and leaves the original raw, so
    # the stored reference spectra keep raw intensities and the reference dot
    # products blow up to the hundreds/thousands instead of cosine values in
    # [0,1]. Re-preprocess the stored references so their intensities are
    # normalized, matching the training-time features.
    ref_enc = enc.encoders[2]
    normalized_refs = []
    for s in ref_enc.ref_spectra:
        s.is_processed = False
        s.is_valid = True
        normalized_refs.append(gleams_spectrum.preprocess(s, **preprocessing))
    ref_enc.ref_spectra = normalized_refs
    return enc, preprocessing


def _build_embedder_arm():
    """Rebuild the embedder architecture from embedder.py (SELU CNN + dense arms).

    Reconstructed in-process rather than via `keras.models.load_model`, which on
    this TF 2.17 install is pathologically slow deserializing the TF-2.2-era
    HDF5 (minutes) — building the graph directly + `load_weights` is seconds.
    Layer names match the saved model so weights load by name.
    """
    from tensorflow.keras import Input
    from tensorflow.keras.layers import (concatenate, Conv1D, Dense, Flatten,
                                         MaxPooling1D, Reshape)
    from tensorflow.keras.models import Model

    np_, nf_, nr_ = (config.num_precursor_features,
                     config.num_fragment_features, config.num_ref_spectra)
    pi = Input((np_,), name='input_precursor')
    pd32 = Dense(32, activation='selu', kernel_initializer='lecun_normal',
                 name='precursor_dense_32')(pi)
    pd5 = Dense(5, activation='selu', kernel_initializer='lecun_normal',
                name='precursor_dense_5')(pd32)

    fi = Input((nf_,), name='input_fragment')
    x = Reshape((nf_, 1), name='fragment_input_reshape')(fi)
    # (num convs, num filters) per VGG-style block, matching embedder.py.
    for bi, (n_conv, n_filt) in enumerate(
            [(2, 30), (2, 60), (3, 120), (3, 240), (3, 240)], 1):
        for ci in range(1, n_conv + 1):
            x = Conv1D(n_filt, 3, strides=1, activation='selu',
                       name=f'fragment_block_{bi}_conv_{ci}')(x)
        x = MaxPooling1D(1, 2, name=f'fragment_block_{bi}_pool')(x)
    fo = Flatten(name='fragment_flatten')(x)

    ri = Input((nr_,), name='input_ref_spectra')
    rd = Dense(750, activation='selu', kernel_initializer='lecun_normal',
               name='ref_spectra_dense_750')(ri)
    ro = Dense(250, activation='selu', kernel_initializer='lecun_normal',
               name='ref_spectra_output')(rd)

    out = Dense(config.embedding_size, activation='selu',
                kernel_initializer='lecun_normal', activity_regularizer='l2',
                name='output')(concatenate([pd5, fo, ro]))
    return Model([pi, fi, ri], [out], name='embedder')


def _load_embedder(model_path):
    """Rebuild the Siamese wrapper, load weights, return the embedder arm.

    The HDF5 stores the full Siamese model with the embedder weights nested under
    a group named 'embedder', so we rebuild the same two-arm wrapper (tied
    weights) and `load_weights` topologically, then hand back the shared arm.
    """
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K

    embedder = _build_embedder_arm()

    def euclidean_distance(xy):
        x, y = xy
        s = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(s, K.epsilon()))

    np_, nf_, nr_ = (config.num_precursor_features,
                     config.num_fragment_features, config.num_ref_spectra)
    il = [Input((np_,), name='input_precursor_left'),
          Input((nf_,), name='input_fragment_left'),
          Input((nr_,), name='input_ref_spectra_left')]
    ir = [Input((np_,), name='input_precursor_right'),
          Input((nf_,), name='input_fragment_right'),
          Input((nr_,), name='input_ref_spectra_right')]
    dist = Lambda(euclidean_distance, output_shape=(1,),
                  name='embedding_euclidean_distance')([embedder(il),
                                                        embedder(ir)])
    siamese = Model([*il, *ir], dist, name='siamese_model')
    siamese.load_weights(model_path)
    print(f"[cnn] loaded CNN weights from {model_path}; "
          f"embedder params={embedder.count_params():,}")
    return embedder


def _parse_block(block: bytes):
    """(mz, intensity, pepmass, charge) from one MGF BEGIN..END block."""
    mz, it, pepmass, charge = [], [], 0.0, 0
    for raw in block.split(b'\n'):
        line = raw.strip()
        if not line:
            continue
        if line[0:1].isdigit():
            p = line.split()
            if len(p) >= 2:
                mz.append(float(p[0]))
                it.append(float(p[1]))
        elif line[:8] == b'PEPMASS=':
            tok = line[8:].split()
            if tok:
                pepmass = float(tok[0])
        elif line[:7] == b'CHARGE=':
            v = line[7:].strip().rstrip(b'+')
            try:
                charge = int(v)
            except ValueError:
                charge = 0
    return (np.asarray(mz, dtype=np.float32), np.asarray(it, dtype=np.float32),
            pepmass, charge)


def _encode_rows(scans, enc, preprocessing, offsets, n_spectra, file_size):
    """Encode each unique referenced spectrum. Returns (encodings, valid_mask).

    `encodings` is a list aligned to `scans`; invalid entries are None.
    """
    encodings = [None] * len(scans)
    valid = np.zeros(len(scans), dtype=bool)
    order = np.argsort(scans)  # ascending file offset keeps seeks moving forward
    with open(MGF_PATH, 'rb') as f:
        for done, i in enumerate(order, 1):
            scan = int(scans[i])
            if not (0 <= scan < n_spectra):
                continue
            start = int(offsets[scan])
            end = int(offsets[scan + 1]) if scan + 1 < n_spectra else file_size
            f.seek(start)
            mz, it, pepmass, charge = _parse_block(f.read(end - start))
            if len(mz) == 0 or charge == 0:
                continue
            spec = MsmsSpectrum(str(scan), float(pepmass), int(charge), mz, it)
            # gleams' preprocess() reads/writes these flags (normally set by
            # ms_io when reading peak files); initialise them for spectra we
            # build directly from MGF bytes.
            spec.is_processed = False
            spec.is_valid = True
            spec = gleams_spectrum.preprocess(spec, **preprocessing)
            if not spec.is_valid:
                continue
            encodings[i] = enc.encode(spec)
            valid[i] = True
            if done % 2000 == 0:
                print(f"\r[cnn] encoded {done:,}/{len(scans):,} spectra "
                      f"({valid.sum():,} valid)", end='', flush=True)
    print(f"\r[cnn] encoded {len(scans):,}/{len(scans):,} spectra "
          f"({valid.sum():,} valid)          ")
    return encodings, valid


def _embed(encodings, valid, embedder):
    """Run valid encodings through the CNN embedder; NaN rows for invalid."""
    feat_split = (config.num_precursor_features,
                  config.num_precursor_features + config.num_fragment_features)
    idx_valid = np.flatnonzero(valid)
    embeddings = None
    for start in range(0, len(idx_valid), ENCODE_BATCH):
        rows = idx_valid[start:start + ENCODE_BATCH]
        x = ss.vstack([encodings[r] for r in rows], 'csr').toarray()
        inputs = [x[:, :feat_split[0]],
                  x[:, feat_split[0]:feat_split[1]],
                  x[:, feat_split[1]:]]
        emb = embedder.predict(inputs, verbose=0)
        if embeddings is None:
            embeddings = np.full((len(valid), emb.shape[1]), np.nan, np.float64)
        embeddings[rows] = emb
        print(f"\r[cnn] embedded {min(start + ENCODE_BATCH, len(idx_valid)):,}"
              f"/{len(idx_valid):,}", end='', flush=True)
    print()
    if embeddings is None:
        embeddings = np.full((len(valid), 1), np.nan)
    return embeddings


def _pair_distances(embeddings, valid, pairs, remap):
    a = remap[pairs[:, 0]]
    b = remap[pairs[:, 1]]
    ok = valid[a] & valid[b]
    dist = np.full(len(pairs), np.nan, dtype=np.float64)
    dist[ok] = np.linalg.norm(embeddings[a[ok]] - embeddings[b[ok]], axis=1)
    return dist


def main():
    parser = argparse.ArgumentParser(description='CNN holdout evaluator.')
    parser.add_argument('--run-dir', default=None,
                        help='Comparison run folder (default: the latest one, '
                             'so it joins the transformer run in progress).')
    parser.add_argument('--model', default=DEFAULT_CNN_MODEL,
                        help='CNN embedder .hdf5 weights.')
    args = parser.parse_args()

    # Default to the latest run folder so `compare_cnn` drops its distances next
    # to a transformer run; pass --run-dir explicitly to target a specific one.
    run_dir = compare_common.resolve_run_dir(args.run_dir, make_new=False)
    out_npz = os.path.join(run_dir, 'cnn_distances.npz')
    print(f"[cnn] run dir: {run_dir}")

    pos_sel, neg_sel = compare_common.select_pairs(
        VAL_POS_PAIRS_FILE, VAL_NEG_PAIRS_FILE)
    print(f"[cnn] selected {len(pos_sel):,} pos, {len(neg_sel):,} neg pairs")

    val_meta = pd.read_parquet(VAL_METADATA_FILE)
    used = np.unique(np.concatenate([pos_sel.ravel(), neg_sel.ravel()]))
    remap = np.full(len(val_meta), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    scans = val_meta['scan'].to_numpy()[used]
    print(f"[cnn] {len(used):,} unique referenced spectra")

    offsets = np.load(OFFSET_INDEX)
    file_size = os.path.getsize(MGF_PATH)

    enc, preprocessing = _build_encoder()
    embedder = _load_embedder(args.model)

    encodings, valid = _encode_rows(
        scans, enc, preprocessing, offsets, len(offsets), file_size)
    embeddings = _embed(encodings, valid, embedder)

    pos_dist = _pair_distances(embeddings, valid, pos_sel, remap)
    neg_dist = _pair_distances(embeddings, valid, neg_sel, remap)
    meta = {'label': 'CNN (original)', 'source': os.path.basename(args.model)}
    compare_common.save_distances(
        out_npz, pos_dist, neg_dist,
        compare_common.DEFAULT_CAP, compare_common.DEFAULT_SEED, meta=meta)


if __name__ == '__main__':
    main()
