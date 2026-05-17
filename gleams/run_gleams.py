"""Legacy GLEAMS-pipeline helpers (embed + regenerate pairs).

The transformer-encoder training pipeline lives in `transformer_encoder/`.
Run it with:
    cd transformer_encoder && python train.py
"""

import logging
import os
import tempfile
from typing import List

import pandas as pd

from feature import feature, encoder  # noqa: F401  (kept for downstream notebooks)
from GLEAMS.gleams import nn_torch as nn
from metadata import metadata
import config


precursor_encoding = {
    'num_bits_mz': config.num_bits_precursor_mz,
    'mz_min': config.precursor_mz_min,
    'mz_max': config.precursor_mz_max,
    'num_bits_mass': config.num_bits_precursor_mass,
    'mass_min': config.precursor_mass_min,
    'mass_max': config.precursor_mass_max,
    'charge_max': config.precursor_charge_max,
}

fragment_encoding = {
    'min_mz': config.fragment_mz_min,
    'max_mz': config.fragment_mz_max,
    'bin_size': config.bin_size,
}

reference_encoding = {
    'filename': config.ref_spectra_filename,
    'preprocessing': {
        'mz_min': config.fragment_mz_min,
        'mz_max': config.fragment_mz_max,
        'min_peaks': config.min_peaks,
        'min_mz_range': config.min_mz_range,
        'remove_precursor_tolerance': config.remove_precursor_tolerance,
        'min_intensity': config.min_intensity,
        'max_peaks_used': config.max_peaks_used,
        'scaling': config.scaling,
    },
    'fragment_mz_tol': config.fragment_mz_tol,
    'num_ref_spectra': config.num_ref_spectra,
}


def run_gleams_embed(peak_files: List[str], embed_name: str = 'GLEAMS_embed') -> None:
    """Run the GLEAMS 'embed' command programmatically against the legacy CNN model."""
    logger = logging.getLogger('gleams')
    logger.info('Starting GLEAMS embedding...')

    if not peak_files:
        raise ValueError('No input peak files specified.')

    temp_dir = tempfile.mkdtemp()
    metadata_filename = os.path.join(temp_dir, f'{embed_name}.parquet')
    embed_dir = os.path.join(temp_dir, 'embed')
    os.mkdir(embed_dir)

    df = pd.DataFrame({'filename': peak_files})
    df['dataset'] = 'GLEAMS'
    df.to_parquet(metadata_filename, index=False)

    embedder_config = {
        'num_precursor_features': config.num_precursor_features,
        'num_fragment_features': config.num_fragment_features,
        'num_ref_spectra_features': config.num_ref_spectra,
        'lr': config.lr,
    }

    model_filename = '/home/saartje/Desktop/Research/GLEAMS/data/gleams.pth'
    nn.embed(
        metadata_filename,
        model_filename,
        f'{embed_name}.npy',
        embed_dir,
        precursor_encoding,
        fragment_encoding,
        reference_encoding,
        embedder_config,
        config.batch_size,
        config.charges,
    )

    logger.info('GLEAMS embedding completed successfully.')


def run_preprocessing():
    """Regenerate positive and negative pairs from the metadata files."""
    train_file = '/home/saartje/Desktop/Research/GLEAMS/data/train_metadata.parquet'
    test_file = '/home/saartje/Desktop/Research/GLEAMS/data/test_metadata.parquet'

    print("Regenerating pairs for train metadata...")
    metadata.generate_pairs_positive(train_file, config.charges)
    metadata.generate_pairs_negative(
        train_file, config.charges, config.pair_mz_tolerance,
        config.negative_pair_fragment_tolerance,
        config.negative_pair_matching_fragments_threshold,
    )

    print("\nRegenerating pairs for test metadata...")
    metadata.generate_pairs_positive(test_file, config.charges)
    metadata.generate_pairs_negative(
        test_file, config.charges, config.pair_mz_tolerance,
        config.negative_pair_fragment_tolerance,
        config.negative_pair_matching_fragments_threshold,
    )

    print("\n✓ Pair regeneration complete!")
