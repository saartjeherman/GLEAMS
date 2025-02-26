import os
import logging
import tempfile
from typing import List
import pandas as pd

from nn import nn_torch as nn
from metadata import metadata
from feature import feature
#from nn import nn
import config


precursor_encoding = {
    'num_bits_mz': config.num_bits_precursor_mz,
    'mz_min': config.precursor_mz_min,
    'mz_max': config.precursor_mz_max,
    'num_bits_mass': config.num_bits_precursor_mass,
    'mass_min': config.precursor_mass_min,
    'mass_max': config.precursor_mass_max,
    'charge_max': config.precursor_charge_max
}

fragment_encoding = {
    'min_mz': config.fragment_mz_min,
    'max_mz': config.fragment_mz_max,
    'bin_size': config.bin_size
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
        'scaling': config.scaling
    },
    'fragment_mz_tol': config.fragment_mz_tol,
    'num_ref_spectra': config.num_ref_spectra
}

def run_gleams_embed(peak_files: List[str], embed_name: str = 'GLEAMS_embed') -> None:
    """
    Run the 'gleams embed' command programmatically.

    Parameters:
    - peak_files: List of input peak files (e.g., mzML, MGF).
    - embed_name: Name for the output files.
    """
    logger = logging.getLogger('gleams')
    logger.info('Starting GLEAMS embedding...')

    if not peak_files:
        raise ValueError('No input peak files specified.')

    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    metadata_filename = os.path.join(temp_dir, f'{embed_name}.parquet')
    embed_dir = os.path.join(temp_dir, 'embed')
    os.mkdir(embed_dir)

    # Create a metadata file with the file names
    metadata = pd.DataFrame({'filename': peak_files})
    metadata['dataset'] = 'GLEAMS'
    metadata.to_parquet(metadata_filename, index=False)

    # Prepare embedding configuration
    

    embedder_config = {
        'num_precursor_features': config.num_precursor_features,
        'num_fragment_features': config.num_fragment_features,
        'num_ref_spectra_features': config.num_ref_spectra,
        'lr': config.lr
    }

    # Run the embedding
    model_filename = '/home/saartje/Desktop/Research/GLEAMS/data/gleams.pth' #config.model_filename  
    nn.embed(
        metadata_filename,
        model_filename,
        #config.model_filename,
        f'{embed_name}.npy',
        embed_dir,
        precursor_encoding,
        fragment_encoding,
        reference_encoding,
        embedder_config,
        config.batch_size,
        config.charges
    )

    logger.info('GLEAMS embedding completed successfully.')

#feat_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'feature')
#op_kwargs={'filename_model': config.model_filename,
#                   'filename_feat_train':
#                       os.path.join(feat_dir,
#                                    f'feature_{config.massivekb_task_id}_'
#                                    f'train.npy'),
#                   'filename_train_pairs_pos':
#                       os.path.join(feat_dir,
#                                    f'feature_{config.massivekb_task_id}_'
#                                    f'train_pairs_pos.npy'),
#                   'filename_train_pairs_neg':
#                       os.path.join(feat_dir,
#                                    f'feature_{config.massivekb_task_id}_'
#                                    f'train_pairs_neg.npy'),
#                   'filename_feat_val':
#                       os.path.join(feat_dir,
#                                    f'feature_{config.massivekb_task_id}_'
#                                    f'val.npy'),
#                   'filename_val_pairs_pos':
#                       os.path.join(feat_dir,
#                                    f'feature_{config.massivekb_task_id}_'
#                                    f'val_pairs_pos.npy'),
#                  'filename_val_pairs_neg':
#                       os.path.join(feat_dir,
#                                    f'feature_{config.massivekb_task_id}_'
#                                    f'val_pairs_neg.npy')}
#def train_model(op_kwargs):
#    nn.train_nn(**op_kwargs, 
#                embedder_config=config.embedder_config,
#                batch_size=config.batch_size,
#                num_epochs=config.num_epochs,
#               steps_per_epoch=config.steps_per_epoch,
#                max_num_pairs_train=config.max_num_pairs_train,
#                max_num_pairs_val=config.max_num_pairs_val)
    
def run_preprocessing():
    filename_data = '/home/saartje/Desktop/Research/GLEAMS/data/metadata_spectra.tsv' 
    reduced_filename_data = '/home/saartje/Desktop/Research/GLEAMS/data/metadata_spectra_reduced.tsv'
    output_file = '/home/saartje/Desktop/Research/GLEAMS/data/metadata_spectra.parquet'

    #reduce the file to only the first 1 million rows
    metadata_data = pd.read_csv(filename_data, sep='\t', nrows=1000000)
    metadata_data.to_csv(reduced_filename_data, index=False, sep='\t')

    metadata.convert_massivekb_metadata(reduced_filename_data, output_file)
    #metadata.generate_pairs_positive(output_file, config.charges)
    #metadata.generate_pairs_negative(output_file, config.charges, config.pair_mz_tolerance, 
                                     #config.negative_pair_fragment_tolerance, config.negative_pair_matching_fragments_threshold)
    
    
    peaks_dir = '/home/saartje/Desktop/Research/GLEAMS/data/peaks'
    metadata.download_massivekb_peaks(reduced_filename_data, peaks_dir)

    output_file_feature = '/home/saartje/Desktop/Research/GLEAMS/data/feature_output.npz'
    #feature.convert_peaks_to_features(output_file, output_file_feature, precursor_encoding, 
                                        #fragment_encoding, reference_encoding, filter_scans=True)

    #read the parquet file
    #metadata_data = pd.read_parquet(output_file)
    #print(metadata_data.head())
    #print(metadata_data.columns)

    
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    peak_files = ["/home/saartje/Desktop/Research/GLEAMS/gleams_reference_spectra_modified.mgf"]  # Replace with your actual file path(s)
    #run_gleams_embed(peak_files, embed_name="GLEAMS_embed")
    #train_model(op_kwargs)
    run_preprocessing()




