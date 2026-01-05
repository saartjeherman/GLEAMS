"""Command-line interface for training configuration."""

import argparse
from pathlib import Path
from . import config


def parse_args():
    """
    Parse command-line arguments for training configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Train GLEAMS spectrum encoder with contrastive learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    data_group = parser.add_argument_group('Data paths')
    data_group.add_argument('--data-dir', type=str, default=str(config.DATA_DIR),
                           help='Base directory for data files')
    data_group.add_argument('--train-metadata', type=str, default=None,
                           help='Path to train metadata parquet file')
    data_group.add_argument('--test-metadata', type=str, default=None,
                           help='Path to test metadata parquet file')
    data_group.add_argument('--train-pairs-pos', type=str, default=None,
                           help='Path to train positive pairs .npy file')
    data_group.add_argument('--train-pairs-neg', type=str, default=None,
                           help='Path to train negative pairs .npy file')
    data_group.add_argument('--test-pairs-pos', type=str, default=None,
                           help='Path to test positive pairs .npy file')
    data_group.add_argument('--test-pairs-neg', type=str, default=None,
                           help='Path to test negative pairs .npy file')
    data_group.add_argument('--mgf-file', type=str, default=None,
                           help='Path to MGF spectrum file')
    
    # Output paths
    output_group = parser.add_argument_group('Output paths')
    output_group.add_argument('--log-csv', type=str, default=None,
                             help='Path to loss log CSV file')
    output_group.add_argument('--log-txt', type=str, default=None,
                             help='Path to training log text file')
    output_group.add_argument('--best-model-path', type=str, default=str(config.MODEL_DIR / config.BEST_MODEL_FILE),
                             help='Path to save best model checkpoint')
    output_group.add_argument('--final-model-path', type=str, default=str(config.MODEL_DIR / config.FINAL_MODEL_FILE),
                             help='Path to save final model')
    
    # Model architecture
    model_group = parser.add_argument_group('Model architecture')
    model_group.add_argument('--dim-model', type=int, default=config.DIM_MODEL,
                            help='Model dimension')
    model_group.add_argument('--n-head', type=int, default=config.N_HEAD,
                            help='Number of attention heads')
    model_group.add_argument('--dim-feedforward', type=int, default=config.DIM_FEEDFORWARD,
                            help='Feedforward dimension')
    model_group.add_argument('--n-layers', type=int, default=config.N_LAYERS,
                            help='Number of transformer layers')
    model_group.add_argument('--dropout', type=float, default=config.DROPOUT,
                            help='Dropout rate')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training hyperparameters')
    train_group.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                            help='Batch size for training')
    train_group.add_argument('--n-epochs', type=int, default=config.N_EPOCHS,
                            help='Number of training epochs')
    train_group.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY,
                            help='Weight decay (L2 regularization)')
    train_group.add_argument('--margin', type=float, default=config.CONTRASTIVE_MARGIN,
                            help='Contrastive loss margin')
    train_group.add_argument('--scheduler-patience', type=int, default=config.SCHEDULER_PATIENCE,
                            help='Patience for learning rate scheduler')
    
    # Dataset limits (for debugging)
    debug_group = parser.add_argument_group('Debug options')
    debug_group.add_argument('--max-train-pairs', type=int, default=config.MAX_TRAIN_PAIRS,
                            help='Limit training pairs (for debugging)')
    debug_group.add_argument('--max-val-pairs', type=int, default=config.MAX_VAL_PAIRS,
                            help='Limit validation pairs (for debugging)')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    data_dir = Path(args.data_dir)
    if args.train_metadata is None:
        args.train_metadata = str(data_dir / config.TRAIN_METADATA_FILE)
    if args.test_metadata is None:
        args.test_metadata = str(data_dir / config.TEST_METADATA_FILE)
    if args.train_pairs_pos is None:
        args.train_pairs_pos = str(data_dir / config.TRAIN_PAIRS_POS_FILE)
    if args.train_pairs_neg is None:
        args.train_pairs_neg = str(data_dir / config.TRAIN_PAIRS_NEG_FILE)
    if args.test_pairs_pos is None:
        args.test_pairs_pos = str(data_dir / config.TEST_PAIRS_POS_FILE)
    if args.test_pairs_neg is None:
        args.test_pairs_neg = str(data_dir / config.TEST_PAIRS_NEG_FILE)
    if args.mgf_file is None:
        args.mgf_file = str(data_dir / config.MGF_FILE)
    if args.log_csv is None:
        args.log_csv = str(data_dir / config.LOG_CSV_FILE)
    if args.log_txt is None:
        args.log_txt = str(data_dir / config.LOG_TXT_FILE)
    
    return args
