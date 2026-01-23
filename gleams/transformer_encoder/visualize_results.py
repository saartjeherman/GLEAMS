#!/usr/bin/env python
"""Script to visualize transformer model embedding results."""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from .visualization import visualize_model_embeddings, compute_embedding_distances, plot_embedding_distance_distribution, print_statistics
from .dataloader import create_dataloader
from .dataset import SpectraDataset
from . import config


def main():
    parser = argparse.ArgumentParser(
        description='Visualize transformer model embedding distances'
    )
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    
    # Data files
    parser.add_argument('--test_metadata', type=str,
                       default='GLEAMS/data/test_metadata.parquet',
                       help='Path to test metadata file')
    parser.add_argument('--test_pairs_pos', type=str,
                       default='GLEAMS/data/test_metadata_pairs_pos_2.npy',
                       help='Path to test positive pairs')
    parser.add_argument('--test_pairs_neg', type=str,
                       default='GLEAMS/data/test_metadata_pairs_neg_2.npy',
                       help='Path to test negative pairs')
    parser.add_argument('--mgf_file', type=str,
                       default='GLEAMS/data/massivekb_82c0124b.mgf',
                       help='Path to MGF file with spectra')
    
    # Sampling
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Maximum number of pairs to process (default: all)')
    parser.add_argument('--sample_pairs', type=int, default=10000,
                       help='Number of pairs to randomly sample for visualization (default: 10000)')
    
    # Analysis parameters
    parser.add_argument('--fdr', type=float, default=0.01,
                       help='False discovery rate for threshold (default: 0.01 = 1%%)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for computing embeddings (default: 128)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='GLEAMS/results',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load metadata
    print(f"\nLoading test metadata from {args.test_metadata}...")
    test_metadata = pd.read_parquet(args.test_metadata)
    print(f"  Loaded {len(test_metadata):,} spectra")
    
    # Load pairs
    print(f"\nLoading test pairs...")
    test_pos_pairs = np.load(args.test_pairs_pos).astype(int)
    test_neg_pairs = np.load(args.test_pairs_neg).astype(int)
    print(f"  Positive pairs: {len(test_pos_pairs):,}")
    print(f"  Negative pairs: {len(test_neg_pairs):,}")
    
    # Sample pairs if requested
    if args.sample_pairs and args.sample_pairs < len(test_pos_pairs) + len(test_neg_pairs):
        print(f"\nSampling {args.sample_pairs:,} random pairs for visualization...")
        n_pos_sample = min(args.sample_pairs // 2, len(test_pos_pairs))
        n_neg_sample = min(args.sample_pairs // 2, len(test_neg_pairs))
        
        pos_indices = np.random.choice(len(test_pos_pairs), n_pos_sample, replace=False)
        neg_indices = np.random.choice(len(test_neg_pairs), n_neg_sample, replace=False)
        
        test_pos_pairs = test_pos_pairs[pos_indices]
        test_neg_pairs = test_neg_pairs[neg_indices]
        
        print(f"  Sampled positive pairs: {len(test_pos_pairs):,}")
        print(f"  Sampled negative pairs: {len(test_neg_pairs):,}")
    
    # Create dataset and dataloader
    print(f"\nCreating dataset...")
    test_dataset = SpectraDataset(
        metadata=test_metadata,
        mgf_path=args.mgf_file
    )
    
    test_dataloader = create_dataloader(
        test_dataset,
        test_pos_pairs,
        test_neg_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Run visualization pipeline
    print("\n" + "="*60)
    print("STARTING VISUALIZATION PIPELINE")
    print("="*60)
    
    results = visualize_model_embeddings(
        model_path=args.model,
        dataloader=test_dataloader,
        device=device,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        fdr=args.fdr
    )
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  AUC Score: {results['auc']:.4f}")
    print(f"  FDR Threshold ({args.fdr*100:.0f}%): {results['threshold']:.4f}")
    print(f"  Positive pairs analyzed: {len(results['positive_distances']):,}")
    print(f"  Negative pairs analyzed: {len(results['negative_distances']):,}")
    print("="*60)
    

if __name__ == '__main__':
    main()
