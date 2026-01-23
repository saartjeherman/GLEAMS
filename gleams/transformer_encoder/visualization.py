"""Visualization functions for analyzing transformer model embeddings."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys


def compute_embedding_distances(
    encoder,
    dataloader: DataLoader,
    device: torch.device,
    max_pairs: int = None
) -> tuple:
    """
    Compute embedding distances for all pairs in a dataset.
    
    Args:
        encoder: Trained spectrum encoder model
        dataloader: DataLoader containing spectrum pairs
        device: Device to run computation on
        max_pairs: Maximum number of pairs to process (None = all)
        
    Returns:
        Tuple of (positive_distances, negative_distances) as numpy arrays
    """
    encoder.eval()
    
    positive_distances = []
    negative_distances = []
    n_pairs_processed = 0
    
    show_progress = hasattr(sys.__stdout__, 'isatty') and sys.__stdout__.isatty()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Computing embeddings", unit="batch",
                   disable=not show_progress, mininterval=0.5, dynamic_ncols=True, 
                   file=sys.__stdout__)
        
        for batch in pbar:
            if max_pairs and n_pairs_processed >= max_pairs:
                break
                
            (mz1, int1, pepmass1, charge1), (mz2, int2, pepmass2, charge2), labels = batch
            
            # Move to device
            mz1 = mz1.to(device)
            int1 = int1.to(device)
            pepmass1 = pepmass1.to(device)
            charge1 = charge1.to(device)
            
            mz2 = mz2.to(device)
            int2 = int2.to(device)
            pepmass2 = pepmass2.to(device)
            charge2 = charge2.to(device)
            
            labels = labels.to(device)
            
            # Get embeddings
            emb1_full, _ = encoder(mz1, int1, pepmass=pepmass1, charge=charge1)
            emb2_full, _ = encoder(mz2, int2, pepmass=pepmass2, charge=charge2)
            
            # Use global token (CLS token)
            emb1 = emb1_full[:, 0, :]
            emb2 = emb2_full[:, 0, :]
            
            # Compute Euclidean distances (no normalization, matching training)
            distances = torch.nn.functional.pairwise_distance(emb1, emb2)
            distances_np = distances.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Separate by label
            pos_mask = labels_np == 1
            neg_mask = labels_np == 0
            
            positive_distances.extend(distances_np[pos_mask].tolist())
            negative_distances.extend(distances_np[neg_mask].tolist())
            
            n_pairs_processed += len(labels)
            pbar.set_postfix({
                'pos': len(positive_distances),
                'neg': len(negative_distances)
            })
    
    return np.array(positive_distances), np.array(negative_distances)


def compute_fdr_threshold(positive_distances, negative_distances, fdr=0.01):
    """
    Compute distance threshold for a given false discovery rate.
    
    Args:
        positive_distances: Array of distances for positive (similar) pairs
        negative_distances: Array of distances for negative (dissimilar) pairs
        fdr: Desired false discovery rate (default: 0.01 for 1%)
        
    Returns:
        Distance threshold value
    """
    # Sort all distances
    all_distances = np.concatenate([positive_distances, negative_distances])
    all_labels = np.concatenate([
        np.ones(len(positive_distances)),
        np.zeros(len(negative_distances))
    ])
    
    # Sort by distance
    sorted_indices = np.argsort(all_distances)
    sorted_distances = all_distances[sorted_indices]
    sorted_labels = all_labels[sorted_indices]
    
    # Compute cumulative FDR at each threshold
    # FDR = FP / (TP + FP) where we call pairs "positive" if distance < threshold
    cumsum_tp = np.cumsum(sorted_labels)  # True positives (positive pairs below threshold)
    cumsum_fp = np.cumsum(1 - sorted_labels)  # False positives (negative pairs below threshold)
    
    # Avoid division by zero
    total_predictions = cumsum_tp + cumsum_fp
    fdr_at_threshold = np.divide(cumsum_fp, total_predictions, 
                                  out=np.zeros_like(cumsum_fp, dtype=float),
                                  where=total_predictions != 0)
    
    # Find threshold where FDR <= desired FDR
    valid_indices = np.where(fdr_at_threshold <= fdr)[0]
    
    if len(valid_indices) == 0:
        print(f"Warning: No threshold found for FDR={fdr}. Using median of negative distances.")
        return np.median(negative_distances)
    
    # Take the last (highest distance) that still satisfies FDR constraint
    threshold_idx = valid_indices[-1]
    threshold = sorted_distances[threshold_idx]
    actual_fdr = fdr_at_threshold[threshold_idx]
    
    print(f"\nFDR threshold computation:")
    print(f"  Target FDR: {fdr*100:.1f}%")
    print(f"  Actual FDR: {actual_fdr*100:.2f}%")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Pairs below threshold: {total_predictions[threshold_idx]:,}")
    
    return threshold


def plot_embedding_distance_distribution(
    positive_distances,
    negative_distances,
    fdr=0.01,
    save_path=None,
    title=None,
    figsize=(10, 6)
):
    """
    Plot the distribution of embedding distances for positive and negative pairs,
    similar to Extended Data Fig. 3 from Bittremieux et al.
    
    Args:
        positive_distances: Array of distances for positive (similar) pairs
        negative_distances: Array of distances for negative (dissimilar) pairs
        fdr: False discovery rate for threshold line (default: 0.01 for 1%)
        save_path: Path to save the figure (optional)
        title: Custom title (optional)
        figsize: Figure size tuple
    """
    # Compute FDR threshold
    threshold = compute_fdr_threshold(positive_distances, negative_distances, fdr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot density distributions
    # Use gaussian KDE for smooth distributions
    pos_kde = stats.gaussian_kde(positive_distances)
    neg_kde = stats.gaussian_kde(negative_distances)
    
    # Create x-axis range
    x_min = min(positive_distances.min(), negative_distances.min())
    x_max = max(positive_distances.max(), negative_distances.max())
    x_range = np.linspace(x_min, x_max, 1000)
    
    # Compute densities
    pos_density = pos_kde(x_range)
    neg_density = neg_kde(x_range)
    
    # Plot with filled areas
    ax.fill_between(x_range, 0, pos_density, alpha=0.6, color='#d4a5c4', 
                     label='Positive pairs', linewidth=1.5, edgecolor='#8b4789')
    ax.fill_between(x_range, 0, neg_density, alpha=0.6, color='#a5c9d4',
                     label='Negative pairs', linewidth=1.5, edgecolor='#4789a5')
    
    # Add threshold line
    ax.axvline(threshold, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add text annotation for threshold
    y_max = max(pos_density.max(), neg_density.max())
    ax.text(threshold, y_max * 0.95, f'  {threshold:.4f}', 
            va='top', ha='left', fontsize=10, color='gray')
    
    # Labels and styling
    ax.set_xlabel('Embedded distance', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=14, pad=15)
    else:
        ax.set_title(f'False negative rate between positive and negative embedding pairs\n'
                    f'FDR threshold: {threshold:.4f} (grey line) at {fdr*100:.0f}% false discovery rate',
                    fontsize=12, pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig, ax


def plot_roc_curve(positive_distances, negative_distances, save_path=None):
    """
    Plot ROC curve for the embedding distance classifier.
    
    Args:
        positive_distances: Array of distances for positive pairs
        negative_distances: Array of distances for negative pairs
        save_path: Path to save figure (optional)
    """
    # Prepare data for ROC curve
    # Lower distances should predict positive pairs (label=1)
    # So we negate distances to use as scores
    scores = np.concatenate([-positive_distances, -negative_distances])
    labels = np.concatenate([
        np.ones(len(positive_distances)),
        np.zeros(len(negative_distances))
    ])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='#4789a5', linewidth=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve for Embedding-based Classification', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    return fig, ax, roc_auc


def print_statistics(positive_distances, negative_distances):
    """
    Print comprehensive statistics about the distance distributions.
    
    Args:
        positive_distances: Array of distances for positive pairs
        negative_distances: Array of distances for negative pairs
    """
    print("\n" + "="*60)
    print("EMBEDDING DISTANCE STATISTICS")
    print("="*60)
    
    print("\nPositive Pairs (should be LOW distances):")
    print(f"  Count:   {len(positive_distances):,}")
    print(f"  Mean:    {np.mean(positive_distances):.4f}")
    print(f"  Median:  {np.median(positive_distances):.4f}")
    print(f"  Std:     {np.std(positive_distances):.4f}")
    print(f"  Min:     {np.min(positive_distances):.4f}")
    print(f"  Max:     {np.max(positive_distances):.4f}")
    print(f"  25th:    {np.percentile(positive_distances, 25):.4f}")
    print(f"  75th:    {np.percentile(positive_distances, 75):.4f}")
    
    print("\nNegative Pairs (should be HIGH distances):")
    print(f"  Count:   {len(negative_distances):,}")
    print(f"  Mean:    {np.mean(negative_distances):.4f}")
    print(f"  Median:  {np.median(negative_distances):.4f}")
    print(f"  Std:     {np.std(negative_distances):.4f}")
    print(f"  Min:     {np.min(negative_distances):.4f}")
    print(f"  Max:     {np.max(negative_distances):.4f}")
    print(f"  25th:    {np.percentile(negative_distances, 25):.4f}")
    print(f"  75th:    {np.percentile(negative_distances, 75):.4f}")
    
    # Separation metric
    pos_mean = np.mean(positive_distances)
    neg_mean = np.mean(negative_distances)
    separation = neg_mean - pos_mean
    
    print(f"\nSeparation:")
    print(f"  Gap between means: {separation:.4f}")
    print(f"  Ratio (neg/pos):   {neg_mean/pos_mean:.2f}x")
    
    # Overlap analysis
    pos_max = np.max(positive_distances)
    neg_min = np.min(negative_distances)
    
    if pos_max < neg_min:
        print(f"\n✓ Perfect separation! No overlap between distributions.")
    else:
        overlap_region = [neg_min, pos_max]
        n_pos_in_overlap = np.sum((positive_distances >= neg_min) & (positive_distances <= pos_max))
        n_neg_in_overlap = np.sum((negative_distances >= neg_min) & (negative_distances <= pos_max))
        print(f"\nOverlap region: [{neg_min:.4f}, {pos_max:.4f}]")
        print(f"  Positive pairs in overlap: {n_pos_in_overlap:,} ({n_pos_in_overlap/len(positive_distances)*100:.2f}%)")
        print(f"  Negative pairs in overlap: {n_neg_in_overlap:,} ({n_neg_in_overlap/len(negative_distances)*100:.2f}%)")
    
    print("\n" + "="*60)


def visualize_model_embeddings(
    model_path,
    dataloader,
    device,
    output_dir='.',
    max_pairs=None,
    fdr=0.01
):
    """
    Complete visualization pipeline: load model, compute distances, create plots.
    
    Args:
        model_path: Path to trained model (.pt file)
        dataloader: DataLoader with test pairs
        device: Device to run on
        output_dir: Directory to save outputs
        max_pairs: Maximum pairs to process (None = all)
        fdr: False discovery rate for threshold
    
    Returns:
        Dictionary with distances and statistics
    """
    # Load model
    print(f"Loading model from {model_path}...")
    encoder = torch.load(model_path, map_location=device)
    encoder.eval()
    
    # Compute distances
    print("\nComputing embedding distances...")
    pos_dist, neg_dist = compute_embedding_distances(
        encoder, dataloader, device, max_pairs
    )
    
    # Print statistics
    print_statistics(pos_dist, neg_dist)
    
    # Create plots
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Distance distribution plot
    dist_plot_path = os.path.join(output_dir, 'embedding_distances.png')
    plot_embedding_distance_distribution(
        pos_dist, neg_dist, fdr=fdr, save_path=dist_plot_path
    )
    
    # ROC curve
    roc_plot_path = os.path.join(output_dir, 'roc_curve.png')
    _, _, auc_score = plot_roc_curve(pos_dist, neg_dist, save_path=roc_plot_path)
    
    print(f"\n✓ Visualizations complete! Files saved to {output_dir}/")
    
    return {
        'positive_distances': pos_dist,
        'negative_distances': neg_dist,
        'auc': auc_score,
        'fdr': fdr,
        'threshold': compute_fdr_threshold(pos_dist, neg_dist, fdr)
    }
