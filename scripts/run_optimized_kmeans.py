#!/usr/bin/env python3
"""
Optimized k-means clustering script for large datasets.

This version bypasses expensive silhouette computation for datasets > 50k points.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import json
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.assemble_dataset import load_processed_dataset

def create_clustering_visualizations(features_scaled, labels, centroids, feature_names, output_dir, k=3):
    """
    Create comprehensive visualization plots for clustering results.
    
    Parameters
    ----------
    features_scaled : np.ndarray
        Scaled feature array
    labels : np.ndarray
        Cluster labels
    centroids : np.ndarray
        Cluster centroids
    feature_names : list
        Names of features
    output_dir : Path
        Output directory for plots
    k : int
        Number of clusters
    """
    print("Creating visualization plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl", k)
    
    # Sample points for visualization if dataset is too large
    n_sample = min(10000, len(features_scaled))
    if len(features_scaled) > n_sample:
        sample_idx = np.random.choice(len(features_scaled), n_sample, replace=False)
        features_viz = features_scaled[sample_idx]
        labels_viz = labels[sample_idx]
    else:
        features_viz = features_scaled
        labels_viz = labels
        sample_idx = np.arange(len(features_scaled))
    
    # 1. Feature Pair Plots
    print("  Creating feature pair plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    feature_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    pair_names = [('dudx', 'speed'), ('dudx', 'mu'), ('speed', 'anisotropy'), ('mu', 'anisotropy')]
    
    for idx, ((i, j), (name_i, name_j)) in enumerate(zip(feature_pairs, pair_names)):
        ax = axes[idx//2, idx%2]
        
        for cluster in range(k):
            cluster_mask = labels_viz == cluster
            if np.any(cluster_mask):
                ax.scatter(
                    features_viz[cluster_mask, i],
                    features_viz[cluster_mask, j],
                    alpha=0.6, s=20, label=f'Cluster {cluster}'
                )
        
        ax.set_xlabel(f'{name_i} (scaled)')
        ax.set_ylabel(f'{name_j} (scaled)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{name_i} vs {name_j}')
    
    plt.suptitle('Feature Space: Pairwise Cluster Separation', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_pairs.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. PCA Visualization
    print("  Creating PCA visualization...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_viz)
    
    plt.figure(figsize=(10, 8))
    for cluster in range(k):
        cluster_mask = labels_viz == cluster
        if np.any(cluster_mask):
            plt.scatter(
                features_pca[cluster_mask, 0],
                features_pca[cluster_mask, 1],
                alpha=0.7, s=30, 
                label=f'Cluster {cluster} ({np.sum(cluster_mask)} points)'
            )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Clusters in Principal Component Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "pca_clusters.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Cluster Analysis (Centroids + Sizes)
    print("  Creating cluster analysis plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Centroid values
    x_pos = np.arange(len(feature_names))
    width = 0.25
    
    for cluster in range(k):
        ax1.bar(
            x_pos + cluster * width, 
            centroids[cluster], 
            width, 
            label=f'Cluster {cluster}',
            alpha=0.8
        )
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Centroid Value (scaled)')
    ax1.set_title('Cluster Centroids in Feature Space')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(feature_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cluster sizes
    sizes = [np.sum(labels == i) for i in range(k)]
    colors = plt.cm.Set2(np.arange(k))
    
    ax2.pie(sizes, labels=[f'Cluster {i}\\n({size:,} points)' for i, size in enumerate(sizes)],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Distributions by Cluster
    print("  Creating feature distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx//2, idx%2]
        
        feature_values = features_scaled[:, idx]
        
        for cluster in range(k):
            cluster_mask = labels == cluster
            if np.any(cluster_mask):
                ax.hist(
                    feature_values[cluster_mask], 
                    alpha=0.7, 
                    bins=50, 
                    label=f'Cluster {cluster}',
                    density=True
                )
        
        ax.set_xlabel(f'{feature_name} (scaled)')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions by Cluster', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Silhouette Analysis (if silhouette scores available)
    print("  Creating silhouette analysis...")
    try:
        from sklearn.metrics import silhouette_samples
        
        # Use sample for silhouette analysis if dataset is large
        if len(features_scaled) > 5000:
            silhouette_idx = np.random.choice(len(features_scaled), 5000, replace=False)
            silhouette_features = features_scaled[silhouette_idx]
            silhouette_labels = labels[silhouette_idx]
        else:
            silhouette_features = features_scaled
            silhouette_labels = labels
        
        silhouette_scores = silhouette_samples(silhouette_features, silhouette_labels)
        silhouette_avg = np.mean(silhouette_scores)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 10
        
        for cluster in range(k):
            cluster_silhouette_values = silhouette_scores[silhouette_labels == cluster]
            cluster_silhouette_values.sort()
            
            size_cluster = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster
            
            color = plt.cm.Set2(cluster)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
            y_lower = y_upper + 10
        
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Cluster')
        ax.set_title(f'Silhouette Analysis (Average Score: {silhouette_avg:.3f})')
        
        # Add vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'Average Score: {silhouette_avg:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "silhouette_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Skipping silhouette analysis: {e}")
    
    print("  Visualization plots completed!")
    
    return [
        "feature_pairs.png",
        "pca_clusters.png", 
        "cluster_analysis.png",
        "feature_distributions.png",
        "silhouette_analysis.png"
    ]

def optimized_kmeans_analysis(features_scaled, k=3, max_silhouette_size=50000):
    """
    Run k-means with optimized silhouette computation for large datasets.
    
    Parameters
    ----------
    features_scaled : np.ndarray
        Scaled feature array
    k : int  
        Number of clusters
    max_silhouette_size : int
        Maximum dataset size for silhouette computation
        
    Returns
    -------
    dict
        Clustering results
    """
    print(f"Running optimized k-means with k={k}...")
    
    # Run k-means clustering
    start = time.time()
    kmeans = KMeans(
        n_clusters=k, 
        random_state=42, 
        n_init=10, 
        max_iter=300
    )
    
    labels = kmeans.fit_predict(features_scaled)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    kmeans_time = time.time() - start
    
    print(f"✓ K-means clustering completed in {kmeans_time:.3f}s")
    
    # Compute silhouette score intelligently
    silhouette_avg = np.nan
    silhouette_samples_scores = np.full(len(labels), np.nan)
    
    if len(labels) <= max_silhouette_size:
        print("Computing silhouette score on full dataset...")
        start = time.time()
        from sklearn.metrics import silhouette_score, silhouette_samples
        
        silhouette_avg = silhouette_score(features_scaled, labels)
        silhouette_samples_scores = silhouette_samples(features_scaled, labels)
        silhouette_time = time.time() - start
        print(f"✓ Silhouette computation completed in {silhouette_time:.3f}s")
    else:
        print(f"Dataset too large ({len(labels):,} > {max_silhouette_size:,}) - sampling for silhouette...")
        start = time.time()
        
        # Sample points for silhouette computation
        np.random.seed(42)
        sample_indices = np.random.choice(len(labels), max_silhouette_size, replace=False)
        sample_features = features_scaled[sample_indices]
        sample_labels = labels[sample_indices]
        
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(sample_features, sample_labels)
        silhouette_time = time.time() - start
        print(f"✓ Silhouette computation (sampled) completed in {silhouette_time:.3f}s")
        print(f"  Silhouette score (on {max_silhouette_size:,} samples): {silhouette_avg:.3f}")
    
    # Compute cluster statistics
    cluster_sizes = np.bincount(labels)
    total_samples = len(labels)
    
    # Basic cluster quality metrics
    cluster_balance = {
        'sizes': cluster_sizes.tolist(),
        'proportions': (cluster_sizes / total_samples).tolist(),
        'balance_ratio': np.min(cluster_sizes) / np.max(cluster_sizes) if np.max(cluster_sizes) > 0 else 0
    }
    
    # Centroid separation analysis
    centroid_distances = []
    for i in range(k):
        for j in range(i+1, k):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            centroid_distances.append(dist)
    
    separation_analysis = {
        'min_centroid_distance': float(np.min(centroid_distances)) if centroid_distances else 0,
        'mean_centroid_distance': float(np.mean(centroid_distances)) if centroid_distances else 0,
        'max_centroid_distance': float(np.max(centroid_distances)) if centroid_distances else 0
    }
    
    return {
        'labels': labels,
        'centroids': centroids,
        'inertia': inertia,
        'silhouette_avg': silhouette_avg,
        'silhouette_samples': silhouette_samples_scores,
        'cluster_sizes': cluster_sizes,
        'cluster_balance': cluster_balance,
        'separation': separation_analysis,
        'n_clusters': k,
        'converged': kmeans.n_iter_ < 300,
        'n_iter': kmeans.n_iter_,
        'timing': {
            'kmeans': kmeans_time,
            'silhouette': silhouette_time if 'silhouette_time' in locals() else 0
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Run optimized k-means clustering')
    parser.add_argument('--processed_data', required=True,
                       help='Path to processed dataset (.npz)')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory for results')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of clusters (default: 3)')
    parser.add_argument('--max_silhouette_size', type=int, default=50000,
                       help='Maximum size for silhouette computation')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip visualization plots (faster execution)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Optimized Ice Shelf K-Means Clustering")
    print("=" * 40)
    print(f"Number of clusters (k): {args.k}")
    print(f"Output directory: {output_dir}")
    print(f"Max silhouette size: {args.max_silhouette_size:,}")
    print()
    
    # Load data
    print("Loading processed dataset...")
    start_total = time.time()
    unified_data, feature_data = load_processed_dataset(args.processed_data)
    
    # Extract features efficiently
    print("Extracting primary features...")
    dudx = unified_data['primary_strain'].flatten()
    speed = unified_data['speed'].flatten()
    mu = unified_data['mu'].flatten()
    anisotropy = unified_data['anisotropy'].flatten()
    
    # Create feature array
    features_raw = np.column_stack([dudx, speed, mu, anisotropy])
    valid_mask = np.all(np.isfinite(features_raw), axis=1)
    features = features_raw[valid_mask]
    
    feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
    
    print(f"Dataset summary:")
    print(f"  Grid shape: {unified_data['x'].shape}")
    print(f"  Total points: {len(valid_mask):,}")
    print(f"  Valid points: {len(features):,} ({100*len(features)/len(valid_mask):.1f}%)")
    print()
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Run optimized k-means
    print()
    results = optimized_kmeans_analysis(features_scaled, args.k, args.max_silhouette_size)
    
    total_time = time.time() - start_total
    print()
    print("Clustering Results:")
    print("-" * 20)
    print(f"✓ Total processing time: {total_time:.3f}s")
    print(f"  K-means time: {results['timing']['kmeans']:.3f}s")
    print(f"  Silhouette time: {results['timing']['silhouette']:.3f}s") 
    print(f"Inertia: {results['inertia']:.2e}")
    print(f"Silhouette score: {results['silhouette_avg']:.3f}")
    print(f"Converged in {results['n_iter']} iterations")
    print()
    
    print("Cluster Analysis:")
    print("-" * 17)
    balance = results['cluster_balance']
    for i, (size, prop) in enumerate(zip(balance['sizes'], balance['proportions'])):
        print(f"Cluster {i}: {size:,} points ({prop:.1%})")
    print(f"Balance ratio: {balance['balance_ratio']:.3f}")
    print()
    
    # Save results
    results_file = output_dir / 'optimized_clustering_results.npz'
    
    # Map labels back to full grid
    full_labels = np.full(len(valid_mask), -1, dtype=int)
    full_labels[valid_mask] = results['labels']
    
    # Get spatial coordinates for valid points
    x_flat = unified_data['x'].flatten()
    y_flat = unified_data['y'].flatten()
    coordinates = np.column_stack([x_flat, y_flat])
    valid_coordinates = coordinates[valid_mask]
    
    save_data = {
        'labels': results['labels'],
        'full_labels': full_labels,
        'centroids': results['centroids'],
        'inertia': results['inertia'],
        'silhouette_avg': results['silhouette_avg'],
        'coordinates': valid_coordinates,
        'features_scaled': features_scaled,
        'feature_names': feature_names,
        'k': args.k,
        'grid_shape': unified_data['x'].shape,
        'cluster_sizes': results['cluster_sizes'],
        'timing': results['timing']
    }
    
    np.savez(results_file, **save_data)
    print(f"✓ Results saved to: {results_file}")
    
    # Save summary
    summary_file = output_dir / 'clustering_summary.json'
    summary_data = {
        'parameters': {
            'k': args.k,
            'feature_set': 'primary',
            'features': feature_names,
            'max_silhouette_size': args.max_silhouette_size
        },
        'data_info': {
            'grid_shape': list(unified_data['x'].shape),
            'total_points': int(len(valid_mask)),
            'valid_points': int(len(features))
        },
        'results': {
            'inertia': float(results['inertia']),
            'silhouette_avg': float(results['silhouette_avg']),
            'n_clusters': int(args.k),
            'converged': bool(results['converged']),
            'n_iter': int(results['n_iter'])
        },
        'cluster_analysis': {
            'sizes': [int(x) for x in balance['sizes']],
            'proportions': [float(x) for x in balance['proportions']],
            'balance_ratio': float(balance['balance_ratio'])
        },
        'timing': {
            'total_time': float(total_time),
            'kmeans_time': float(results['timing']['kmeans']),
            'silhouette_time': float(results['timing']['silhouette'])
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    # Create visualizations (unless disabled)
    if not args.no_plots:
        print()
        try:
            plot_files = create_clustering_visualizations(
                features_scaled, results['labels'], results['centroids'], 
                feature_names, output_dir, args.k
            )
            print("Visualization files created:")
            for plot_file in plot_files:
                print(f"  - {plot_file}")
        except Exception as e:
            print(f"Warning: Visualization creation failed: {e}")
            print("  Continuing without plots...")
    else:
        print("\nVisualization plots skipped (--no_plots flag used)")
    
    print("\nOptimized clustering completed successfully!")

if __name__ == '__main__':
    main()