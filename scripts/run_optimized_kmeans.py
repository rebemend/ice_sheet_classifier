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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.assemble_dataset import load_processed_dataset

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
    
    print(f"✓ Summary saved to: {summary_file}")
    
    print("\n🎉 Optimized clustering completed successfully!")

if __name__ == '__main__':
    main()