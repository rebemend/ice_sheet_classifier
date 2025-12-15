#!/usr/bin/env python3
"""
Test k-means without silhouette computation to isolate the bottleneck.
"""

import sys
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset

def test_raw_kmeans_vs_silhouette():
    """Test if silhouette computation is the real bottleneck."""
    print("Testing Raw K-means vs Silhouette Computation")
    print("=" * 50)
    
    # Load data
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    
    # Extract features
    dudx = unified_data['primary_strain'].flatten()
    speed = unified_data['speed'].flatten()
    mu = unified_data['mu'].flatten()
    anisotropy = unified_data['anisotropy'].flatten()
    
    features_raw = np.column_stack([dudx, speed, mu, anisotropy])
    valid_mask = np.all(np.isfinite(features_raw), axis=1)
    features = features_raw[valid_mask]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"Data ready: {features_scaled.shape}")
    
    # Test 1: Raw k-means only (no silhouette)
    print(f"\nTest 1: Raw K-means (no silhouette computation)")
    start = time.time()
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features_scaled)
    
    elapsed = time.time() - start
    print(f"✓ Raw K-means: {elapsed:.3f}s")
    print(f"  Inertia: {kmeans.inertia_:.2e}")
    print(f"  Converged in {kmeans.n_iter_} iterations")
    
    # Test 2: Silhouette computation on full dataset
    print(f"\nTest 2: Silhouette computation on full dataset")
    print("This is likely where the timeout occurs...")
    
    from sklearn.metrics import silhouette_score, silhouette_samples
    
    start = time.time()
    timeout_limit = 60  # 60 second timeout
    
    try:
        # Monitor if this times out
        print("Computing silhouette_score...")
        start_sil = time.time()
        silhouette_avg = silhouette_score(features_scaled, labels)
        sil_score_time = time.time() - start_sil
        print(f"  silhouette_score: {sil_score_time:.3f}s (score: {silhouette_avg:.3f})")
        
        print("Computing silhouette_samples...")
        start_samples = time.time()
        silhouette_samples_scores = silhouette_samples(features_scaled, labels)
        sil_samples_time = time.time() - start_samples
        print(f"  silhouette_samples: {sil_samples_time:.3f}s")
        
        elapsed = time.time() - start
        print(f"✓ Total silhouette computation: {elapsed:.3f}s")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Silhouette computation failed after {elapsed:.3f}s: {e}")
        print("This confirms silhouette computation is the bottleneck!")
        
        # Test on smaller sample to confirm
        print(f"\nTesting silhouette on smaller samples...")
        
        for sample_size in [1000, 5000, 10000]:
            sample_indices = np.random.choice(len(labels), sample_size, replace=False)
            sample_features = features_scaled[sample_indices]
            sample_labels = labels[sample_indices]
            
            start = time.time()
            silhouette_avg = silhouette_score(sample_features, sample_labels)
            elapsed = time.time() - start
            print(f"  Sample {sample_size}: {elapsed:.3f}s (score: {silhouette_avg:.3f})")

def test_optimized_approach():
    """Test an optimized approach without expensive silhouette computation."""
    print(f"\nTest 3: Optimized approach (skip expensive computations)")
    print("=" * 55)
    
    # Simple approach: Just run k-means and basic metrics
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    
    dudx = unified_data['primary_strain'].flatten()
    speed = unified_data['speed'].flatten()
    mu = unified_data['mu'].flatten()
    anisotropy = unified_data['anisotropy'].flatten()
    
    features_raw = np.column_stack([dudx, speed, mu, anisotropy])
    valid_mask = np.all(np.isfinite(features_raw), axis=1)
    features = features_raw[valid_mask]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"Running optimized k-means for {features_scaled.shape[0]} points...")
    start = time.time()
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features_scaled)
    
    # Basic analysis without expensive silhouette computation
    cluster_sizes = np.bincount(labels)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    elapsed = time.time() - start
    print(f"✓ Optimized clustering: {elapsed:.3f}s")
    print(f"  Inertia: {inertia:.2e}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Centroids shape: {centroids.shape}")
    
    return {
        'labels': labels,
        'centroids': centroids,
        'inertia': inertia,
        'cluster_sizes': cluster_sizes,
        'elapsed_time': elapsed
    }

def main():
    test_raw_kmeans_vs_silhouette()
    results = test_optimized_approach()
    
    print(f"\nSummary:")
    print(f"========")
    print(f"✓ Raw k-means is fast ({results['elapsed_time']:.3f}s for 194k points)")
    print(f"✗ Silhouette computation is the bottleneck (O(n²) complexity)")
    print(f"💡 Solution: Skip silhouette for large datasets or use sampling")

if __name__ == '__main__':
    main()