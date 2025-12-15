#!/usr/bin/env python3
"""
Test the fixed k-means approach - only run k=3, not k=[2,3,4,5,6,7,8].
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset
from clustering.kmeans_runner import run_kmeans_analysis

def main():
    print("Testing Fixed K-Means (Only k=3)")
    print("=" * 35)
    
    # Load and prepare data
    print("Loading data...")
    start = time.time()
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    
    # Extract and scale features
    dudx = unified_data['primary_strain'].flatten()
    speed = unified_data['speed'].flatten()
    mu = unified_data['mu'].flatten()
    anisotropy = unified_data['anisotropy'].flatten()
    
    features_raw = np.column_stack([dudx, speed, mu, anisotropy])
    valid_mask = np.all(np.isfinite(features_raw), axis=1)
    features = features_raw[valid_mask]
    
    # Scale features properly
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    prep_time = time.time() - start
    print(f"Data preparation: {prep_time:.3f}s")
    print(f"Dataset shape: {features_scaled.shape}")
    
    # Test 1: Run ONLY k=3 (this should be fast)
    feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
    
    print(f"\nTest 1: Running k-means with k=3 ONLY...")
    start = time.time()
    
    # IMPORTANT: Pass k_range=[3] to only run k=3
    results = run_kmeans_analysis(
        features_scaled, feature_names, 
        k_range=[3],  # Only test k=3, not [2,3,4,5,6,7,8]
        method='combined'
    )
    
    elapsed = time.time() - start
    print(f"✓ K-means (k=3 only) completed in {elapsed:.3f}s")
    
    # Display results
    main_result = results['optimal_result']
    print(f"  Silhouette score: {main_result['silhouette_avg']:.3f}")
    print(f"  Inertia: {main_result['inertia']:.2e}")
    print(f"  Cluster sizes: {main_result['cluster_sizes']}")
    
    # Test 2: Compare with the problematic approach (k_range=None)
    print(f"\nTest 2: Running k-means with DEFAULT k_range (should be slow)...")
    print("This will test k=[2,3,4,5,6,7,8] and likely timeout due to silhouette computation")
    
    # Test on a smaller sample first to confirm the issue
    sample_size = 10000
    sample_indices = np.random.choice(features_scaled.shape[0], sample_size, replace=False)
    sample_features = features_scaled[sample_indices]
    
    print(f"Testing on {sample_size} sample first...")
    start = time.time()
    
    try:
        results_sample = run_kmeans_analysis(
            sample_features, feature_names,
            k_range=None,  # This will use [2,3,4,5,6,7,8]
            method='combined'
        )
        elapsed_sample = time.time() - start
        print(f"✓ Sample k-range analysis completed in {elapsed_sample:.3f}s")
        print(f"  This ran 7 different k values on {sample_size} points")
        
        # Estimate full dataset time
        estimated_full_time = elapsed_sample * (features_scaled.shape[0] / sample_size)**2  # O(n²) scaling
        print(f"  Estimated time for full dataset: {estimated_full_time:.1f}s")
        
        if estimated_full_time > 300:
            print("  ⚠ Would timeout on full dataset!")
            
    except Exception as e:
        elapsed_sample = time.time() - start
        print(f"✗ Sample k-range analysis failed after {elapsed_sample:.3f}s: {e}")
    
    print(f"\nSolution: Always specify k_range=[desired_k] to avoid redundant computation!")

if __name__ == '__main__':
    main()