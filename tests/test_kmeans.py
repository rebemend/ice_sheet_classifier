#!/usr/bin/env python3
"""
Test k-means clustering performance.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset
from features.feature_sets import create_feature_set
from utils.scaling import scale_features_for_clustering
from clustering.kmeans_runner import run_kmeans_analysis

def main():
    print("Testing K-Means Performance")
    print("=" * 30)
    
    # Load and prepare data quickly
    print("Loading and preparing data...")
    start = time.time()
    
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    
    features, feature_names, valid_mask = create_feature_set(unified_data, 'primary')
    features_scaled, scaler, scaling_mask = scale_features_for_clustering(
        features, feature_names, 'standard'
    )
    
    prep_time = time.time() - start
    print(f"Data preparation: {prep_time:.3f}s")
    print(f"Final data shape: {features_scaled.shape}")
    
    # Test k-means on different sample sizes
    sample_sizes = [1000, 5000, 10000, features_scaled.shape[0]]
    
    for n_sample in sample_sizes:
        if n_sample > features_scaled.shape[0]:
            continue
            
        print(f"\nTesting k-means with {n_sample} samples...")
        
        if n_sample < features_scaled.shape[0]:
            # Sample data
            sample_indices = np.random.choice(
                features_scaled.shape[0], n_sample, replace=False
            )
            test_features = features_scaled[sample_indices]
        else:
            test_features = features_scaled
        
        start = time.time()
        try:
            results = run_kmeans_analysis(
                test_features, feature_names, [3], 'combined'
            )
            elapsed = time.time() - start
            
            silhouette = results['optimal_result']['silhouette_avg']
            print(f"✓ K-means completed in {elapsed:.3f}s (silhouette: {silhouette:.3f})")
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"✗ K-means failed after {elapsed:.3f}s: {e}")
            break
    
    print("\nK-means performance test completed!")

if __name__ == '__main__':
    main()