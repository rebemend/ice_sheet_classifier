#!/usr/bin/env python3
"""
Performance test to identify clustering pipeline bottleneck.
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

def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def main():
    print("Performance Test for Clustering Pipeline")
    print("=" * 45)
    
    # Test with existing processed data
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    
    print("Step 1: Loading processed dataset...")
    (unified_data, feature_data), load_time = time_function(
        load_processed_dataset, processed_data_path
    )
    print(f"Load time: {load_time:.3f}s")
    print(f"Dataset shape: {unified_data['x'].shape}")
    print(f"Valid points: {np.sum(feature_data['mask'])}")
    print()
    
    # Test feature creation in steps
    print("Step 2a: Computing all features...")
    from features.feature_sets import compute_all_features
    (all_features), all_features_time = time_function(
        compute_all_features, unified_data
    )
    print(f"All features computation time: {all_features_time:.3f}s")
    
    print("Step 2b: Extracting feature set...")
    from features.feature_sets import FeatureSetDefinitions, extract_features_from_data
    feature_names = FeatureSetDefinitions.get_primary_features()
    print(f"Target features: {feature_names}")
    
    (features), extract_time = time_function(
        extract_features_from_data, all_features, feature_names
    )
    print(f"Feature extraction time: {extract_time:.3f}s")
    
    print("Step 2c: Creating validity mask...")
    start_mask = time.time()
    valid_mask = np.all(np.isfinite(features), axis=1)
    mask_time = time.time() - start_mask
    print(f"Mask creation time: {mask_time:.3f}s")
    print(f"Valid points: {np.sum(valid_mask)} / {len(valid_mask)}")
    
    print("Step 2d: Filtering valid features...")
    start_filter = time.time()
    valid_features = features[valid_mask]
    filter_time = time.time() - start_filter
    print(f"Filter time: {filter_time:.3f}s")
    
    features = valid_features  # Use filtered features
    print(f"Final feature shape: {features.shape}")
    print()
    
    # Test feature scaling
    print("Step 3: Scaling features...")
    (features_scaled, scaler, scaling_mask), scaling_time = time_function(
        scale_features_for_clustering, features, feature_names, 'standard'
    )
    print(f"Scaling time: {scaling_time:.3f}s")
    print(f"Scaled feature shape: {features_scaled.shape}")
    print()
    
    # Test on sample of data first
    n_sample = min(10000, features_scaled.shape[0])
    if features_scaled.shape[0] > n_sample:
        print(f"Step 4a: Testing k-means on sample ({n_sample} points)...")
        sample_indices = np.random.choice(features_scaled.shape[0], n_sample, replace=False)
        features_sample = features_scaled[sample_indices]
        
        (sample_results), sample_time = time_function(
            run_kmeans_analysis, features_sample, feature_names, [3], 'combined'
        )
        print(f"Sample clustering time: {sample_time:.3f}s")
        print(f"Sample silhouette score: {sample_results['optimal_result']['silhouette_avg']:.3f}")
        print()
    
    # Test on full dataset
    print(f"Step 4b: Testing k-means on full dataset ({features_scaled.shape[0]} points)...")
    start_full = time.time()
    try:
        clustering_results, full_time = time_function(
            run_kmeans_analysis, features_scaled, feature_names, [3], 'combined'
        )
        print(f"Full clustering time: {full_time:.3f}s")
        print(f"Silhouette score: {clustering_results['optimal_result']['silhouette_avg']:.3f}")
        
        # Performance summary
        total_time = load_time + feature_time + scaling_time + full_time
        print(f"\nPerformance Summary:")
        print(f"  Load data: {load_time:.3f}s ({load_time/total_time*100:.1f}%)")
        print(f"  Create features: {feature_time:.3f}s ({feature_time/total_time*100:.1f}%)")
        print(f"  Scale features: {scaling_time:.3f}s ({scaling_time/total_time*100:.1f}%)")
        print(f"  K-means clustering: {full_time:.3f}s ({full_time/total_time*100:.1f}%)")
        print(f"  Total: {total_time:.3f}s")
        
    except Exception as e:
        elapsed = time.time() - start_full
        print(f"Clustering failed after {elapsed:.3f}s with error: {e}")
        print("This indicates the bottleneck is in the clustering step")

if __name__ == '__main__':
    main()