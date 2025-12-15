#!/usr/bin/env python3
"""
Test k-means clustering directly, bypassing slow functions.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset
from utils.scaling import scale_features_for_clustering
from clustering.kmeans_runner import run_kmeans_analysis

def main():
    print("Testing Direct K-Means Clustering")
    print("=" * 35)
    
    # Load processed dataset quickly
    start = time.time()
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    load_time = time.time() - start
    print(f"Data loaded in {load_time:.3f}s")
    print(f"Dataset shape: {unified_data['x'].shape}")
    
    # Directly extract primary features without going through slow create_feature_set
    # Primary features: ['dudx', 'speed', 'mu', 'anisotropy'] 
    print("Directly extracting primary features...")
    start = time.time()
    
    # Extract fields and flatten them
    if 'primary_strain' in unified_data:
        dudx = unified_data['primary_strain'].flatten()
    else:
        dudx = unified_data['dudx'].flatten()
    
    speed = unified_data['speed'].flatten()  # Already flattened due to our fix
    mu = unified_data['mu'].flatten()       # Already flattened due to our fix
    anisotropy = unified_data['anisotropy'].flatten()  # Already flattened due to our fix
    
    # Create feature array
    features = np.column_stack([dudx, speed, mu, anisotropy])
    feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
    
    # Create validity mask (remove NaN/inf values)
    valid_mask = np.all(np.isfinite(features), axis=1)
    valid_features = features[valid_mask]
    
    extract_time = time.time() - start
    print(f"Feature extraction: {extract_time:.3f}s")
    print(f"Feature shape: {valid_features.shape}")
    print(f"Valid points: {np.sum(valid_mask)} / {len(valid_mask)} ({100*np.mean(valid_mask):.1f}%)")
    
    # Test feature scaling
    print("Scaling features...")
    start = time.time()
    features_scaled, scaler, scaling_mask = scale_features_for_clustering(
        valid_features, feature_names, 'standard'
    )
    scale_time = time.time() - start
    print(f"Feature scaling: {scale_time:.3f}s")
    print(f"Scaled shape: {features_scaled.shape}")
    
    # Test k-means with different sample sizes
    sample_sizes = [1000, 10000, features_scaled.shape[0]]
    
    for n_sample in sample_sizes:
        if n_sample > features_scaled.shape[0]:
            continue
            
        print(f"\nTesting k-means with {n_sample} samples...")
        
        if n_sample < features_scaled.shape[0]:
            # Sample data randomly
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(
                features_scaled.shape[0], n_sample, replace=False
            )
            test_features = features_scaled[sample_indices]
        else:
            test_features = features_scaled
            n_sample = features_scaled.shape[0]
        
        start = time.time()
        try:
            print(f"  Running k-means on {test_features.shape[0]} points...")
            results = run_kmeans_analysis(
                test_features, feature_names, [3], 'combined'
            )
            elapsed = time.time() - start
            
            main_result = results['optimal_result']
            silhouette = main_result['silhouette_avg']
            inertia = main_result['inertia']
            
            print(f"  ✓ K-means completed in {elapsed:.3f}s")
            print(f"    Silhouette score: {silhouette:.3f}")
            print(f"    Inertia: {inertia:.2e}")
            
            # Performance analysis
            if n_sample == features_scaled.shape[0]:
                total_pipeline_time = load_time + extract_time + scale_time + elapsed
                print(f"\n  Full Pipeline Performance:")
                print(f"    Data loading: {load_time:.3f}s ({load_time/total_pipeline_time*100:.1f}%)")
                print(f"    Feature extraction: {extract_time:.3f}s ({extract_time/total_pipeline_time*100:.1f}%)")
                print(f"    Feature scaling: {scale_time:.3f}s ({scale_time/total_pipeline_time*100:.1f}%)")
                print(f"    K-means clustering: {elapsed:.3f}s ({elapsed/total_pipeline_time*100:.1f}%)")
                print(f"    Total pipeline: {total_pipeline_time:.3f}s")
                
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ K-means failed after {elapsed:.3f}s: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\nDirect clustering test completed!")

if __name__ == '__main__':
    main()