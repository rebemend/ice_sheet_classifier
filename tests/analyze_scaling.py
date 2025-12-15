#!/usr/bin/env python3
"""
Systematic analysis of the scaling issue in the ice shelf classifier.
"""

import sys
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset

def test_kmeans_scaling(features, sample_sizes, max_time=30):
    """Test k-means performance across different sample sizes."""
    print("Testing K-Means Scaling Behavior")
    print("=" * 40)
    print(f"Full dataset size: {features.shape}")
    print(f"Features: {features.shape[1]}D")
    print()
    
    results = []
    
    for n_sample in sample_sizes:
        if n_sample > features.shape[0]:
            print(f"Skipping {n_sample} (exceeds dataset size)")
            continue
            
        print(f"Testing with {n_sample:,} samples...")
        
        # Sample the data
        np.random.seed(42)
        if n_sample < features.shape[0]:
            indices = np.random.choice(features.shape[0], n_sample, replace=False)
            sample_data = features[indices]
        else:
            sample_data = features
        
        # Test standard k-means
        start_time = time.time()
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(sample_data)
            elapsed = time.time() - start_time
            
            if elapsed > max_time:
                print(f"  Standard K-Means: {elapsed:.1f}s (TOO SLOW)")
                results.append((n_sample, elapsed, None, "timeout"))
                break
            else:
                inertia = kmeans.inertia_
                print(f"  Standard K-Means: {elapsed:.3f}s (inertia: {inertia:.2e})")
                results.append((n_sample, elapsed, inertia, "success"))
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Standard K-Means: FAILED after {elapsed:.3f}s - {e}")
            results.append((n_sample, elapsed, None, "error"))
            break
    
    return results

def test_alternative_algorithms(features, n_sample=10000):
    """Test alternative clustering algorithms."""
    print(f"\nTesting Alternative Algorithms ({n_sample:,} samples)")
    print("=" * 50)
    
    # Sample the data
    np.random.seed(42)
    if n_sample < features.shape[0]:
        indices = np.random.choice(features.shape[0], n_sample, replace=False)
        sample_data = features[indices]
    else:
        sample_data = features
    
    algorithms = [
        ("Standard K-Means", KMeans(n_clusters=3, random_state=42, n_init=10)),
        ("Mini-Batch K-Means", MiniBatchKMeans(n_clusters=3, random_state=42, n_init=10, batch_size=1000)),
        ("Fast K-Means", KMeans(n_clusters=3, random_state=42, n_init=1, max_iter=100))
    ]
    
    for name, algorithm in algorithms:
        print(f"Testing {name}...")
        start_time = time.time()
        try:
            labels = algorithm.fit_predict(sample_data)
            elapsed = time.time() - start_time
            inertia = algorithm.inertia_
            print(f"  ✓ {name}: {elapsed:.3f}s (inertia: {inertia:.2e})")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ {name}: FAILED after {elapsed:.3f}s - {e}")

def analyze_data_characteristics(features):
    """Analyze the characteristics of the feature data."""
    print("Data Characteristics Analysis")
    print("=" * 35)
    print(f"Dataset shape: {features.shape}")
    print(f"Total points: {features.shape[0]:,}")
    print(f"Features: {features.shape[1]}")
    print()
    
    # Check for any obvious issues
    print("Data Quality Checks:")
    has_nan = np.any(np.isnan(features))
    has_inf = np.any(np.isinf(features))
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ⚠ Data quality issues detected!")
        return False
    
    # Feature statistics
    print("\nFeature Statistics:")
    feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
    for i, name in enumerate(feature_names):
        if i < features.shape[1]:
            feat_data = features[:, i]
            print(f"  {name}:")
            print(f"    Range: [{np.min(feat_data):.2e}, {np.max(feat_data):.2e}]")
            print(f"    Mean±Std: {np.mean(feat_data):.2e}±{np.std(feat_data):.2e}")
    
    # Check for scale differences
    feature_ranges = np.ptp(features, axis=0)  # peak-to-peak range
    max_range = np.max(feature_ranges)
    min_range = np.min(feature_ranges)
    scale_ratio = max_range / (min_range + 1e-12)
    
    print(f"\nScale Analysis:")
    print(f"  Feature ranges: {feature_ranges}")
    print(f"  Scale ratio: {scale_ratio:.1f}")
    
    if scale_ratio > 1000:
        print("  ⚠ Large scale differences detected - scaling recommended")
    
    return True

def main():
    print("Systematic Ice Shelf Classifier Scaling Analysis")
    print("=" * 55)
    
    # Load real data
    print("Loading processed dataset...")
    start = time.time()
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    load_time = time.time() - start
    print(f"Data loaded in {load_time:.3f}s")
    
    # Extract features directly (bypass slow functions)
    print("\nExtracting primary features...")
    start = time.time()
    
    dudx = unified_data['primary_strain'].flatten() if 'primary_strain' in unified_data else unified_data['dudx'].flatten()
    speed = unified_data['speed'].flatten()
    mu = unified_data['mu'].flatten()
    anisotropy = unified_data['anisotropy'].flatten()
    
    # Create feature array and remove invalid values
    features_raw = np.column_stack([dudx, speed, mu, anisotropy])
    valid_mask = np.all(np.isfinite(features_raw), axis=1)
    features = features_raw[valid_mask]
    
    extract_time = time.time() - start
    print(f"Feature extraction: {extract_time:.3f}s")
    print(f"Valid features shape: {features.shape}")
    
    # Analyze data characteristics
    print()
    data_ok = analyze_data_characteristics(features)
    
    if not data_ok:
        print("Data quality issues detected. Aborting analysis.")
        return
    
    # Test scaling behavior with progressively larger samples
    sample_sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, features.shape[0]]
    print()
    scaling_results = test_kmeans_scaling(features, sample_sizes)
    
    # Test alternative algorithms on a reasonable sample size
    test_alternative_algorithms(features, 10000)
    
    # Analyze results
    print(f"\nScaling Analysis Results:")
    print("=" * 30)
    
    successful_results = [r for r in scaling_results if r[3] == "success"]
    if len(successful_results) >= 2:
        # Estimate scaling behavior
        sizes = [r[0] for r in successful_results]
        times = [r[1] for r in successful_results]
        
        print("Performance scaling:")
        for size, time_taken in zip(sizes, times):
            print(f"  {size:,} points: {time_taken:.3f}s")
        
        # Estimate time for full dataset
        if len(successful_results) >= 2:
            # Simple linear extrapolation (k-means is roughly linear in n for fixed k,d)
            last_size, last_time = successful_results[-1][0], successful_results[-1][1]
            full_size = features.shape[0]
            estimated_full_time = last_time * (full_size / last_size)
            
            print(f"\nEstimated time for full dataset ({full_size:,} points): {estimated_full_time:.1f}s")
            
            if estimated_full_time > 300:  # 5 minutes
                print("⚠ Full dataset clustering will be very slow")
                print("Recommended solutions:")
                print("  1. Use Mini-Batch K-Means")
                print("  2. Sample the dataset (e.g., use 50k-100k points)")
                print("  3. Reduce max_iter or n_init parameters")
    
    print("\nAnalysis completed.")

if __name__ == '__main__':
    main()