#!/usr/bin/env python3
"""
Test our kmeans_runner vs raw sklearn to find the bottleneck.
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
from clustering.kmeans_runner import run_kmeans_analysis

def test_step_by_step():
    """Test each step of our pipeline vs sklearn directly."""
    print("Testing Step-by-Step Pipeline vs Raw Sklearn")
    print("=" * 50)
    
    # Get data
    print("Loading data...")
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
    feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
    
    print(f"Data ready: {features.shape}")
    
    # Test 1: Raw sklearn (unscaled)
    print(f"\nTest 1: Raw sklearn K-means (unscaled)")
    start = time.time()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features)
    elapsed = time.time() - start
    print(f"Raw sklearn: {elapsed:.3f}s")
    
    # Test 2: Raw sklearn (manually scaled)
    print(f"\nTest 2: Raw sklearn K-means (manually scaled)")
    start = time.time()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    labels_scaled = kmeans.fit_predict(features_scaled)
    elapsed = time.time() - start
    print(f"Raw sklearn (scaled): {elapsed:.3f}s")
    
    # Test 3: Our kmeans_runner (small sample first)
    sample_size = 10000
    print(f"\nTest 3: Our kmeans_runner ({sample_size} samples)")
    sample_indices = np.random.choice(features_scaled.shape[0], sample_size, replace=False)
    sample_features = features_scaled[sample_indices]
    
    start = time.time()
    try:
        results = run_kmeans_analysis(sample_features, feature_names, [3], 'combined')
        elapsed = time.time() - start
        print(f"Our kmeans_runner (sample): {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"Our kmeans_runner (sample): FAILED after {elapsed:.3f}s - {e}")
        return
    
    # Test 4: Our kmeans_runner (full dataset) - with timeout monitoring
    print(f"\nTest 4: Our kmeans_runner (full dataset)")
    print("Starting full dataset test...")
    start = time.time()
    
    # Set a reasonable timeout
    timeout_limit = 60  # 60 seconds
    
    try:
        results = run_kmeans_analysis(features_scaled, feature_names, [3], 'combined')
        elapsed = time.time() - start
        print(f"Our kmeans_runner (full): {elapsed:.3f}s")
        
        # Show results
        main_result = results['optimal_result']
        print(f"  Silhouette: {main_result['silhouette_avg']:.3f}")
        print(f"  Inertia: {main_result['inertia']:.2e}")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"Our kmeans_runner (full): FAILED after {elapsed:.3f}s")
        print(f"Error: {e}")
        print("This indicates the bottleneck is in our wrapper functions!")
        
        # Let's check what happens if we interrupt it manually
        import traceback
        traceback.print_exc()

def check_kmeans_runner_components():
    """Check individual components of kmeans_runner."""
    print("\nAnalyzing kmeans_runner Components")
    print("=" * 40)
    
    # Load the kmeans_runner source to understand what it's doing
    from clustering import kmeans_runner
    import inspect
    
    # Look at the run_kmeans_analysis function
    source = inspect.getsource(kmeans_runner.run_kmeans_analysis)
    print("kmeans_runner.run_kmeans_analysis source:")
    print("-" * 40)
    lines = source.split('\n')
    for i, line in enumerate(lines[:30], 1):  # First 30 lines
        print(f"{i:2d}: {line}")
    
    if len(lines) > 30:
        print(f"... ({len(lines)-30} more lines)")

def main():
    test_step_by_step()
    check_kmeans_runner_components()

if __name__ == '__main__':
    main()