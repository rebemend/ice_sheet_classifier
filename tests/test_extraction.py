#!/usr/bin/env python3
"""
Test feature extraction bottleneck.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset
from features.feature_sets import compute_all_features, FeatureSetDefinitions, extract_features_from_data

def main():
    print("Testing Feature Extraction Performance")
    print("=" * 40)
    
    # Load data
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    
    print(f"Dataset loaded: {unified_data['x'].shape}")
    
    # Compute all features quickly
    start = time.time()
    all_features = compute_all_features(unified_data)
    compute_time = time.time() - start
    print(f"Compute all features: {compute_time:.3f}s")
    print(f"Total features: {len(all_features)}")
    
    # Test extraction step
    feature_names = FeatureSetDefinitions.get_primary_features()
    print(f"Target features: {feature_names}")
    
    print("Testing feature extraction...")
    start = time.time()
    try:
        features = extract_features_from_data(all_features, feature_names)
        extract_time = time.time() - start
        print(f"✓ Feature extraction completed in {extract_time:.3f}s")
        print(f"Feature array shape: {features.shape}")
        
        # Test mask creation
        print("Testing mask creation...")
        start = time.time()
        valid_mask = np.all(np.isfinite(features), axis=1)
        mask_time = time.time() - start
        print(f"✓ Mask creation completed in {mask_time:.3f}s")
        print(f"Valid points: {np.sum(valid_mask)} / {len(valid_mask)} ({100*np.mean(valid_mask):.1f}%)")
        
        # Test filtering
        print("Testing filtering...")
        start = time.time()
        valid_features = features[valid_mask]
        filter_time = time.time() - start
        print(f"✓ Filtering completed in {filter_time:.3f}s")
        print(f"Final shape: {valid_features.shape}")
        
        # Performance summary
        total_time = compute_time + extract_time + mask_time + filter_time
        print(f"\nPerformance breakdown:")
        print(f"  Compute features: {compute_time:.3f}s ({compute_time/total_time*100:.1f}%)")
        print(f"  Extract features: {extract_time:.3f}s ({extract_time/total_time*100:.1f}%)")
        print(f"  Create mask: {mask_time:.3f}s ({mask_time/total_time*100:.1f}%)")
        print(f"  Filter data: {filter_time:.3f}s ({filter_time/total_time*100:.1f}%)")
        print(f"  Total: {total_time:.3f}s")
        
    except Exception as e:
        extract_time = time.time() - start
        print(f"✗ Feature extraction failed after {extract_time:.3f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()