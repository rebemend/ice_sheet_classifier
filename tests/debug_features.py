#!/usr/bin/env python3
"""
Granular debug script for feature creation bottleneck.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset
from features.strain_features import compute_velocity_features
from features.viscosity_features import compute_viscosity_features

def time_function(func, name, *args, **kwargs):
    """Time a function execution with detailed output."""
    print(f"Starting {name}...")
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"✓ {name} completed in {end_time - start_time:.3f}s")
        return result
    except Exception as e:
        end_time = time.time()
        print(f"✗ {name} failed after {end_time - start_time:.3f}s: {e}")
        raise

def main():
    print("Debug Feature Creation Performance")
    print("=" * 40)
    
    # Load data
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = time_function(
        load_processed_dataset, "Loading processed dataset", processed_data_path
    )
    
    print(f"Data shape: {unified_data['x'].shape}")
    print(f"Total elements: {unified_data['x'].size}")
    
    # Check what fields we have
    print(f"Available fields: {list(unified_data.keys())}")
    
    # Get reference shape and check for shape consistency
    reference_shape = unified_data['x'].shape
    print(f"Reference shape: {reference_shape}")
    
    shape_issues = []
    for key, arr in unified_data.items():
        if isinstance(arr, np.ndarray) and arr.size == np.prod(reference_shape):
            if arr.shape != reference_shape:
                shape_issues.append(f"{key}: {arr.shape} (should be {reference_shape})")
    
    if shape_issues:
        print("Shape inconsistencies found:")
        for issue in shape_issues:
            print(f"  {issue}")
    else:
        print("All array shapes are consistent")
    
    # Test velocity features
    u = unified_data['u']
    v = unified_data['v']
    print(f"u shape: {u.shape}, v shape: {v.shape}")
    
    velocity_features = time_function(
        compute_velocity_features, "Velocity features", u, v
    )
    
    # Test viscosity features if available
    if 'mu' in unified_data and 'eta' in unified_data:
        mu = unified_data['mu']
        eta = unified_data['eta']
        h = unified_data.get('h')
        
        print(f"mu shape: {mu.shape}, eta shape: {eta.shape}")
        print(f"mu range: [{np.nanmin(mu):.2e}, {np.nanmax(mu):.2e}]")
        print(f"eta range: [{np.nanmin(eta):.2e}, {np.nanmax(eta):.2e}]")
        
        # Test viscosity features step by step
        print("Testing viscosity feature computation...")
        
        # Test basic anisotropy computation
        print("Computing anisotropy...")
        start = time.time()
        anisotropy = mu / (eta + 1e-12)
        print(f"Anisotropy computed in {time.time() - start:.3f}s")
        
        # Test log transforms
        print("Computing log transforms...")
        start = time.time()
        epsilon = 1e-12
        log_mu = np.log(np.maximum(mu, epsilon))
        log_eta = np.log(np.maximum(eta, epsilon))
        print(f"Log transforms computed in {time.time() - start:.3f}s")
        
        # Test full viscosity features function
        viscosity_features = time_function(
            compute_viscosity_features, "Viscosity features", mu, eta, h
        )
        
        print(f"Viscosity features created: {list(viscosity_features.keys())}")
    
    print("\nDebug completed!")

if __name__ == '__main__':
    main()