#!/usr/bin/env python3
"""
Debug the compute_all_features function step by step.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.assemble_dataset import load_processed_dataset
from features.strain_features import compute_velocity_features, compute_strain_rate_tensor, compute_deformation_features, validate_strain_features
from features.viscosity_features import compute_viscosity_features, compute_stress_features, validate_viscosity_features

def time_step(name, func, *args, **kwargs):
    """Time a step with timeout detection."""
    print(f"Starting {name}...")
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✓ {name} completed in {elapsed:.3f}s")
        return result, elapsed
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"⚠ {name} interrupted after {elapsed:.3f}s")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {name} failed after {elapsed:.3f}s: {e}")
        raise

def main():
    print("Debug compute_all_features Step by Step")
    print("=" * 45)
    
    # Load data
    processed_data_path = "real_data_analysis/processed_dataset.npz"
    unified_data, feature_data = load_processed_dataset(processed_data_path)
    
    print(f"Dataset loaded: {unified_data['x'].shape}")
    
    # Start replicating compute_all_features step by step
    all_features = unified_data.copy()
    
    # Step 1: Handle strain field compatibility
    print("\nStep 1: Strain field compatibility...")
    if 'primary_strain' in unified_data and 'dudx' not in unified_data:
        all_features['dudx'] = unified_data['primary_strain']
        print("Using primary_strain as dudx for feature compatibility")
    
    # Step 2: Extract basic fields
    print("\nStep 2: Extract basic fields...")
    u = all_features['u']
    v = all_features['v']
    print(f"u shape: {u.shape}, v shape: {v.shape}")
    
    # Step 3: Compute velocity features
    print("\nStep 3: Velocity features...")
    velocity_features, vel_time = time_step(
        "compute_velocity_features", compute_velocity_features, u, v
    )
    all_features.update(velocity_features)
    
    # Step 4: Strain rate features
    print("\nStep 4: Strain rate features...")
    required_strain_keys = ['dudx', 'dvdy', 'dudy', 'dvdx']
    if all(key in all_features for key in required_strain_keys):
        print("Required strain gradients found")
        
        # Check if already computed
        strain_tensor_keys = ['epsilon_xx', 'epsilon_yy', 'epsilon_xy', 'effective_strain']
        if all(key in all_features for key in strain_tensor_keys):
            print("Using existing strain tensor values")
        else:
            print("Computing strain rate tensor...")
            strain_tensor, strain_time = time_step(
                "compute_strain_rate_tensor",
                compute_strain_rate_tensor,
                all_features['dudx'], all_features['dvdy'],
                all_features['dudy'], all_features['dvdx']
            )
            all_features.update(strain_tensor)
            
            print("Computing deformation features...")
            deformation_features, deform_time = time_step(
                "compute_deformation_features", 
                compute_deformation_features, strain_tensor
            )
            all_features.update(deformation_features)
            
            print("Validating strain features...")
            validate_result, validate_time = time_step(
                "validate_strain_features", validate_strain_features, strain_tensor
            )
    
    # Step 5: Viscosity features
    print("\nStep 5: Viscosity features...")
    if 'mu' in all_features and 'eta' in all_features:
        mu = all_features['mu']
        eta = all_features['eta']
        h = all_features.get('h')
        
        print(f"mu shape: {mu.shape}, eta shape: {eta.shape}")
        
        print("Computing basic viscosity features...")
        viscosity_features, visc_time = time_step(
            "compute_viscosity_features", compute_viscosity_features, mu, eta, h
        )
        all_features.update(viscosity_features)
        
        # Stress features
        if 'effective_strain' in all_features:
            print("Computing stress features...")
            stress_features, stress_time = time_step(
                "compute_stress_features", 
                compute_stress_features, mu, eta, all_features
            )
            all_features.update(stress_features)
        
        print("Validating viscosity features...")
        validate_visc_result, validate_visc_time = time_step(
            "validate_viscosity_features", validate_viscosity_features, viscosity_features
        )
    
    print(f"\nTotal features computed: {len(all_features)}")
    print("Debug completed successfully!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"\nScript failed: {e}")
        import traceback
        traceback.print_exc()