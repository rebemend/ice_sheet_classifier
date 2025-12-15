#!/usr/bin/env python3
"""
Minimal test to find exactly where the hang occurs.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_step_by_step():
    """Test each step individually with timing."""
    print("Minimal Step-by-Step Test")
    print("=" * 30)
    
    # Step 1: Test basic imports
    print("Step 1: Testing imports...")
    start = time.time()
    try:
        from data_loading.assemble_dataset import load_processed_dataset
        print(f"✓ Imports: {time.time() - start:.3f}s")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Step 2: Test data loading
    print("Step 2: Testing data loading...")
    start = time.time()
    try:
        processed_data_path = "real_data_analysis/processed_dataset.npz"
        print(f"Loading {processed_data_path}...")
        unified_data, feature_data = load_processed_dataset(processed_data_path)
        print(f"✓ Data loading: {time.time() - start:.3f}s")
        print(f"  Dataset shape: {unified_data['x'].shape}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    # Step 3: Test feature extraction (small operations only)
    print("Step 3: Testing feature extraction...")
    start = time.time()
    try:
        print("  Extracting primary_strain...")
        dudx = unified_data['primary_strain']
        print(f"  Shape: {dudx.shape}")
        
        print("  Flattening...")
        dudx_flat = dudx.flatten()
        print(f"  Flattened shape: {dudx_flat.shape}")
        
        print(f"✓ Feature extraction: {time.time() - start:.3f}s")
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return
    
    # Step 4: Test feature assembly (this might be slow)
    print("Step 4: Testing feature assembly...")
    start = time.time()
    try:
        print("  Getting all features...")
        speed = unified_data['speed'].flatten()
        mu = unified_data['mu'].flatten()
        anisotropy = unified_data['anisotropy'].flatten()
        
        print("  Creating feature array...")
        features_raw = np.column_stack([dudx_flat, speed, mu, anisotropy])
        print(f"  Feature array shape: {features_raw.shape}")
        
        print("  Creating validity mask...")
        valid_mask = np.all(np.isfinite(features_raw), axis=1)
        print(f"  Valid points: {np.sum(valid_mask)} / {len(valid_mask)}")
        
        print("  Filtering valid features...")
        features = features_raw[valid_mask]
        print(f"  Final shape: {features.shape}")
        
        print(f"✓ Feature assembly: {time.time() - start:.3f}s")
    except Exception as e:
        print(f"✗ Feature assembly failed after {time.time() - start:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test scaling (this might be slow)
    print("Step 5: Testing feature scaling...")
    start = time.time()
    try:
        from sklearn.preprocessing import StandardScaler
        print("  Creating scaler...")
        scaler = StandardScaler()
        
        print("  Fitting scaler...")
        scaler.fit(features)
        
        print("  Transforming features...")
        features_scaled = scaler.transform(features)
        print(f"  Scaled shape: {features_scaled.shape}")
        
        print(f"✓ Feature scaling: {time.time() - start:.3f}s")
    except Exception as e:
        print(f"✗ Feature scaling failed after {time.time() - start:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Test raw k-means only
    print("Step 6: Testing raw k-means...")
    start = time.time()
    try:
        from sklearn.cluster import KMeans
        print("  Creating KMeans...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=1, max_iter=100)  # Reduced params
        
        print("  Running fit_predict...")
        labels = kmeans.fit_predict(features_scaled)
        print(f"  Labels shape: {labels.shape}")
        print(f"  Unique labels: {np.unique(labels)}")
        
        print(f"✓ Raw k-means: {time.time() - start:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Raw k-means failed after {time.time() - start:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_step_by_step()
    
    if success:
        print("\n✓ All steps completed successfully!")
        print("The timeout issue is likely in our wrapper functions.")
    else:
        print("\n✗ Found the bottleneck in the step-by-step test.")

if __name__ == '__main__':
    main()