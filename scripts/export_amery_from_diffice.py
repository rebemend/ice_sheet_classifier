#!/usr/bin/env python3
"""
Export Amery Ice Shelf data from DIFFICE_jax repository.

This script extracts the required fields from DIFFICE_jax Amery case
and exports them to a format suitable for the ice shelf classifier.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.extract_diffice_amery import load_and_process_diffice_amery


def main():
    parser = argparse.ArgumentParser(description='Export Amery Ice Shelf data from DIFFICE_jax')
    parser.add_argument('--diffice_repo', required=True,
                       help='Path to DIFFICE_jax repository root')
    parser.add_argument('--output_dir', default='data/raw/diffice_amery',
                       help='Output directory for extracted data')
    parser.add_argument('--amery_case', default='examples/real_data/amery',
                       help='Path to Amery case within DIFFICE repo')
    
    args = parser.parse_args()
    
    # Construct paths
    diffice_repo = Path(args.diffice_repo)
    amery_path = diffice_repo / args.amery_case
    output_dir = Path(args.output_dir)
    
    print("DIFFICE_jax Amery Data Export")
    print("=" * 35)
    print(f"DIFFICE repository: {diffice_repo}")
    print(f"Amery case path: {amery_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if paths exist
    if not diffice_repo.exists():
        print(f"Error: DIFFICE repository not found at {diffice_repo}")
        return 1
    
    if not amery_path.exists():
        print(f"Error: Amery case not found at {amery_path}")
        print("Available directories in examples/real_data:")
        real_data_path = diffice_repo / 'examples' / 'real_data'
        if real_data_path.exists():
            for item in real_data_path.iterdir():
                if item.is_dir():
                    print(f"  {item.name}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Scanning DIFFICE Amery directory...")
    
    # List available files
    available_files = list(amery_path.rglob('*'))
    data_files = [f for f in available_files if f.is_file() and 
                  f.suffix.lower() in ['.npy', '.npz', '.pkl', '.pickle', '.mat']]
    
    print(f"Found {len(data_files)} potential data files:")
    for file in data_files:
        rel_path = file.relative_to(amery_path)
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"  {rel_path} ({file_size:.1f} MB)")
    print()
    
    # Try to load and process data
    try:
        print("Attempting to load DIFFICE Amery data...")
        processed_data = load_and_process_diffice_amery(str(amery_path))
        
        print("Successfully loaded DIFFICE data!")
        print(f"Available fields: {list(processed_data.keys())}")
        
        # Print data shapes and basic info
        for field, data in processed_data.items():
            if isinstance(data, np.ndarray):
                print(f"  {field}: shape {data.shape}, dtype {data.dtype}")
                if data.size > 0:
                    finite_mask = np.isfinite(data)
                    n_finite = np.sum(finite_mask)
                    n_total = data.size
                    print(f"    finite values: {n_finite}/{n_total} ({100*n_finite/n_total:.1f}%)")
                    
                    if n_finite > 0:
                        finite_data = data[finite_mask]
                        print(f"    range: [{np.min(finite_data):.2e}, {np.max(finite_data):.2e}]")
        print()
        
        # Save processed data
        output_file = output_dir / 'amery_processed_data.npz'
        np.savez_compressed(output_file, **processed_data)
        print(f"Processed data saved to: {output_file}")
        
        # Create metadata file
        metadata = {
            'source_repository': str(diffice_repo),
            'amery_case_path': str(amery_path),
            'extraction_date': str(np.datetime64('now')),
            'fields': list(processed_data.keys()),
            'grid_shape': processed_data['u'].shape if 'u' in processed_data else None,
            'coordinate_ranges': {}
        }
        
        # Add coordinate ranges
        if 'x' in processed_data and 'y' in processed_data:
            x_data = processed_data['x']
            y_data = processed_data['y']
            
            if np.any(np.isfinite(x_data)):
                finite_x = x_data[np.isfinite(x_data)]
                metadata['coordinate_ranges']['x'] = [float(np.min(finite_x)), float(np.max(finite_x))]
            
            if np.any(np.isfinite(y_data)):
                finite_y = y_data[np.isfinite(y_data)]
                metadata['coordinate_ranges']['y'] = [float(np.min(finite_y)), float(np.max(finite_y))]
        
        metadata_file = output_dir / 'amery_metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_file}")
        
        # Create summary report
        summary_file = output_dir / 'extraction_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("DIFFICE_jax Amery Data Extraction Summary\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Source repository: {diffice_repo}\n")
            f.write(f"Amery case path: {amery_path}\n")
            f.write(f"Extraction date: {np.datetime64('now')}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            f.write("Extracted Fields:\n")
            f.write("-" * 16 + "\n")
            for field, data in processed_data.items():
                if isinstance(data, np.ndarray):
                    f.write(f"{field:15s}: {str(data.shape):15s} {data.dtype}\n")
            
            f.write("\nData Quality:\n")
            f.write("-" * 13 + "\n")
            for field, data in processed_data.items():
                if isinstance(data, np.ndarray) and data.size > 0:
                    finite_mask = np.isfinite(data)
                    n_finite = np.sum(finite_mask)
                    n_total = data.size
                    f.write(f"{field:15s}: {n_finite:8d}/{n_total:8d} finite ({100*n_finite/n_total:5.1f}%)\n")
            
            if metadata['coordinate_ranges']:
                f.write("\nSpatial Extent:\n")
                f.write("-" * 15 + "\n")
                for coord, (min_val, max_val) in metadata['coordinate_ranges'].items():
                    extent = max_val - min_val
                    f.write(f"{coord} range: [{min_val:.1f}, {max_val:.1f}] (extent: {extent:.1f})\n")
            
            f.write("\nFile Locations:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Processed data: {output_file}\n")
            f.write(f"Metadata: {metadata_file}\n")
            f.write(f"Summary: {summary_file}\n")
        
        print(f"Summary report saved to: {summary_file}")
        print()
        print("Data extraction completed successfully!")
        
        # Validation checks
        print("Running validation checks...")
        required_fields = ['x', 'y', 'u', 'v', 'h', 'dudx', 'speed']
        missing_fields = [field for field in required_fields if field not in processed_data]
        
        if missing_fields:
            print(f"Warning: Missing required fields: {missing_fields}")
        else:
            print("✓ All required fields present")
        
        # Check grid consistency
        if 'u' in processed_data and 'v' in processed_data:
            if processed_data['u'].shape == processed_data['v'].shape:
                print("✓ Velocity components have consistent shapes")
            else:
                print(f"Warning: Velocity shape mismatch: u{processed_data['u'].shape} vs v{processed_data['v'].shape}")
        
        # Check coordinate coverage
        if 'x' in processed_data and 'y' in processed_data:
            x_coverage = np.sum(np.isfinite(processed_data['x'])) / processed_data['x'].size
            y_coverage = np.sum(np.isfinite(processed_data['y'])) / processed_data['y'].size
            
            if x_coverage > 0.9 and y_coverage > 0.9:
                print("✓ Good coordinate coverage")
            else:
                print(f"Warning: Poor coordinate coverage: x={x_coverage:.1%}, y={y_coverage:.1%}")
        
        return 0
        
    except Exception as e:
        print(f"Error loading DIFFICE data: {e}")
        print("\nThis might indicate:")
        print("1. The Amery case is not in the expected format")
        print("2. Required data files are missing or corrupted")
        print("3. The DIFFICE_jax repository structure has changed")
        print("\nPlease check the DIFFICE_jax repository and ensure the Amery case is properly configured.")
        
        # Try to provide helpful information
        print(f"\nAvailable files in {amery_path}:")
        if amery_path.exists():
            for item in amery_path.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(amery_path)
                    print(f"  {rel_path}")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())