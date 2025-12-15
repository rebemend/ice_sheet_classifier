#!/usr/bin/env python3
"""
Main k-means clustering script for ice shelf classification.

This script runs the complete clustering pipeline with a specified k value
and generates comprehensive visualization and analysis outputs.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.assemble_dataset import create_complete_dataset, load_processed_dataset
from features.feature_sets import create_feature_set, get_feature_set_description
from utils.scaling import scale_features_for_clustering, get_scaling_summary
from clustering.kmeans_runner import run_kmeans_analysis
from visualization.spatial_maps import create_spatial_analysis_summary
from visualization.feature_space import create_feature_space_summary


def main():
    parser = argparse.ArgumentParser(description='Run k-means clustering for ice shelf classification')
    parser.add_argument('--diffice_data', required=True,
                       help='Path to DIFFICE Amery data directory')
    parser.add_argument('--viscosity_data', required=True,
                       help='Path to viscosity MATLAB file (results.mat)')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory for results and plots')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of clusters (default: 3)')
    parser.add_argument('--feature_set', default='primary',
                       choices=['baseline', 'primary', 'extended', 'stress'],
                       help='Feature set to use for clustering')
    parser.add_argument('--scaling_method', default='standard',
                       choices=['standard', 'minmax', 'robust', 'none'],
                       help='Feature scaling method')
    parser.add_argument('--processed_data', default=None,
                       help='Path to pre-processed dataset (skip data loading if provided)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--analysis_only', action='store_true',
                       help='Run analysis without creating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Ice Shelf K-Means Clustering")
    print("=" * 40)
    print(f"Number of clusters (k): {args.k}")
    print(f"Feature set: {args.feature_set}")
    print(f"Scaling method: {args.scaling_method}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: Load or create dataset
    if args.processed_data:
        print("Loading pre-processed dataset...")
        unified_data, feature_data = load_processed_dataset(args.processed_data)
    else:
        print("Creating dataset from raw data...")
        unified_data, feature_data = create_complete_dataset(
            args.diffice_data,
            args.viscosity_data,
            output_path=str(output_dir / 'processed_dataset.npz')
        )
    
    print(f"Dataset loaded: {np.sum(feature_data['mask'])} valid points")
    print()
    
    # Step 2: Create feature set
    print(f"Creating {args.feature_set} feature set...")
    features, feature_names, valid_mask = create_feature_set(unified_data, args.feature_set)
    
    print(f"Feature set description:")
    print(get_feature_set_description(args.feature_set))
    print(f"Features: {feature_names}")
    print(f"Feature array shape: {features.shape}")
    print()
    
    # Step 3: Scale features
    print(f"Scaling features using {args.scaling_method} method...")
    features_scaled, scaler, scaling_mask = scale_features_for_clustering(
        features, feature_names, method=args.scaling_method
    )
    
    # Print scaling summary
    scaling_summary = get_scaling_summary(features, features_scaled, feature_names)
    print(scaling_summary)
    print()
    
    # Step 4: Run k-means clustering
    print(f"Running k-means clustering with k={args.k}...")
    
    # Run full analysis
    clustering_results = run_kmeans_analysis(
        features_scaled, feature_names, 
        k_range=[args.k], method='combined'
    )
    
    # Extract main results
    main_result = clustering_results['optimal_result']
    labels = main_result['labels']
    centroids = main_result['centroids']
    inertia = main_result['inertia']
    silhouette_avg = main_result['silhouette_avg']
    silhouette_samples = main_result['silhouette_samples']
    
    print(f"Clustering completed successfully!")
    print(f"Inertia: {inertia:.2e}")
    print(f"Silhouette score: {silhouette_avg:.3f}")
    print()
    
    # Step 5: Analyze cluster quality
    quality_analysis = clustering_results['quality_analysis']
    
    print("Cluster Quality Analysis:")
    print("-" * 25)
    cluster_balance = quality_analysis['cluster_balance']
    print(f"Cluster sizes: {cluster_balance['sizes']}")
    print(f"Cluster proportions: {[f'{p:.1%}' for p in cluster_balance['proportions']]}")
    print(f"Balance ratio (min/max): {cluster_balance['balance_ratio']:.2f}")
    
    if 'separation' in quality_analysis:
        separation = quality_analysis['separation']
        print(f"Min centroid distance: {separation['min_centroid_distance']:.3f}")
        print(f"Mean centroid distance: {separation['mean_centroid_distance']:.3f}")
    print()
    
    # Step 6: Create spatial coordinates for valid points
    coordinates = feature_data['coordinates'][scaling_mask]
    grid_shape = feature_data['grid_shape']
    
    # Create feature dictionary for valid points
    features_for_viz = {}
    for feature_name in ['dudx', 'speed', 'mu', 'anisotropy']:
        if feature_name in unified_data:
            feature_values = unified_data[feature_name].flatten()[scaling_mask]
            features_for_viz[feature_name] = feature_values
    
    # Step 7: Generate visualizations
    if args.create_plots and not args.analysis_only:
        print("Creating spatial visualizations...")
        
        # Spatial maps
        spatial_figures = create_spatial_analysis_summary(
            coordinates, labels, features_for_viz, grid_shape,
            save_path=str(output_dir / 'spatial')
        )
        
        print(f"Created {len(spatial_figures)} spatial plots")
        
        # Feature space visualizations
        print("Creating feature space visualizations...")
        
        feature_space_figures = create_feature_space_summary(
            features_scaled, labels, feature_names, centroids,
            silhouette_samples=silhouette_samples,
            save_path=str(output_dir / 'feature_space')
        )
        
        print(f"Created {len(feature_space_figures)} feature space plots")
        print()
    
    # Step 8: Save clustering results
    results_file = output_dir / 'clustering_results.npz'
    
    # Map labels back to full grid
    full_labels = np.full(feature_data['mask'].shape, -1, dtype=int)
    full_labels[feature_data['mask']][scaling_mask] = labels
    
    save_data = {
        'labels': labels,
        'full_labels': full_labels,
        'centroids': centroids,
        'inertia': inertia,
        'silhouette_avg': silhouette_avg,
        'silhouette_samples': silhouette_samples,
        'coordinates': coordinates,
        'features_scaled': features_scaled,
        'feature_names': feature_names,
        'k': args.k,
        'grid_shape': grid_shape,
        'cluster_sizes': main_result['cluster_sizes']
    }
    
    # Add features for visualization
    for name, values in features_for_viz.items():
        save_data[f'feature_{name}'] = values
    
    np.savez(results_file, **save_data)
    print(f"Clustering results saved to: {results_file}")
    
    # Step 9: Save analysis summary
    summary_file = output_dir / 'clustering_summary.json'
    
    summary_data = {
        'parameters': {
            'k': args.k,
            'feature_set': args.feature_set,
            'scaling_method': args.scaling_method,
            'random_state': args.random_state,
            'features': feature_names
        },
        'data_info': {
            'total_grid_points': int(np.prod(grid_shape)),
            'valid_points': int(np.sum(feature_data['mask'])),
            'points_after_scaling': int(len(labels)),
            'grid_shape': grid_shape
        },
        'clustering_metrics': {
            'inertia': float(inertia),
            'silhouette_avg': float(silhouette_avg),
            'n_clusters': int(args.k)
        },
        'cluster_analysis': {
            'sizes': [int(x) for x in cluster_balance['sizes']],
            'proportions': [float(x) for x in cluster_balance['proportions']],
            'balance_ratio': float(cluster_balance['balance_ratio'])
        }
    }
    
    # Add centroid analysis
    centroid_analysis = quality_analysis['centroid_analysis']
    summary_data['centroids'] = {}
    for feature, analysis in centroid_analysis.items():
        summary_data['centroids'][feature] = {
            'values': [float(x) for x in analysis['centroids']],
            'range': float(analysis['range']),
            'mean': float(analysis['mean']),
            'std': float(analysis['std'])
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Analysis summary saved to: {summary_file}")
    
    # Step 10: Create text summary
    text_summary_file = output_dir / 'clustering_summary.txt'
    
    with open(text_summary_file, 'w') as f:
        f.write("Ice Shelf K-Means Clustering Summary\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Dataset: {args.diffice_data}\n")
        f.write(f"Viscosity data: {args.viscosity_data}\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"Features: {', '.join(feature_names)}\n")
        f.write(f"Scaling method: {args.scaling_method}\n")
        f.write(f"Number of clusters: {args.k}\n")
        f.write(f"Grid shape: {grid_shape}\n")
        f.write(f"Valid data points: {len(labels)}\n\n")
        
        f.write("Clustering Results:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Inertia: {inertia:.2e}\n")
        f.write(f"Silhouette score: {silhouette_avg:.3f}\n\n")
        
        f.write("Cluster Analysis:\n")
        f.write("-" * 16 + "\n")
        for i, (size, prop) in enumerate(zip(cluster_balance['sizes'], cluster_balance['proportions'])):
            f.write(f"Cluster {i}: {size} points ({prop:.1%})\n")
        f.write(f"Balance ratio: {cluster_balance['balance_ratio']:.3f}\n\n")
        
        f.write("Cluster Centroids (Standardized):\n")
        f.write("-" * 35 + "\n")
        f.write("Feature".ljust(15))
        for i in range(args.k):
            f.write(f"Cluster {i}".ljust(12))
        f.write("\n")
        f.write("-" * (15 + 12 * args.k) + "\n")
        
        for j, feature in enumerate(feature_names):
            f.write(f"{feature[:14]:15s}")
            for i in range(args.k):
                f.write(f"{centroids[i, j]:11.3f} ")
            f.write("\n")
        
        # Physical interpretation for ice shelf regimes
        f.write("\nPhysical Interpretation:\n")
        f.write("-" * 23 + "\n")
        
        if args.k == 3 and 'dudx' in feature_names:
            # Analyze strain rate patterns
            dudx_idx = feature_names.index('dudx')
            strain_centroids = centroids[:, dudx_idx]
            
            # Sort clusters by strain rate
            sorted_indices = np.argsort(strain_centroids)
            
            f.write("Based on longitudinal strain rate patterns:\n")
            f.write(f"Cluster {sorted_indices[0]}: Likely COMPRESSION regime (ε_xx = {strain_centroids[sorted_indices[0]]:.2e})\n")
            f.write(f"Cluster {sorted_indices[1]}: Likely TRANSITION regime (ε_xx = {strain_centroids[sorted_indices[1]]:.2e})\n")
            f.write(f"Cluster {sorted_indices[2]}: Likely EXTENSION regime (ε_xx = {strain_centroids[sorted_indices[2]]:.2e})\n\n")
            
            f.write("Regime characteristics:\n")
            f.write("- Compression: Negative strain rate, near grounding line\n")
            f.write("- Transition: Low strain rate, rheological boundary\n")
            f.write("- Extension: Positive strain rate, toward calving front\n")
    
    print(f"Text summary saved to: {text_summary_file}")
    print()
    print("Clustering analysis completed successfully!")
    
    # Print final summary
    print("Final Results Summary:")
    print("-" * 20)
    print(f"Successfully classified ice shelf into {args.k} clusters")
    print(f"Silhouette score: {silhouette_avg:.3f} ({'Good' if silhouette_avg > 0.5 else 'Moderate' if silhouette_avg > 0.25 else 'Poor'})")
    print(f"Cluster balance: {cluster_balance['balance_ratio']:.2f} ({'Well-balanced' if cluster_balance['balance_ratio'] > 0.3 else 'Imbalanced'})")
    
    if args.k == 3:
        print("\nPhysical interpretation:")
        if 'dudx' in feature_names:
            dudx_idx = feature_names.index('dudx')
            strain_centroids = centroids[:, dudx_idx]
            sorted_indices = np.argsort(strain_centroids)
            print(f"  Cluster {sorted_indices[0]}: Compression regime")
            print(f"  Cluster {sorted_indices[1]}: Transition regime")  
            print(f"  Cluster {sorted_indices[2]}: Extension regime")


if __name__ == '__main__':
    main()