#!/usr/bin/env python3
"""
K-selection analysis script for ice shelf clustering.

This script runs k-means clustering over a range of k values and
provides comprehensive analysis to select the optimal number of clusters.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.assemble_dataset import create_complete_dataset, load_processed_dataset
from features.feature_sets import create_feature_set, get_feature_set_description
from utils.scaling import scale_features_for_clustering
from clustering.kmeans_runner import KMeansRunner
from clustering.k_selection import KSelectionAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Run k-selection analysis for ice shelf clustering')
    parser.add_argument('--diffice_data', required=True, 
                       help='Path to DIFFICE Amery data directory')
    parser.add_argument('--viscosity_data', required=True,
                       help='Path to viscosity MATLAB file (results.mat)')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory for results and plots')
    parser.add_argument('--feature_set', default='primary',
                       choices=['baseline', 'primary', 'extended', 'stress'],
                       help='Feature set to use for clustering')
    parser.add_argument('--k_range', nargs=2, type=int, default=[2, 8],
                       help='Range of k values to test (min max)')
    parser.add_argument('--scaling_method', default='standard',
                       choices=['standard', 'minmax', 'robust', 'none'],
                       help='Feature scaling method')
    parser.add_argument('--processed_data', default=None,
                       help='Path to pre-processed dataset (skip data loading if provided)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Ice Shelf K-Selection Analysis")
    print("=" * 50)
    print(f"Feature set: {args.feature_set}")
    print(f"K range: {args.k_range[0]} to {args.k_range[1]}")
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
    
    print(f"Scaled features shape: {features_scaled.shape}")
    print()
    
    # Step 4: Run k-selection analysis
    k_range = list(range(args.k_range[0], args.k_range[1] + 1))
    print(f"Running k-means analysis for k = {k_range}...")
    
    # Basic k-means analysis
    runner = KMeansRunner(random_state=args.random_state)
    k_results = runner.run_k_range(features_scaled, k_range)
    
    # Find optimal k using different methods
    methods = ['elbow', 'silhouette', 'combined']
    optimal_k_results = {}
    
    for method in methods:
        optimal_k, selection_info = runner.find_optimal_k(k_results, method)
        optimal_k_results[method] = {'k': optimal_k, 'info': selection_info}
        print(f"Optimal k ({method}): {optimal_k}")
    
    print()
    
    # Step 5: Comprehensive k-selection analysis
    print("Running comprehensive k-selection analysis...")
    analyzer = KSelectionAnalyzer()
    comprehensive_analysis = analyzer.comprehensive_k_analysis(features_scaled, k_results)
    
    # Display consensus recommendation
    consensus = comprehensive_analysis['consensus']
    print(f"Consensus recommendation: k = {consensus['consensus_k']}")
    print(f"Confidence level: {consensus['confidence_level']} ({consensus['confidence_score']:.2f})")
    print(f"Method agreement: {consensus['agreement_count']}/{consensus['total_methods']}")
    print()
    
    # Step 6: Create diagnostic plots
    print("Creating diagnostic plots...")
    
    # Full comprehensive k-selection plot (all 4 plots)
    fig_full = analyzer.create_k_selection_plot(
        comprehensive_analysis, k_results,
        save_path=str(output_dir / 'k_selection_analysis_full.png')
    )
    
    # Simplified k-selection plot for GUI (only elbow + silhouette)
    fig_simple = analyzer.create_simple_k_selection_plot(
        comprehensive_analysis, k_results,
        save_path=str(output_dir / 'k_selection_analysis.png')
    )
    
    print(f"Full k-selection plot saved to: {output_dir / 'k_selection_analysis_full.png'}")
    print(f"Simplified k-selection plot (GUI) saved to: {output_dir / 'k_selection_analysis.png'}")
    
    # Step 7: Save results
    results_file = output_dir / 'k_selection_results.npz'
    
    # Prepare results for saving
    save_data = {
        'k_range': np.array(k_range),
        'k_results_inertias': np.array([k_results[k]['inertia'] for k in k_range]),
        'k_results_silhouettes': np.array([k_results[k]['silhouette_avg'] for k in k_range]),
        'optimal_k_methods': np.array([optimal_k_results[m]['k'] for m in methods]),
        'consensus_k': consensus['consensus_k'],
        'consensus_confidence': consensus['confidence_score'],
        'feature_names': feature_names,
        'features_scaled': features_scaled,
        'scaling_method': args.scaling_method
    }
    
    np.savez(results_file, **save_data)
    print(f"Results saved to: {results_file}")
    
    # Step 8: Create summary report
    summary_file = output_dir / 'k_selection_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("Ice Shelf K-Selection Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {args.diffice_data}\n")
        f.write(f"Viscosity data: {args.viscosity_data}\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"Features: {', '.join(feature_names)}\n")
        f.write(f"Scaling method: {args.scaling_method}\n")
        f.write(f"K range tested: {args.k_range[0]} to {args.k_range[1]}\n")
        f.write(f"Valid data points: {features_scaled.shape[0]}\n\n")
        
        f.write("K-Selection Results:\n")
        f.write("-" * 20 + "\n")
        for method in methods:
            result = optimal_k_results[method]
            f.write(f"{method.capitalize()} method: k = {result['k']}\n")
        
        f.write(f"\nConsensus recommendation: k = {consensus['consensus_k']}\n")
        f.write(f"Confidence: {consensus['confidence_level']} ({consensus['confidence_score']:.3f})\n")
        f.write(f"Method agreement: {consensus['agreement_count']}/{consensus['total_methods']}\n\n")
        
        f.write("Metrics by K:\n")
        f.write("-" * 15 + "\n")
        f.write("K\tInertia\t\tSilhouette\n")
        for k in k_range:
            result = k_results[k]
            f.write(f"{k}\t{result['inertia']:.2e}\t{result['silhouette_avg']:.3f}\n")
        
        # Add analysis details
        f.write("\nDetailed Analysis:\n")
        f.write("-" * 18 + "\n")
        
        if 'elbow' in comprehensive_analysis:
            elbow = comprehensive_analysis['elbow']
            f.write(f"Elbow analysis recommended k: {elbow.get('recommended_k', 'N/A')}\n")
        
        if 'silhouette' in comprehensive_analysis:
            sil = comprehensive_analysis['silhouette']
            f.write(f"Best silhouette score: {sil.get('max_silhouette', 'N/A'):.3f} at k={sil.get('best_k', 'N/A')}\n")
        
        if 'gap_statistic' in comprehensive_analysis and 'optimal_k' in comprehensive_analysis['gap_statistic']:
            gap = comprehensive_analysis['gap_statistic']
            f.write(f"Gap statistic recommended k: {gap['optimal_k']}\n")
    
    print(f"Summary report saved to: {summary_file}")
    print()
    print("K-selection analysis completed successfully!")
    print(f"Recommended k: {consensus['consensus_k']} ({consensus['confidence_level']} confidence)")


if __name__ == '__main__':
    main()