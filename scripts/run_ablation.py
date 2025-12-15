#!/usr/bin/env python3
"""
Feature ablation analysis script for ice shelf clustering.

This script runs systematic feature ablation to understand the importance
of different features for clustering quality and interpretability.
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
from features.feature_sets import create_feature_set, create_ablation_feature_sets
from utils.scaling import scale_features_for_clustering
from clustering.ablation import FeatureAblationAnalyzer, run_predefined_ablation_study


def main():
    parser = argparse.ArgumentParser(description='Run feature ablation analysis for ice shelf clustering')
    parser.add_argument('--diffice_data', required=True,
                       help='Path to DIFFICE Amery data directory')
    parser.add_argument('--viscosity_data', required=True,
                       help='Path to viscosity MATLAB file (results.mat)')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory for results')
    parser.add_argument('--feature_set', default='primary',
                       choices=['baseline', 'primary', 'extended', 'stress'],
                       help='Base feature set for ablation study')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of clusters to use for ablation')
    parser.add_argument('--scaling_method', default='standard',
                       choices=['standard', 'minmax', 'robust', 'none'],
                       help='Feature scaling method')
    parser.add_argument('--processed_data', default=None,
                       help='Path to pre-processed dataset')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--predefined_sets', action='store_true',
                       help='Run predefined feature set ablation instead of individual features')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Ice Shelf Feature Ablation Analysis")
    print("=" * 45)
    print(f"Base feature set: {args.feature_set}")
    print(f"Number of clusters: {args.k}")
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
    
    # Step 4: Run ablation analysis
    if args.predefined_sets:
        print("Running predefined feature set ablation...")
        
        # Run predefined ablation study
        ablation_results = run_predefined_ablation_study(features_scaled, feature_names, args.k)
        
        # Extract results
        individual_ablation = ablation_results['individual_ablation']
        importance_analysis = ablation_results['importance_analysis']
        set_ablation = ablation_results['feature_set_ablation']
        recommendations = ablation_results['recommendations']
        
    else:
        print("Running individual feature ablation...")
        
        # Initialize ablation analyzer
        analyzer = FeatureAblationAnalyzer(random_state=args.random_state)
        
        # Run baseline clustering
        baseline_result = analyzer.run_baseline_clustering(features_scaled, feature_names, args.k)
        
        # Run individual feature ablation
        individual_ablation = analyzer.run_comprehensive_ablation()
        
        # Analyze feature importance
        importance_analysis = analyzer.analyze_feature_importance()
        
        # Get recommendations
        recommendations = analyzer.recommend_feature_subset(importance_analysis)
        
        # No predefined set ablation in this mode
        set_ablation = None
    
    # Step 5: Display results
    print("Ablation Analysis Results:")
    print("-" * 26)
    
    if 'importance_ranking' in importance_analysis:
        ranking = importance_analysis['importance_ranking']
        
        print("Feature Importance Ranking:")
        print("Rank\tFeature\t\tImportance\tInertia Δ%\tSilhouette Δ")
        print("-" * 65)
        
        for item in ranking:
            print(f"{item['rank']}\t{item['feature'][:12]:12s}\t{item['importance_score']:8.1f}\t"
                  f"{item['inertia_pct_change']:8.1f}\t{item['silhouette_change']:10.3f}")
        
        print()
        print(f"Most important feature: {importance_analysis['most_important_feature']}")
        print(f"Least important feature: {importance_analysis['least_important_feature']}")
        
        if importance_analysis['critical_features']:
            print(f"Critical features: {', '.join(importance_analysis['critical_features'])}")
    
    print()
    
    # Step 6: Display recommendations
    if 'recommended_features' in recommendations:
        rec_features = recommendations['recommended_features']
        print(f"Recommended feature subset ({len(rec_features)} features):")
        print(f"  {', '.join(rec_features)}")
        print()
    
    # Step 7: Display predefined set results (if available)
    if set_ablation:
        print("Predefined Feature Set Comparison:")
        print("-" * 35)
        print("Set Name\t\tFeatures\tInertia\t\tSilhouette")
        print("-" * 65)
        
        for set_name, result in set_ablation.items():
            if 'error' not in result:
                n_features = result['n_features']
                inertia = result['inertia']
                silhouette = result['silhouette_avg']
                print(f"{set_name[:15]:15s}\t{n_features}\t\t{inertia:.2e}\t{silhouette:.3f}")
            else:
                print(f"{set_name[:15]:15s}\tERROR: {result['error']}")
        print()
    
    # Step 8: Save results
    results_file = output_dir / 'ablation_results.npz'
    
    # Prepare data for saving
    save_data = {
        'feature_names': feature_names,
        'k': args.k,
        'scaling_method': args.scaling_method
    }
    
    # Add individual ablation results
    if individual_ablation:
        for feature, result in individual_ablation.items():
            if 'error' not in result:
                save_data[f'ablation_{feature}_inertia_change'] = result['inertia_pct_change']
                save_data[f'ablation_{feature}_silhouette_change'] = result['silhouette_change']
    
    # Add importance ranking
    if 'importance_ranking' in importance_analysis:
        ranking = importance_analysis['importance_ranking']
        save_data['importance_scores'] = np.array([item['importance_score'] for item in ranking])
        save_data['importance_features'] = [item['feature'] for item in ranking]
        save_data['importance_ranks'] = np.array([item['rank'] for item in ranking])
    
    # Add recommendations
    if 'recommended_features' in recommendations:
        save_data['recommended_features'] = recommendations['recommended_features']
        save_data['n_recommended_features'] = len(recommendations['recommended_features'])
    
    np.savez(results_file, **save_data)
    print(f"Ablation results saved to: {results_file}")
    
    # Step 9: Create detailed JSON summary
    summary_file = output_dir / 'ablation_summary.json'
    
    summary_data = {
        'parameters': {
            'base_feature_set': args.feature_set,
            'k': args.k,
            'scaling_method': args.scaling_method,
            'random_state': args.random_state,
            'features': feature_names
        },
        'individual_ablation': individual_ablation if individual_ablation else {},
        'importance_analysis': importance_analysis if importance_analysis else {},
        'recommendations': recommendations if recommendations else {},
        'predefined_set_ablation': set_ablation if set_ablation else {}
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    summary_data = convert_numpy(summary_data)
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Detailed summary saved to: {summary_file}")
    
    # Step 10: Create text summary
    text_summary_file = output_dir / 'ablation_summary.txt'
    
    with open(text_summary_file, 'w') as f:
        f.write("Ice Shelf Feature Ablation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {args.diffice_data}\n")
        f.write(f"Viscosity data: {args.viscosity_data}\n")
        f.write(f"Base feature set: {args.feature_set}\n")
        f.write(f"Features tested: {', '.join(feature_names)}\n")
        f.write(f"Number of clusters: {args.k}\n")
        f.write(f"Scaling method: {args.scaling_method}\n")
        f.write(f"Valid data points: {features_scaled.shape[0]}\n\n")
        
        # Individual ablation results
        if individual_ablation:
            f.write("Individual Feature Ablation Results:\n")
            f.write("-" * 38 + "\n")
            f.write("Feature".ljust(15) + "Inertia Δ%".ljust(12) + "Silhouette Δ".ljust(15) + "Impact\n")
            f.write("-" * 60 + "\n")
            
            for feature, result in individual_ablation.items():
                if 'error' not in result:
                    inertia_change = result['inertia_pct_change']
                    sil_change = result['silhouette_change']
                    
                    # Determine impact level
                    if inertia_change > 10 or sil_change < -0.1:
                        impact = "High"
                    elif inertia_change > 5 or sil_change < -0.05:
                        impact = "Medium"
                    else:
                        impact = "Low"
                    
                    f.write(f"{feature[:14]:15s}{inertia_change:+8.1f}%   {sil_change:+10.3f}     {impact}\n")
                else:
                    f.write(f"{feature[:14]:15s}ERROR: {result['error']}\n")
            f.write("\n")
        
        # Importance ranking
        if 'importance_ranking' in importance_analysis:
            f.write("Feature Importance Ranking:\n")
            f.write("-" * 27 + "\n")
            ranking = importance_analysis['importance_ranking']
            
            for i, item in enumerate(ranking[:5]):  # Top 5 features
                f.write(f"{i+1}. {item['feature']} (score: {item['importance_score']:.1f})\n")
            
            f.write(f"\nMost important: {importance_analysis['most_important_feature']}\n")
            f.write(f"Least important: {importance_analysis['least_important_feature']}\n")
            
            if importance_analysis['critical_features']:
                f.write(f"Critical features: {', '.join(importance_analysis['critical_features'])}\n")
            f.write("\n")
        
        # Recommendations
        if 'recommended_features' in recommendations:
            rec_features = recommendations['recommended_features']
            f.write("Feature Subset Recommendation:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Recommended features ({len(rec_features)}): {', '.join(rec_features)}\n")
            f.write(f"Reduction from {len(feature_names)} to {len(rec_features)} features\n")
            f.write(f"({100*(1-len(rec_features)/len(feature_names)):.0f}% reduction)\n\n")
        
        # Predefined set comparison
        if set_ablation:
            f.write("Predefined Feature Set Comparison:\n")
            f.write("-" * 35 + "\n")
            
            # Sort by silhouette score
            valid_sets = {k: v for k, v in set_ablation.items() if 'error' not in v}
            if valid_sets:
                sorted_sets = sorted(valid_sets.items(), 
                                   key=lambda x: x[1]['silhouette_avg'], reverse=True)
                
                f.write("Ranked by silhouette score:\n")
                for i, (set_name, result) in enumerate(sorted_sets):
                    sil = result['silhouette_avg']
                    n_feat = result['n_features']
                    features_list = ', '.join(result['feature_names'])
                    f.write(f"{i+1}. {set_name}: {sil:.3f} ({n_feat} features: {features_list})\n")
        
        f.write("\nInterpretation and Recommendations:\n")
        f.write("-" * 35 + "\n")
        
        if 'importance_ranking' in importance_analysis:
            ranking = importance_analysis['importance_ranking']
            top_feature = ranking[0]['feature']
            worst_feature = ranking[-1]['feature']
            
            f.write(f"• The most important feature is '{top_feature}'\n")
            f.write(f"• The least important feature is '{worst_feature}'\n")
            
            if len(ranking) > 2:
                mid_performance = [item for item in ranking[1:-1] 
                                 if item['importance_score'] > 0]
                if mid_performance:
                    f.write(f"• Moderately important features: {', '.join([item['feature'] for item in mid_performance])}\n")
            
            if importance_analysis['critical_features']:
                f.write(f"• Critical features that significantly impact clustering: {', '.join(importance_analysis['critical_features'])}\n")
            
            if 'recommended_features' in recommendations:
                rec_features = recommendations['recommended_features']
                f.write(f"• For optimal performance with reduced complexity, use: {', '.join(rec_features)}\n")
    
    print(f"Text summary saved to: {text_summary_file}")
    print()
    
    # Final summary
    print("Ablation Analysis Summary:")
    print("-" * 25)
    
    if 'importance_ranking' in importance_analysis:
        ranking = importance_analysis['importance_ranking']
        print(f"Most important feature: {ranking[0]['feature']}")
        print(f"Least important feature: {ranking[-1]['feature']}")
        
        if importance_analysis['critical_features']:
            print(f"Critical features: {', '.join(importance_analysis['critical_features'])}")
    
    if 'recommended_features' in recommendations:
        rec_features = recommendations['recommended_features']
        original_count = len(feature_names)
        recommended_count = len(rec_features)
        reduction = 100 * (1 - recommended_count / original_count)
        
        print(f"Recommended subset: {recommended_count}/{original_count} features ({reduction:.0f}% reduction)")
        print(f"Optimal features: {', '.join(rec_features)}")
    
    print("\nAblation analysis completed successfully!")


if __name__ == '__main__':
    main()