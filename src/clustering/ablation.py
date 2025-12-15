import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from .kmeans_runner import KMeansRunner
try:
    from ..features.feature_sets import FeatureSetDefinitions
except ImportError:
    # When running as script, relative imports may fail
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from features.feature_sets import FeatureSetDefinitions


class FeatureAblationAnalyzer:
    """
    Feature ablation analysis for understanding feature importance in clustering.
    
    Systematically removes features to assess their contribution to
    cluster quality and interpretability.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ablation analyzer.
        
        Parameters
        ----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.baseline_results = None
        self.ablation_results = {}
    
    def run_baseline_clustering(self, features: np.ndarray, feature_names: List[str],
                              k: int) -> Dict:
        """
        Run baseline clustering with all features.
        
        Parameters
        ----------
        features : np.ndarray
            Full feature array (n_samples, n_features)
        feature_names : List[str]
            Names of all features
        k : int
            Number of clusters to use
            
        Returns
        -------
        Dict
            Baseline clustering results
        """
        runner = KMeansRunner(random_state=self.random_state)
        
        print(f"Running baseline clustering with all {len(feature_names)} features...")
        baseline_result = runner.run_single_kmeans(features, k)
        
        # Store baseline results
        self.baseline_results = {
            'result': baseline_result,
            'features': features,
            'feature_names': feature_names,
            'k': k
        }
        
        print(f"Baseline - Inertia: {baseline_result['inertia']:.2e}, "
              f"Silhouette: {baseline_result['silhouette_avg']:.3f}")
        
        return baseline_result
    
    def run_single_feature_ablation(self, exclude_feature: str) -> Dict:
        """
        Run clustering with one feature removed.
        
        Parameters
        ----------
        exclude_feature : str
            Name of feature to exclude
            
        Returns
        -------
        Dict
            Ablation results for this feature
        """
        if self.baseline_results is None:
            raise ValueError("Must run baseline clustering first")
        
        baseline_features = self.baseline_results['features']
        baseline_names = self.baseline_results['feature_names']
        k = self.baseline_results['k']
        
        if exclude_feature not in baseline_names:
            raise ValueError(f"Feature '{exclude_feature}' not in baseline features")
        
        # Create feature subset (all except excluded feature)
        feature_idx = baseline_names.index(exclude_feature)
        feature_indices = [i for i in range(len(baseline_names)) if i != feature_idx]
        
        ablated_features = baseline_features[:, feature_indices]
        ablated_names = [name for name in baseline_names if name != exclude_feature]
        
        # Run clustering on ablated features
        runner = KMeansRunner(random_state=self.random_state)
        ablation_result = runner.run_single_kmeans(ablated_features, k)
        
        # Compute metrics relative to baseline
        baseline_result = self.baseline_results['result']
        
        metrics = {
            'excluded_feature': exclude_feature,
            'remaining_features': ablated_names,
            'n_remaining_features': len(ablated_names),
            'clustering_result': ablation_result,
            'inertia_change': ablation_result['inertia'] - baseline_result['inertia'],
            'inertia_pct_change': ((ablation_result['inertia'] - baseline_result['inertia']) / 
                                 baseline_result['inertia'] * 100),
            'silhouette_change': (ablation_result['silhouette_avg'] - baseline_result['silhouette_avg']),
            'silhouette_pct_change': None
        }
        
        # Compute silhouette percentage change (handle NaN case)
        baseline_sil = baseline_result['silhouette_avg']
        if not np.isnan(baseline_sil) and baseline_sil != 0:
            metrics['silhouette_pct_change'] = (metrics['silhouette_change'] / baseline_sil * 100)
        
        return metrics
    
    def run_comprehensive_ablation(self) -> Dict[str, Dict]:
        """
        Run ablation analysis for all features.
        
        Returns
        -------
        Dict[str, Dict]
            Ablation results for each feature
        """
        if self.baseline_results is None:
            raise ValueError("Must run baseline clustering first")
        
        feature_names = self.baseline_results['feature_names']
        ablation_results = {}
        
        print(f"Running ablation analysis for {len(feature_names)} features...")
        
        for feature_name in feature_names:
            print(f"  Ablating feature: {feature_name}")
            try:
                result = self.run_single_feature_ablation(feature_name)
                ablation_results[feature_name] = result
                
                print(f"    Inertia change: {result['inertia_pct_change']:+.1f}%, "
                      f"Silhouette change: {result['silhouette_change']:+.3f}")
                
            except Exception as e:
                warnings.warn(f"Ablation failed for feature '{feature_name}': {e}")
                ablation_results[feature_name] = {'error': str(e)}
        
        self.ablation_results = ablation_results
        return ablation_results
    
    def analyze_feature_importance(self, ablation_results: Optional[Dict] = None) -> Dict:
        """
        Analyze feature importance from ablation results.
        
        Parameters
        ----------
        ablation_results : Optional[Dict]
            Ablation results (uses stored results if None)
            
        Returns
        -------
        Dict
            Feature importance analysis
        """
        if ablation_results is None:
            ablation_results = self.ablation_results
        
        if not ablation_results:
            raise ValueError("No ablation results available")
        
        # Extract metrics for valid results
        valid_results = {k: v for k, v in ablation_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid ablation results'}
        
        # Compile importance metrics
        feature_names = list(valid_results.keys())
        inertia_changes = [valid_results[f]['inertia_pct_change'] for f in feature_names]
        silhouette_changes = [valid_results[f]['silhouette_change'] for f in feature_names]
        
        # Feature importance ranking
        # More positive inertia change = more important (worse clustering without it)
        # More negative silhouette change = more important (worse clustering without it)
        
        importance_scores = []
        for i, feature in enumerate(feature_names):
            # Combine metrics (higher score = more important)
            inertia_score = inertia_changes[i]  # Higher = more important
            silhouette_score = -silhouette_changes[i] * 100  # Convert to percentage, flip sign
            
            # Weight combination (can be adjusted)
            combined_score = 0.6 * inertia_score + 0.4 * silhouette_score
            importance_scores.append(combined_score)
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
        
        importance_ranking = []
        for idx in sorted_indices:
            feature = feature_names[idx]
            importance_ranking.append({
                'feature': feature,
                'importance_score': importance_scores[idx],
                'inertia_pct_change': inertia_changes[idx],
                'silhouette_change': silhouette_changes[idx],
                'rank': len(importance_ranking) + 1
            })
        
        # Categorize features by importance
        most_important = importance_ranking[0]['feature'] if importance_ranking else None
        least_important = importance_ranking[-1]['feature'] if importance_ranking else None
        
        # Features that significantly degrade clustering when removed
        significant_threshold = 5.0  # 5% inertia increase or significant silhouette drop
        critical_features = []
        
        for item in importance_ranking:
            if (item['inertia_pct_change'] > significant_threshold or 
                item['silhouette_change'] < -0.1):
                critical_features.append(item['feature'])
        
        return {
            'importance_ranking': importance_ranking,
            'most_important_feature': most_important,
            'least_important_feature': least_important,
            'critical_features': critical_features,
            'summary_stats': {
                'mean_inertia_change': float(np.mean(inertia_changes)),
                'std_inertia_change': float(np.std(inertia_changes)),
                'mean_silhouette_change': float(np.mean(silhouette_changes)),
                'std_silhouette_change': float(np.std(silhouette_changes))
            }
        }
    
    def run_feature_set_ablation(self, features_dict: Dict[str, np.ndarray],
                                feature_sets: Dict[str, List[str]], k: int) -> Dict:
        """
        Run ablation analysis comparing different feature sets.
        
        Parameters
        ----------
        features_dict : Dict[str, np.ndarray]
            Dictionary mapping feature names to feature arrays
        feature_sets : Dict[str, List[str]]
            Dictionary mapping set names to lists of feature names
        k : int
            Number of clusters
            
        Returns
        -------
        Dict
            Feature set comparison results
        """
        runner = KMeansRunner(random_state=self.random_state)
        set_results = {}
        
        print(f"Running feature set ablation for {len(feature_sets)} sets...")
        
        for set_name, feature_names in feature_sets.items():
            print(f"  Testing feature set: {set_name}")
            
            try:
                # Build feature array for this set
                feature_arrays = []
                valid_features = []
                
                for feature_name in feature_names:
                    if feature_name in features_dict:
                        feature_arrays.append(features_dict[feature_name])
                        valid_features.append(feature_name)
                    else:
                        warnings.warn(f"Feature '{feature_name}' not available, skipping")
                
                if not feature_arrays:
                    set_results[set_name] = {'error': 'No valid features in set'}
                    continue
                
                # Stack features
                set_features = np.column_stack(feature_arrays)
                
                # Run clustering
                clustering_result = runner.run_single_kmeans(set_features, k)
                
                set_results[set_name] = {
                    'feature_names': valid_features,
                    'n_features': len(valid_features),
                    'clustering_result': clustering_result,
                    'inertia': clustering_result['inertia'],
                    'silhouette_avg': clustering_result['silhouette_avg']
                }
                
                print(f"    Features: {valid_features}")
                print(f"    Inertia: {clustering_result['inertia']:.2e}, "
                      f"Silhouette: {clustering_result['silhouette_avg']:.3f}")
                
            except Exception as e:
                warnings.warn(f"Feature set '{set_name}' failed: {e}")
                set_results[set_name] = {'error': str(e)}
        
        return set_results
    
    def recommend_feature_subset(self, importance_analysis: Dict,
                                min_features: int = 2) -> Dict:
        """
        Recommend optimal feature subset based on importance analysis.
        
        Parameters
        ----------
        importance_analysis : Dict
            Results from analyze_feature_importance
        min_features : int
            Minimum number of features to retain
            
        Returns
        -------
        Dict
            Feature subset recommendation
        """
        if 'importance_ranking' not in importance_analysis:
            return {'error': 'Invalid importance analysis'}
        
        ranking = importance_analysis['importance_ranking']
        critical_features = importance_analysis.get('critical_features', [])
        
        # Strategy 1: Keep all critical features
        recommended_features = list(critical_features)
        
        # Strategy 2: Add most important features until we have enough
        for item in ranking:
            feature = item['feature']
            if feature not in recommended_features:
                recommended_features.append(feature)
                
                # Stop when we have sufficient features and good performance
                if len(recommended_features) >= min_features:
                    # Check if adding more features provides minimal improvement
                    if len(recommended_features) >= 4:  # Arbitrary threshold
                        break
        
        # Ensure minimum number of features
        if len(recommended_features) < min_features:
            for item in ranking:
                if item['feature'] not in recommended_features:
                    recommended_features.append(item['feature'])
                    if len(recommended_features) >= min_features:
                        break
        
        return {
            'recommended_features': recommended_features,
            'n_features': len(recommended_features),
            'reasoning': {
                'critical_features_included': critical_features,
                'importance_based_selection': True,
                'min_features_constraint': min_features
            }
        }


def run_predefined_ablation_study(features: np.ndarray, feature_names: List[str],
                                 k: int = 3) -> Dict:
    """
    Run ablation study using predefined feature sets from the project.
    
    Parameters
    ----------
    features : np.ndarray
        Full feature array
    feature_names : List[str]
        Feature names
    k : int
        Number of clusters
        
    Returns
    -------
    Dict
        Complete ablation study results
    """
    analyzer = FeatureAblationAnalyzer()
    
    # Run baseline clustering
    baseline_result = analyzer.run_baseline_clustering(features, feature_names, k)
    
    # Run individual feature ablation
    individual_ablation = analyzer.run_comprehensive_ablation()
    
    # Analyze feature importance
    importance_analysis = analyzer.analyze_feature_importance(individual_ablation)
    
    # Test predefined feature sets
    ablation_sets = FeatureSetDefinitions.get_ablation_features()
    
    # Convert features to dictionary format
    features_dict = {name: features[:, i] for i, name in enumerate(feature_names)}
    
    # Run feature set ablation
    set_ablation = analyzer.run_feature_set_ablation(features_dict, ablation_sets, k)
    
    # Get feature recommendations
    recommendations = analyzer.recommend_feature_subset(importance_analysis)
    
    return {
        'baseline_result': baseline_result,
        'individual_ablation': individual_ablation,
        'importance_analysis': importance_analysis,
        'feature_set_ablation': set_ablation,
        'recommendations': recommendations,
        'summary': {
            'n_features_tested': len(feature_names),
            'n_feature_sets_tested': len(ablation_sets),
            'most_important_feature': importance_analysis.get('most_important_feature'),
            'recommended_n_features': recommendations.get('n_features', len(feature_names))
        }
    }