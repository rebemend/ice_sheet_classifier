import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings


class KMeansRunner:
    """
    K-means clustering runner for ice shelf regime classification.
    
    Provides consistent interface for running k-means with different
    parameters and collecting diagnostics.
    """
    
    def __init__(self, random_state: int = 42, max_iter: int = 300):
        """
        Initialize k-means runner.
        
        Parameters
        ----------
        random_state : int
            Random state for reproducibility
        max_iter : int
            Maximum iterations for k-means
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.results = {}
    
    def run_single_kmeans(self, features: np.ndarray, n_clusters: int,
                         init: str = 'k-means++') -> Dict:
        """
        Run k-means clustering for a single k value.
        
        Parameters
        ----------
        features : np.ndarray
            Scaled feature array (n_samples, n_features)
        n_clusters : int
            Number of clusters
        init : str
            Initialization method
            
        Returns
        -------
        Dict
            Clustering results with labels, centroids, and metrics
        """
        if features.shape[0] < n_clusters:
            raise ValueError(f"Not enough samples ({features.shape[0]}) for {n_clusters} clusters")
        
        # Run k-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10  # Multiple runs for stability
        )
        
        labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        # Compute silhouette score
        if n_clusters > 1 and features.shape[0] > n_clusters:
            try:
                silhouette_avg = silhouette_score(features, labels)
                silhouette_samples_scores = silhouette_samples(features, labels)
            except Exception as e:
                warnings.warn(f"Failed to compute silhouette score: {e}")
                silhouette_avg = np.nan
                silhouette_samples_scores = np.full(len(labels), np.nan)
        else:
            silhouette_avg = np.nan
            silhouette_samples_scores = np.full(len(labels), np.nan)
        
        # Compute cluster sizes
        cluster_sizes = np.bincount(labels)
        
        # Compute within-cluster sum of squares for each cluster
        wcss_per_cluster = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                cluster_points = features[cluster_mask]
                wcss_per_cluster[i] = np.sum((cluster_points - centroids[i])**2)
        
        return {
            'n_clusters': n_clusters,
            'labels': labels,
            'centroids': centroids,
            'inertia': inertia,
            'silhouette_avg': silhouette_avg,
            'silhouette_samples': silhouette_samples_scores,
            'cluster_sizes': cluster_sizes,
            'wcss_per_cluster': wcss_per_cluster,
            'converged': kmeans.n_iter_ < self.max_iter,
            'n_iter': kmeans.n_iter_
        }
    
    def run_k_range(self, features: np.ndarray, k_range: List[int]) -> Dict[int, Dict]:
        """
        Run k-means for a range of k values.
        
        Parameters
        ----------
        features : np.ndarray
            Scaled feature array
        k_range : List[int]
            List of k values to test
            
        Returns
        -------
        Dict[int, Dict]
            Results for each k value
        """
        results = {}
        
        for k in k_range:
            try:
                result = self.run_single_kmeans(features, k)
                results[k] = result
                print(f"k={k}: inertia={result['inertia']:.2e}, "
                      f"silhouette={result['silhouette_avg']:.3f}")
            except Exception as e:
                warnings.warn(f"Failed to run k-means for k={k}: {e}")
                results[k] = {
                    'n_clusters': k,
                    'error': str(e),
                    'inertia': np.inf,
                    'silhouette_avg': np.nan
                }
        
        self.results.update(results)
        return results
    
    def get_metrics_summary(self, results: Dict[int, Dict]) -> Dict[str, np.ndarray]:
        """
        Extract metrics summary from k-range results.
        
        Parameters
        ----------
        results : Dict[int, Dict]
            Results from run_k_range
            
        Returns
        -------
        Dict[str, np.ndarray]
            Arrays of metrics indexed by k
        """
        k_values = sorted(results.keys())
        
        inertias = []
        silhouettes = []
        converged = []
        
        for k in k_values:
            result = results[k]
            inertias.append(result.get('inertia', np.inf))
            silhouettes.append(result.get('silhouette_avg', np.nan))
            converged.append(result.get('converged', False))
        
        return {
            'k_values': np.array(k_values),
            'inertias': np.array(inertias),
            'silhouettes': np.array(silhouettes),
            'converged': np.array(converged)
        }
    
    def find_optimal_k(self, results: Dict[int, Dict],
                      method: str = 'combined') -> Tuple[int, Dict]:
        """
        Find optimal k using specified method.
        
        Parameters
        ----------
        results : Dict[int, Dict]
            Results from k-range analysis
        method : str
            Method for k selection: 'elbow', 'silhouette', 'combined'
            
        Returns
        -------
        Tuple[int, Dict]
            Optimal k and selection details
        """
        metrics = self.get_metrics_summary(results)
        k_values = metrics['k_values']
        inertias = metrics['inertias']
        silhouettes = metrics['silhouettes']
        
        selection_info = {'method': method}
        
        if method == 'elbow':
            # Find elbow in inertia curve using second derivative
            optimal_k = self._find_elbow_point(k_values, inertias)
            selection_info['elbow_k'] = optimal_k
            
        elif method == 'silhouette':
            # Find k with maximum silhouette score
            valid_silhouettes = ~np.isnan(silhouettes)
            if np.any(valid_silhouettes):
                max_idx = np.argmax(silhouettes[valid_silhouettes])
                optimal_k = k_values[valid_silhouettes][max_idx]
                selection_info['silhouette_k'] = optimal_k
                selection_info['max_silhouette'] = silhouettes[valid_silhouettes][max_idx]
            else:
                optimal_k = k_values[0]
                warnings.warn("No valid silhouette scores, using minimum k")
        
        elif method == 'combined':
            # Combine elbow and silhouette methods
            elbow_k = self._find_elbow_point(k_values, inertias)
            
            valid_silhouettes = ~np.isnan(silhouettes)
            if np.any(valid_silhouettes):
                silhouette_k = k_values[np.argmax(silhouettes[valid_silhouettes])]
            else:
                silhouette_k = elbow_k
            
            # Prefer silhouette if it's reasonable, otherwise use elbow
            if abs(silhouette_k - elbow_k) <= 1:
                optimal_k = silhouette_k
            else:
                # Choose based on which gives better silhouette score
                elbow_sil = silhouettes[k_values == elbow_k][0] if elbow_k in k_values else np.nan
                sil_sil = silhouettes[k_values == silhouette_k][0] if silhouette_k in k_values else np.nan
                
                if np.isnan(elbow_sil) and not np.isnan(sil_sil):
                    optimal_k = silhouette_k
                elif not np.isnan(elbow_sil) and np.isnan(sil_sil):
                    optimal_k = elbow_k
                elif not np.isnan(elbow_sil) and not np.isnan(sil_sil):
                    optimal_k = silhouette_k if sil_sil > elbow_sil else elbow_k
                else:
                    optimal_k = elbow_k
            
            selection_info['elbow_k'] = elbow_k
            selection_info['silhouette_k'] = silhouette_k
            selection_info['chosen_k'] = optimal_k
        
        else:
            raise ValueError(f"Unknown k selection method: {method}")
        
        return optimal_k, selection_info
    
    def _find_elbow_point(self, k_values: np.ndarray, inertias: np.ndarray) -> int:
        """
        Find elbow point in inertia curve using second derivative method.
        
        Parameters
        ----------
        k_values : np.ndarray
            K values
        inertias : np.ndarray
            Corresponding inertias
            
        Returns
        -------
        int
            K value at elbow point
        """
        if len(k_values) < 3:
            return k_values[0] if len(k_values) > 0 else 2
        
        # Compute second derivative of inertia curve
        # Use finite differences: d2y/dx2 ≈ (y[i+1] - 2*y[i] + y[i-1]) / dx^2
        second_derivs = []
        
        for i in range(1, len(inertias) - 1):
            d2 = inertias[i+1] - 2*inertias[i] + inertias[i-1]
            second_derivs.append(d2)
        
        if not second_derivs:
            return k_values[0]
        
        # Find point where second derivative is maximum (steepest change in slope)
        max_d2_idx = np.argmax(second_derivs)
        elbow_k = k_values[max_d2_idx + 1]  # +1 because we started from index 1
        
        return elbow_k
    
    def analyze_cluster_quality(self, result: Dict, feature_names: List[str]) -> Dict:
        """
        Analyze quality of clustering result.
        
        Parameters
        ----------
        result : Dict
            Single k-means result
        feature_names : List[str]
            Names of features
            
        Returns
        -------
        Dict
            Quality analysis
        """
        labels = result['labels']
        centroids = result['centroids']
        n_clusters = result['n_clusters']
        
        analysis = {
            'n_clusters': n_clusters,
            'cluster_balance': {},
            'centroid_analysis': {},
            'separation': {}
        }
        
        # Cluster balance analysis
        cluster_sizes = result['cluster_sizes']
        total_samples = len(labels)
        
        analysis['cluster_balance'] = {
            'sizes': cluster_sizes.tolist(),
            'proportions': (cluster_sizes / total_samples).tolist(),
            'min_size': np.min(cluster_sizes),
            'max_size': np.max(cluster_sizes),
            'balance_ratio': np.min(cluster_sizes) / np.max(cluster_sizes) if np.max(cluster_sizes) > 0 else 0
        }
        
        # Centroid analysis
        centroid_analysis = {}
        for i, feature_name in enumerate(feature_names):
            feature_centroids = centroids[:, i]
            centroid_analysis[feature_name] = {
                'centroids': feature_centroids.tolist(),
                'range': float(np.ptp(feature_centroids)),
                'mean': float(np.mean(feature_centroids)),
                'std': float(np.std(feature_centroids))
            }
        
        analysis['centroid_analysis'] = centroid_analysis
        
        # Cluster separation
        if n_clusters > 1:
            # Compute pairwise distances between centroids
            centroid_distances = []
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    centroid_distances.append(dist)
            
            analysis['separation'] = {
                'min_centroid_distance': float(np.min(centroid_distances)),
                'max_centroid_distance': float(np.max(centroid_distances)),
                'mean_centroid_distance': float(np.mean(centroid_distances)),
                'silhouette_avg': float(result.get('silhouette_avg', np.nan))
            }
        
        return analysis


def run_kmeans_analysis(features: np.ndarray, 
                       feature_names: List[str],
                       k_range: Optional[List[int]] = None,
                       method: str = 'combined') -> Dict:
    """
    Complete k-means analysis workflow.
    
    Parameters
    ----------
    features : np.ndarray
        Scaled feature array
    feature_names : List[str]
        Feature names
    k_range : Optional[List[int]]
        K values to test (default: 2-8)
    method : str
        K selection method
        
    Returns
    -------
    Dict
        Complete analysis results
    """
    if k_range is None:
        k_range = list(range(2, min(9, features.shape[0])))
    
    # Initialize runner
    runner = KMeansRunner()
    
    # Run k-range analysis
    print(f"Running k-means analysis for k = {k_range}")
    k_results = runner.run_k_range(features, k_range)
    
    # Find optimal k
    optimal_k, selection_info = runner.find_optimal_k(k_results, method)
    
    print(f"Optimal k selected: {optimal_k} (method: {method})")
    
    # Get detailed analysis for optimal k
    if optimal_k in k_results:
        optimal_result = k_results[optimal_k]
        quality_analysis = runner.analyze_cluster_quality(optimal_result, feature_names)
    else:
        warnings.warn(f"Optimal k={optimal_k} not in results, using k={k_range[0]}")
        optimal_k = k_range[0]
        optimal_result = k_results[optimal_k]
        quality_analysis = runner.analyze_cluster_quality(optimal_result, feature_names)
    
    # Compile final results
    analysis_results = {
        'k_range_results': k_results,
        'optimal_k': optimal_k,
        'selection_info': selection_info,
        'optimal_result': optimal_result,
        'quality_analysis': quality_analysis,
        'metrics_summary': runner.get_metrics_summary(k_results)
    }
    
    return analysis_results