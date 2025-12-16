import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from scipy.optimize import curve_fit
import warnings


class KSelectionAnalyzer:
    """
    Advanced methods for selecting optimal number of clusters.
    
    Provides multiple techniques for k selection including elbow method,
    silhouette analysis, gap statistic, and information criteria.
    """
    
    def __init__(self):
        """Initialize k selection analyzer."""
        self.results = {}
    
    def compute_elbow_metrics(self, k_values: np.ndarray, inertias: np.ndarray) -> Dict:
        """
        Compute elbow method metrics.
        
        Parameters
        ----------
        k_values : np.ndarray
            K values tested
        inertias : np.ndarray
            Corresponding inertias (within-cluster sum of squares)
            
        Returns
        -------
        Dict
            Elbow analysis results
        """
        if len(k_values) < 3:
            return {'elbow_k': k_values[0] if len(k_values) > 0 else 2, 'method': 'insufficient_data'}
        
        # Method 1: Second derivative (curvature)
        curvatures = []
        for i in range(1, len(inertias) - 1):
            curvature = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            curvatures.append(curvature)
        
        curvature_k = k_values[1 + np.argmax(curvatures)] if curvatures else k_values[0]
        
        # Method 2: Rate of change of inertia
        rates = np.diff(inertias)
        rate_changes = np.diff(rates)
        
        if len(rate_changes) > 0:
            # Find where rate of change stabilizes (smallest change)
            rate_k = k_values[2 + np.argmin(np.abs(rate_changes))]
        else:
            rate_k = curvature_k
        
        # Method 3: Percentage of variance explained
        total_var = inertias[0]  # k=1 case or maximum inertia
        var_explained = 1 - (inertias / total_var)
        var_improvements = np.diff(var_explained)
        
        # Find where improvement drops below threshold
        threshold = 0.05  # 5% improvement threshold
        significant_improvements = var_improvements > threshold
        
        if np.any(significant_improvements):
            last_significant = np.where(significant_improvements)[0][-1]
            variance_k = k_values[last_significant + 1]
        else:
            variance_k = k_values[0]
        
        return {
            'curvature_k': int(curvature_k),
            'rate_change_k': int(rate_k),
            'variance_k': int(variance_k),
            'curvatures': curvatures,
            'rate_changes': rate_changes.tolist() if len(rate_changes) > 0 else [],
            'var_explained': var_explained.tolist(),
            'recommended_k': int(curvature_k)  # Default to curvature method
        }
    
    def compute_silhouette_analysis(self, k_values: np.ndarray, 
                                  silhouette_scores: np.ndarray) -> Dict:
        """
        Analyze silhouette scores for k selection.
        
        Parameters
        ----------
        k_values : np.ndarray
            K values tested
        silhouette_scores : np.ndarray
            Average silhouette scores
            
        Returns
        -------
        Dict
            Silhouette analysis results
        """
        valid_mask = ~np.isnan(silhouette_scores)
        
        if not np.any(valid_mask):
            return {
                'best_k': k_values[0] if len(k_values) > 0 else 2,
                'max_silhouette': np.nan,
                'method': 'no_valid_scores'
            }
        
        valid_k = k_values[valid_mask]
        valid_scores = silhouette_scores[valid_mask]
        
        # Find k with maximum silhouette score
        max_idx = np.argmax(valid_scores)
        best_k = valid_k[max_idx]
        max_silhouette = valid_scores[max_idx]
        
        # Find all k values within 5% of the maximum
        threshold = 0.95 * max_silhouette
        good_ks = valid_k[valid_scores >= threshold]
        
        # Additional analysis
        analysis = {
            'best_k': int(best_k),
            'max_silhouette': float(max_silhouette),
            'good_k_range': good_ks.tolist(),
            'score_range': [float(np.min(valid_scores)), float(np.max(valid_scores))],
            'score_std': float(np.std(valid_scores))
        }
        
        # Check for stability (consistent high scores)
        if len(valid_scores) > 2:
            score_diff = np.diff(valid_scores)
            analysis['stability'] = float(np.mean(np.abs(score_diff)))
        
        return analysis
    
    def compute_gap_statistic(self, features: np.ndarray, k_values: np.ndarray,
                            inertias: np.ndarray, n_refs: int = 10) -> Dict:
        """
        Compute gap statistic for k selection.
        
        The gap statistic compares the within-cluster dispersion with
        that expected under an appropriate null reference distribution.
        
        Parameters
        ----------
        features : np.ndarray
            Original feature data
        k_values : np.ndarray
            K values tested
        inertias : np.ndarray
            Observed inertias
        n_refs : int
            Number of reference datasets
            
        Returns
        -------
        Dict
            Gap statistic results
        """
        from sklearn.cluster import KMeans
        
        n_samples, n_features = features.shape
        
        # Generate reference datasets (uniform random in feature space)
        refs = []
        for _ in range(n_refs):
            ref_data = np.random.uniform(
                low=np.min(features, axis=0),
                high=np.max(features, axis=0),
                size=(n_samples, n_features)
            )
            refs.append(ref_data)
        
        # Compute expected inertias for reference data
        ref_inertias = []
        
        for k in k_values:
            k_ref_inertias = []
            for ref_data in refs:
                if ref_data.shape[0] >= k:
                    kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=1)
                    kmeans_ref.fit(ref_data)
                    k_ref_inertias.append(kmeans_ref.inertia_)
                else:
                    k_ref_inertias.append(np.inf)
            
            ref_inertias.append(np.mean(k_ref_inertias))
        
        ref_inertias = np.array(ref_inertias)
        
        # Compute gap statistics
        log_observed = np.log(inertias + 1e-10)  # Add small constant to avoid log(0)
        log_expected = np.log(ref_inertias + 1e-10)
        gaps = log_expected - log_observed
        
        # Compute standard errors
        ref_std = []
        for k_idx, k in enumerate(k_values):
            k_ref_inertias = []
            for ref_data in refs:
                if ref_data.shape[0] >= k:
                    kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=1)
                    kmeans_ref.fit(ref_data)
                    k_ref_inertias.append(np.log(kmeans_ref.inertia_ + 1e-10))
            
            if k_ref_inertias:
                std_k = np.std(k_ref_inertias) * np.sqrt(1 + 1/n_refs)
                ref_std.append(std_k)
            else:
                ref_std.append(np.inf)
        
        ref_std = np.array(ref_std)
        
        # Find optimal k using gap statistic rule
        # Choose smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
        optimal_k = k_values[0]
        
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i+1] - ref_std[i+1]:
                optimal_k = k_values[i]
                break
        
        return {
            'gaps': gaps.tolist(),
            'ref_std': ref_std.tolist(),
            'optimal_k': int(optimal_k),
            'max_gap_k': int(k_values[np.argmax(gaps)]),
            'ref_inertias': ref_inertias.tolist(),
            'observed_inertias': inertias.tolist()
        }
    
    def compute_information_criteria(self, features: np.ndarray, k_values: np.ndarray,
                                   inertias: np.ndarray) -> Dict:
        """
        Compute information criteria (AIC, BIC) for k selection.
        
        Parameters
        ----------
        features : np.ndarray
            Feature data
        k_values : np.ndarray
            K values tested  
        inertias : np.ndarray
            Inertias (within-cluster sum of squares)
            
        Returns
        -------
        Dict
            Information criteria results
        """
        n_samples, n_features = features.shape
        
        # Estimate variance from inertias
        # Assuming Gaussian clusters: log-likelihood ≈ -0.5 * inertia / sigma^2
        # We'll estimate sigma^2 from the data
        
        aics = []
        bics = []
        
        for k_idx, k in enumerate(k_values):
            # Number of parameters: k centroids * n_features + k-1 cluster probabilities
            n_params = k * n_features + (k - 1)
            
            # Estimate log-likelihood from inertia
            # This is an approximation assuming spherical Gaussian clusters
            inertia = inertias[k_idx]
            
            # Estimate variance (rough approximation)
            if inertia > 0:
                sigma_sq = inertia / (n_samples - k)
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * sigma_sq) - 0.5 * inertia / sigma_sq
            else:
                log_likelihood = 0  # Perfect clustering
            
            # Compute information criteria
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(n_samples) * n_params - 2 * log_likelihood
            
            aics.append(aic)
            bics.append(bic)
        
        aics = np.array(aics)
        bics = np.array(bics)
        
        # Find optimal k (minimum AIC/BIC)
        aic_k = k_values[np.argmin(aics)]
        bic_k = k_values[np.argmin(bics)]
        
        return {
            'aics': aics.tolist(),
            'bics': bics.tolist(),
            'aic_optimal_k': int(aic_k),
            'bic_optimal_k': int(bic_k),
            'min_aic': float(np.min(aics)),
            'min_bic': float(np.min(bics))
        }
    
    def comprehensive_k_analysis(self, features: np.ndarray, k_results: Dict[int, Dict]) -> Dict:
        """
        Perform comprehensive k selection analysis using multiple methods.
        
        Parameters
        ----------
        features : np.ndarray
            Feature data
        k_results : Dict[int, Dict]
            K-means results for different k values
            
        Returns
        -------
        Dict
            Comprehensive analysis results
        """
        # Extract metrics
        k_values = np.array(sorted(k_results.keys()))
        inertias = np.array([k_results[k].get('inertia', np.inf) for k in k_values])
        silhouettes = np.array([k_results[k].get('silhouette_avg', np.nan) for k in k_values])
        
        analysis = {}
        
        # Elbow analysis
        print("Computing elbow analysis...")
        analysis['elbow'] = self.compute_elbow_metrics(k_values, inertias)
        
        # Silhouette analysis
        print("Computing silhouette analysis...")
        analysis['silhouette'] = self.compute_silhouette_analysis(k_values, silhouettes)
        
        # Gap statistic (computationally expensive)
        if len(k_values) <= 6 and features.shape[0] <= 10000:  # Reasonable limits
            try:
                print("Computing gap statistic...")
                analysis['gap_statistic'] = self.compute_gap_statistic(features, k_values, inertias)
            except Exception as e:
                warnings.warn(f"Gap statistic computation failed: {e}")
                analysis['gap_statistic'] = {'error': str(e)}
        else:
            analysis['gap_statistic'] = {'skipped': 'data_too_large'}
        
        # Information criteria
        print("Computing information criteria...")
        try:
            analysis['information_criteria'] = self.compute_information_criteria(features, k_values, inertias)
        except Exception as e:
            warnings.warn(f"Information criteria computation failed: {e}")
            analysis['information_criteria'] = {'error': str(e)}
        
        # Consensus recommendation
        analysis['consensus'] = self._compute_consensus_k(analysis)
        
        return analysis
    
    def _compute_consensus_k(self, analysis: Dict) -> Dict:
        """
        Compute consensus k recommendation from multiple methods.
        
        Parameters
        ----------
        analysis : Dict
            Results from different k selection methods
            
        Returns
        -------
        Dict
            Consensus recommendation
        """
        recommendations = {}
        
        # Collect recommendations from each method
        if 'elbow' in analysis and 'recommended_k' in analysis['elbow']:
            recommendations['elbow'] = analysis['elbow']['recommended_k']
        
        if 'silhouette' in analysis and 'best_k' in analysis['silhouette']:
            recommendations['silhouette'] = analysis['silhouette']['best_k']
        
        if 'gap_statistic' in analysis and 'optimal_k' in analysis['gap_statistic']:
            recommendations['gap_statistic'] = analysis['gap_statistic']['optimal_k']
        
        if 'information_criteria' in analysis:
            ic = analysis['information_criteria']
            if 'aic_optimal_k' in ic:
                recommendations['aic'] = ic['aic_optimal_k']
            if 'bic_optimal_k' in ic:
                recommendations['bic'] = ic['bic_optimal_k']
        
        if not recommendations:
            return {'consensus_k': 3, 'method': 'default', 'confidence': 'none'}
        
        # Find most common recommendation
        rec_values = list(recommendations.values())
        unique_ks, counts = np.unique(rec_values, return_counts=True)
        
        # Get the most frequent k
        consensus_k = int(unique_ks[np.argmax(counts)])
        max_count = np.max(counts)
        confidence = max_count / len(rec_values)
        
        # Determine confidence level
        if confidence >= 0.75:
            confidence_level = 'high'
        elif confidence >= 0.5:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
            # If low confidence, prefer silhouette or elbow methods
            if 'silhouette' in recommendations:
                consensus_k = recommendations['silhouette']
            elif 'elbow' in recommendations:
                consensus_k = recommendations['elbow']
        
        return {
            'consensus_k': consensus_k,
            'confidence_score': confidence,
            'confidence_level': confidence_level,
            'method_recommendations': recommendations,
            'agreement_count': max_count,
            'total_methods': len(recommendations)
        }
    
    def create_k_selection_plot(self, analysis: Dict, k_results: Dict[int, Dict],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive k selection diagnostic plot.
        
        Parameters
        ----------
        analysis : Dict
            Comprehensive analysis results
        k_results : Dict[int, Dict]
            Original k-means results
        save_path : Optional[str]
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        # Extract data
        k_values = np.array(sorted(k_results.keys()))
        inertias = np.array([k_results[k].get('inertia', np.inf) for k in k_values])
        silhouettes = np.array([k_results[k].get('silhouette_avg', np.nan) for k in k_values])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Elbow curve
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Highlight elbow recommendation
        if 'elbow' in analysis and 'recommended_k' in analysis['elbow']:
            elbow_k = analysis['elbow']['recommended_k']
            if elbow_k in k_values:
                elbow_idx = np.where(k_values == elbow_k)[0][0]
                ax1.plot(elbow_k, inertias[elbow_idx], 'ro', markersize=12, alpha=0.7)
                ax1.annotate(f'Elbow: k={elbow_k}', (elbow_k, inertias[elbow_idx]),
                           xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        # Plot 2: Silhouette scores
        valid_mask = ~np.isnan(silhouettes)
        ax2.plot(k_values[valid_mask], silhouettes[valid_mask], 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Good threshold')
        
        # Highlight silhouette recommendation
        if 'silhouette' in analysis and 'best_k' in analysis['silhouette']:
            sil_k = analysis['silhouette']['best_k']
            if sil_k in k_values:
                sil_idx = np.where(k_values == sil_k)[0][0]
                ax2.plot(sil_k, silhouettes[sil_idx], 'ro', markersize=12, alpha=0.7)
                ax2.annotate(f'Best: k={sil_k}', (sil_k, silhouettes[sil_idx]),
                           xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        # Plot 3: Gap statistic (if available)
        if 'gap_statistic' in analysis and 'gaps' in analysis['gap_statistic']:
            gaps = analysis['gap_statistic']['gaps']
            ax3.plot(k_values, gaps, 'mo-', linewidth=2, markersize=8)
            ax3.set_xlabel('Number of Clusters (k)')
            ax3.set_ylabel('Gap Statistic')
            ax3.set_title('Gap Statistic')
            ax3.grid(True, alpha=0.3)
            
            # Highlight gap recommendation
            gap_k = analysis['gap_statistic']['optimal_k']
            if gap_k in k_values:
                gap_idx = np.where(k_values == gap_k)[0][0]
                ax3.plot(gap_k, gaps[gap_idx], 'ro', markersize=12, alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'Gap Statistic\nNot Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Gap Statistic')
        
        # Plot 4: Information criteria (if available)
        if 'information_criteria' in analysis and 'aics' in analysis['information_criteria']:
            ic = analysis['information_criteria']
            aics = ic['aics']
            bics = ic['bics']
            
            ax4.plot(k_values, aics, 'co-', linewidth=2, markersize=6, label='AIC')
            ax4.plot(k_values, bics, 'yo-', linewidth=2, markersize=6, label='BIC')
            ax4.set_xlabel('Number of Clusters (k)')
            ax4.set_ylabel('Information Criterion')
            ax4.set_title('Information Criteria')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Information Criteria\nNot Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Information Criteria')

        # Add consensus recommendation as suptitle
        if 'consensus' in analysis:
            consensus = analysis['consensus']
            fig.suptitle(f'K-Selection Analysis - Consensus: k={consensus["consensus_k"]} '
                        f'({consensus["confidence_level"]} confidence)', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_simple_k_selection_plot(self, analysis: Dict, k_results: Dict[int, Dict],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create simplified k selection plot with only elbow and silhouette for GUI display.
        
        Parameters
        ----------
        analysis : Dict
            Comprehensive analysis results
        k_results : Dict[int, Dict]
            Original k-means results
        save_path : Optional[str]
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure with only top two plots
        """
        # Extract data
        k_values = np.array(sorted(k_results.keys()))
        inertias = np.array([k_results[k].get('inertia', np.inf) for k in k_values])
        silhouettes = np.array([k_results[k].get('silhouette_avg', np.nan) for k in k_values])
        
        # Create figure with just two subplots (elbow and silhouette)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Elbow curve
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
        ax1.set_title('Elbow Method\nLook for the "elbow" where curve flattens', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Highlight elbow recommendation
        if 'elbow' in analysis and 'recommended_k' in analysis['elbow']:
            elbow_k = analysis['elbow']['recommended_k']
            if elbow_k in k_values:
                elbow_idx = np.where(k_values == elbow_k)[0][0]
                ax1.plot(elbow_k, inertias[elbow_idx], 'ro', markersize=12, alpha=0.7)
                ax1.annotate(f'Elbow: k={elbow_k}', (elbow_k, inertias[elbow_idx]),
                           xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Plot 2: Silhouette scores
        valid_mask = ~np.isnan(silhouettes)
        ax2.plot(k_values[valid_mask], silhouettes[valid_mask], 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis\nHigher values indicate better separation', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Good threshold')
        ax2.legend()
        
        # Highlight silhouette recommendation
        if 'silhouette' in analysis and 'best_k' in analysis['silhouette']:
            sil_k = analysis['silhouette']['best_k']
            if sil_k in k_values:
                sil_idx = np.where(k_values == sil_k)[0][0]
                ax2.plot(sil_k, silhouettes[sil_idx], 'ro', markersize=12, alpha=0.7)
                ax2.annotate(f'Best: k={sil_k}', (sil_k, silhouettes[sil_idx]),
                           xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Add consensus recommendation as suptitle
        if 'consensus' in analysis:
            consensus = analysis['consensus']
            fig.suptitle(f'K-Selection Analysis - Recommended: k={consensus["consensus_k"]} '
                        f'({consensus["confidence_level"]} confidence)', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig