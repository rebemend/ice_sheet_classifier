import numpy as np
import pytest
import sys
import os
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.kmeans_runner import KMeansRunner, run_kmeans_analysis
from clustering.k_selection import KSelectionAnalyzer
from utils.scaling import FeatureScaler, scale_features_for_clustering


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    np.random.seed(42)
    
    # Create 3 well-separated clusters in 2D
    n_samples_per_cluster = 30
    
    # Cluster 1: around (0, 0)
    cluster1 = np.random.normal([0, 0], 0.5, (n_samples_per_cluster, 2))
    
    # Cluster 2: around (3, 3) 
    cluster2 = np.random.normal([3, 3], 0.5, (n_samples_per_cluster, 2))
    
    # Cluster 3: around (-2, 4)
    cluster3 = np.random.normal([-2, 4], 0.5, (n_samples_per_cluster, 2))
    
    features = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0] * n_samples_per_cluster + 
                          [1] * n_samples_per_cluster + 
                          [2] * n_samples_per_cluster)
    
    return features, true_labels


@pytest.fixture
def sample_ice_features():
    """Create sample ice shelf features."""
    np.random.seed(123)
    n_samples = 100
    
    # Simulate ice shelf features
    dudx = np.random.normal(0, 1e-6, n_samples)  # Strain rate
    speed = np.random.uniform(50, 300, n_samples)  # Velocity magnitude
    mu = np.random.lognormal(32, 0.5, n_samples)  # Horizontal viscosity
    anisotropy = np.random.uniform(1, 5, n_samples)  # Anisotropy ratio
    
    features = np.column_stack([dudx, speed, mu, anisotropy])
    feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
    
    return features, feature_names


def test_feature_scaler_standard():
    """Test standard scaling functionality."""
    # Create test data with different scales
    data = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]])
    
    scaler = FeatureScaler(method='standard')
    scaled_data = scaler.fit_transform(data)
    
    # Check standardization worked
    assert np.allclose(np.mean(scaled_data, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(scaled_data, axis=0), 1, atol=1e-10)
    
    # Check inverse transform
    recovered_data = scaler.inverse_transform(scaled_data)
    np.testing.assert_allclose(data, recovered_data, rtol=1e-10)


def test_feature_scaler_invalid_handling():
    """Test handling of invalid values in features."""
    # Data with NaN and inf values
    data = np.array([[1, 2], [np.nan, 3], [4, np.inf], [5, 6]])
    
    # Test removal of invalid values
    scaler_remove = FeatureScaler(method='standard', handle_invalid='remove')
    scaled_data = scaler_remove.fit_transform(data)
    
    # Should have only 2 valid samples
    assert scaled_data.shape[0] == 2
    assert np.all(np.isfinite(scaled_data))
    
    # Test imputation
    scaler_impute = FeatureScaler(method='standard', handle_invalid='impute')
    scaled_imputed = scaler_impute.fit_transform(data)
    
    # Should have all 4 samples
    assert scaled_imputed.shape[0] == 4
    assert np.all(np.isfinite(scaled_imputed))


def test_kmeans_runner_single(sample_features):
    """Test single k-means run."""
    features, true_labels = sample_features
    
    # Scale features
    scaler = FeatureScaler(method='standard')
    features_scaled = scaler.fit_transform(features)
    
    runner = KMeansRunner(random_state=42)
    result = runner.run_single_kmeans(features_scaled, n_clusters=3)
    
    # Check result structure
    assert 'labels' in result
    assert 'centroids' in result
    assert 'inertia' in result
    assert 'silhouette_avg' in result
    assert 'cluster_sizes' in result
    
    # Check result validity
    assert len(result['labels']) == len(features)
    assert result['centroids'].shape == (3, 2)  # 3 clusters, 2 features
    assert result['inertia'] > 0
    assert len(result['cluster_sizes']) == 3
    
    # Should get reasonable silhouette score for well-separated clusters
    assert result['silhouette_avg'] > 0.3


def test_kmeans_runner_k_range(sample_features):
    """Test k-means over range of k values."""
    features, _ = sample_features
    
    # Scale features
    scaler = FeatureScaler(method='standard')
    features_scaled = scaler.fit_transform(features)
    
    runner = KMeansRunner(random_state=42)
    results = runner.run_k_range(features_scaled, k_range=[2, 3, 4, 5])
    
    # Check all k values were tested
    assert set(results.keys()) == {2, 3, 4, 5}
    
    # Check each result is valid
    for k, result in results.items():
        assert result['n_clusters'] == k
        assert 'inertia' in result
        assert 'silhouette_avg' in result
    
    # Check metrics summary
    metrics = runner.get_metrics_summary(results)
    assert 'k_values' in metrics
    assert 'inertias' in metrics
    assert 'silhouettes' in metrics
    
    # Inertia should generally decrease with k
    assert np.all(np.diff(metrics['inertias']) <= 0)


def test_kmeans_runner_optimal_k(sample_features):
    """Test optimal k selection."""
    features, _ = sample_features
    
    scaler = FeatureScaler(method='standard') 
    features_scaled = scaler.fit_transform(features)
    
    runner = KMeansRunner(random_state=42)
    results = runner.run_k_range(features_scaled, k_range=[2, 3, 4, 5])
    
    # Test different k selection methods
    for method in ['elbow', 'silhouette', 'combined']:
        optimal_k, selection_info = runner.find_optimal_k(results, method)
        
        assert isinstance(optimal_k, (int, np.integer))
        assert optimal_k in [2, 3, 4, 5]
        assert 'method' in selection_info
        assert selection_info['method'] == method


def test_k_selection_analyzer(sample_features):
    """Test comprehensive k selection analysis."""
    features, _ = sample_features
    
    # Scale features
    scaler = FeatureScaler(method='standard')
    features_scaled = scaler.fit_transform(features)
    
    # Run k-means for range
    runner = KMeansRunner(random_state=42)
    k_results = runner.run_k_range(features_scaled, k_range=[2, 3, 4])
    
    # Run comprehensive analysis
    analyzer = KSelectionAnalyzer()
    analysis = analyzer.comprehensive_k_analysis(features_scaled, k_results)
    
    # Check analysis components
    assert 'elbow' in analysis
    assert 'silhouette' in analysis
    assert 'consensus' in analysis
    
    # Check elbow analysis
    elbow = analysis['elbow']
    assert 'recommended_k' in elbow
    assert 'curvature_k' in elbow
    
    # Check silhouette analysis
    silhouette = analysis['silhouette']
    assert 'best_k' in silhouette
    assert 'max_silhouette' in silhouette
    
    # Check consensus
    consensus = analysis['consensus']
    assert 'consensus_k' in consensus
    assert 'confidence_level' in consensus


def test_scale_features_for_clustering(sample_ice_features):
    """Test complete feature scaling workflow."""
    features, feature_names = sample_ice_features
    
    # Test scaling
    features_scaled, scaler, valid_mask = scale_features_for_clustering(
        features, feature_names, method='standard'
    )
    
    # Check results
    assert features_scaled.shape[1] == features.shape[1]  # Same number of features
    assert len(valid_mask) == features.shape[0]  # Valid mask for all samples
    assert scaler.is_fitted
    
    # Check standardization
    if features_scaled.shape[0] > 1:
        assert np.allclose(np.mean(features_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(features_scaled, axis=0), 1, atol=1e-10)


def test_run_kmeans_analysis(sample_ice_features):
    """Test complete k-means analysis workflow."""
    features, feature_names = sample_ice_features
    
    # Scale features first
    scaler = FeatureScaler(method='standard')
    features_scaled = scaler.fit_transform(features)
    
    # Run complete analysis
    results = run_kmeans_analysis(
        features_scaled, feature_names, k_range=[2, 3, 4], method='combined'
    )
    
    # Check result structure
    assert 'k_range_results' in results
    assert 'optimal_k' in results
    assert 'selection_info' in results
    assert 'optimal_result' in results
    assert 'quality_analysis' in results
    assert 'metrics_summary' in results
    
    # Check optimal k is reasonable
    assert 2 <= results['optimal_k'] <= 4
    
    # Check quality analysis
    quality = results['quality_analysis']
    assert 'cluster_balance' in quality
    assert 'centroid_analysis' in quality
    assert 'separation' in quality


def test_kmeans_error_handling():
    """Test error handling in k-means clustering."""
    # Test insufficient samples
    small_features = np.array([[1, 2], [3, 4]])  # Only 2 samples
    
    runner = KMeansRunner()
    
    with pytest.raises(ValueError, match="Not enough samples"):
        runner.run_single_kmeans(small_features, n_clusters=5)


def test_feature_scaler_edge_cases():
    """Test edge cases for feature scaler."""
    # Test constant features
    constant_features = np.array([[1, 1], [1, 1], [1, 1]])
    
    scaler = FeatureScaler(method='standard')
    
    # Should handle constant features gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore sklearn warnings about constant features
        scaled = scaler.fit_transform(constant_features)
        
    # Check that scaling completed
    assert scaled.shape == constant_features.shape


def test_scaler_no_scaling():
    """Test scaler with no scaling applied."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    
    scaler = FeatureScaler(method='none')
    scaled_data = scaler.fit_transform(data)
    
    # Data should be unchanged
    np.testing.assert_array_equal(data, scaled_data)


if __name__ == '__main__':
    pytest.main([__file__])