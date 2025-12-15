import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.viscosity_features import (
    compute_viscosity_features, compute_viscosity_gradients,
    classify_rheology_regime, compute_stress_features,
    validate_viscosity_features, compute_viscosity_clustering_features
)


@pytest.fixture
def sample_viscosity_data():
    """Create sample viscosity data."""
    mu = np.array([[1e14, 2e14], [1.5e14, 3e14]])    # Horizontal viscosity
    eta = np.array([[5e13, 8e13], [6e13, 1e14]])     # Vertical viscosity
    h = np.array([[100, 150], [120, 180]])           # Ice thickness
    return mu, eta, h


@pytest.fixture
def sample_coordinates():
    """Create sample coordinate arrays."""
    x = np.array([[0, 1000], [0, 1000]])
    y = np.array([[0, 0], [1000, 1000]])
    return x, y


@pytest.fixture
def sample_strain_data():
    """Create sample strain rate data."""
    return {
        'epsilon_xx': np.array([[1e-6, -2e-6], [0.5e-6, 1.5e-6]]),
        'epsilon_yy': np.array([[-0.5e-6, 1e-6], [0.2e-6, -1e-6]]),
        'epsilon_xy': np.array([[0.1e-6, 0.2e-6], [-0.1e-6, 0.3e-6]]),
        'effective_strain': np.array([[1.2e-6, 2.3e-6], [0.6e-6, 1.8e-6]])
    }


def test_compute_viscosity_features_basic(sample_viscosity_data):
    """Test basic viscosity feature computation."""
    mu, eta, h = sample_viscosity_data
    
    result = compute_viscosity_features(mu, eta)
    
    # Check expected fields
    expected_fields = ['anisotropy', 'log_mu', 'log_eta', 'log_anisotropy', 
                      'viscosity_contrast', 'mean_viscosity']
    for field in expected_fields:
        assert field in result
    
    # Check anisotropy calculation
    expected_anisotropy = mu / eta
    np.testing.assert_array_almost_equal(result['anisotropy'], expected_anisotropy)
    
    # Check mean viscosity
    expected_mean = 0.5 * (mu + eta)
    np.testing.assert_array_almost_equal(result['mean_viscosity'], expected_mean)
    
    # Check viscosity contrast
    expected_contrast = (mu - eta) / (mu + eta)
    np.testing.assert_array_almost_equal(result['viscosity_contrast'], expected_contrast)


def test_compute_viscosity_features_with_thickness(sample_viscosity_data):
    """Test viscosity features with thickness normalization."""
    mu, eta, h = sample_viscosity_data
    
    result = compute_viscosity_features(mu, eta, h)
    
    # Check thickness-normalized features are present
    assert 'thickness_normalized_mu' in result
    assert 'thickness_normalized_eta' in result
    
    # Check calculations - use relative tolerance for large numbers
    expected_norm_mu = mu / h
    np.testing.assert_allclose(result['thickness_normalized_mu'], expected_norm_mu, rtol=1e-12)


def test_compute_viscosity_gradients(sample_viscosity_data, sample_coordinates):
    """Test viscosity gradient computation."""
    mu, eta, _ = sample_viscosity_data
    x, y = sample_coordinates
    
    result = compute_viscosity_gradients(mu, eta, x, y)
    
    # Check expected fields
    expected_fields = ['dmu_dx', 'dmu_dy', 'deta_dx', 'deta_dy',
                      'mu_gradient_magnitude', 'eta_gradient_magnitude', 
                      'anisotropy_gradient']
    for field in expected_fields:
        assert field in result
    
    # Check gradient magnitudes are non-negative
    assert np.all(result['mu_gradient_magnitude'] >= 0)
    assert np.all(result['eta_gradient_magnitude'] >= 0)
    assert np.all(result['anisotropy_gradient'] >= 0)


def test_classify_rheology_regime(sample_viscosity_data):
    """Test rheological regime classification."""
    mu, eta, _ = sample_viscosity_data
    anisotropy = mu / eta
    
    result = classify_rheology_regime(anisotropy, mu)
    
    # Check result is integer array
    assert result.dtype == int
    
    # Check values are in expected range (0-3)
    assert np.all((result >= 0) & (result <= 3))
    
    # Check shape matches input
    assert result.shape == anisotropy.shape


def test_classify_rheology_regime_with_threshold():
    """Test rheology classification with specific thresholds."""
    # Create controlled test data
    mu = np.array([1e14, 2e14, 1e14, 2e14])
    anisotropy = np.array([1.0, 1.0, 3.0, 3.0])  # isotropic vs anisotropic
    
    # Use median of mu as threshold (1.5e14)
    result = classify_rheology_regime(anisotropy, mu, 
                                    anisotropy_threshold=2.0,
                                    mu_threshold=1.5e14)
    
    # Expected: 
    # [0] isotropic (1.0), low mu (1e14 < 1.5e14) -> 0
    # [1] isotropic (1.0), high mu (2e14 >= 1.5e14) -> 1  
    # [2] anisotropic (3.0), low mu (1e14 < 1.5e14) -> 2
    # [3] anisotropic (3.0), high mu (2e14 >= 1.5e14) -> 3
    expected = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_compute_stress_features(sample_viscosity_data, sample_strain_data):
    """Test stress feature computation."""
    mu, eta, _ = sample_viscosity_data
    
    result = compute_stress_features(mu, eta, sample_strain_data)
    
    # Check expected fields
    expected_fields = ['deviatoric_stress', 'stress_anisotropy', 'stress_indicator',
                      'sigma_xx', 'sigma_yy', 'tau_xy']
    for field in expected_fields:
        assert field in result
    
    # Check deviatoric stress calculation
    expected_deviatoric = 2 * mu * sample_strain_data['effective_strain']
    np.testing.assert_array_almost_equal(result['deviatoric_stress'], expected_deviatoric)
    
    # Check stress components are finite
    for field in ['sigma_xx', 'sigma_yy', 'tau_xy']:
        assert np.all(np.isfinite(result[field]))


def test_validate_viscosity_features_good_data(sample_viscosity_data):
    """Test validation with good viscosity features."""
    mu, eta, h = sample_viscosity_data
    features = compute_viscosity_features(mu, eta, h)
    
    # Should not raise any warnings
    validate_viscosity_features(features)


def test_validate_viscosity_features_negative_anisotropy():
    """Test validation with invalid anisotropy."""
    bad_features = {'anisotropy': np.array([2.0, -1.0, 0.5])}
    
    with pytest.warns(UserWarning, match="non-positive values"):
        validate_viscosity_features(bad_features)


def test_validate_viscosity_features_extreme_anisotropy():
    """Test validation with extreme anisotropy values."""
    bad_features = {'anisotropy': np.array([2.0, 150.0, 0.5])}
    
    with pytest.warns(UserWarning, match="very large values"):
        validate_viscosity_features(bad_features)


def test_validate_viscosity_features_unreasonable_viscosity():
    """Test validation with unreasonable viscosity values."""
    # log(1e10) ≈ 23, which is below expected range
    bad_features = {'log_mu': np.array([30.0, 15.0, 35.0])}
    
    with pytest.warns(UserWarning, match="outside expected range"):
        validate_viscosity_features(bad_features)


def test_compute_viscosity_clustering_features_minimal(sample_viscosity_data, sample_strain_data):
    """Test minimal feature set for clustering."""
    mu, eta, _ = sample_viscosity_data
    
    result = compute_viscosity_clustering_features(mu, eta, sample_strain_data, 'minimal')
    
    # Should only have essential features
    expected_fields = ['anisotropy', 'log_mu']
    assert set(result.keys()) == set(expected_fields)


def test_compute_viscosity_clustering_features_standard(sample_viscosity_data, sample_strain_data):
    """Test standard feature set for clustering."""
    mu, eta, _ = sample_viscosity_data
    
    result = compute_viscosity_clustering_features(mu, eta, sample_strain_data, 'standard')
    
    # Should have standard features
    expected_fields = ['anisotropy', 'log_mu', 'viscosity_contrast', 'mean_viscosity']
    assert set(result.keys()) == set(expected_fields)


def test_compute_viscosity_clustering_features_extended(sample_viscosity_data, sample_strain_data):
    """Test extended feature set for clustering."""
    mu, eta, _ = sample_viscosity_data
    
    result = compute_viscosity_clustering_features(mu, eta, sample_strain_data, 'extended')
    
    # Should include stress features
    assert 'deviatoric_stress' in result
    assert 'stress_anisotropy' in result
    assert 'anisotropy' in result


def test_compute_viscosity_clustering_features_unknown_set(sample_viscosity_data, sample_strain_data):
    """Test error handling for unknown feature set."""
    mu, eta, _ = sample_viscosity_data
    
    with pytest.raises(ValueError, match="Unknown feature set"):
        compute_viscosity_clustering_features(mu, eta, sample_strain_data, 'unknown')


def test_log_transformations_avoid_zero():
    """Test that log transformations handle zero and negative values."""
    mu = np.array([1e14, 0, -1e13])  # Include problematic values
    eta = np.array([1e13, 1e13, 1e13])
    
    result = compute_viscosity_features(mu, eta)
    
    # All log values should be finite
    assert np.all(np.isfinite(result['log_mu']))
    assert np.all(np.isfinite(result['log_eta']))
    assert np.all(np.isfinite(result['log_anisotropy']))


def test_division_by_zero_protection():
    """Test protection against division by zero."""
    mu = np.array([1e14, 2e14])
    eta = np.array([1e13, 0])  # Zero in denominator
    
    result = compute_viscosity_features(mu, eta)
    
    # Should not have inf values due to epsilon protection
    assert np.all(np.isfinite(result['anisotropy']))
    assert np.all(np.isfinite(result['viscosity_contrast']))


if __name__ == '__main__':
    pytest.main([__file__])