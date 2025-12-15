import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.strain_features import (
    compute_strain_rate_tensor, compute_velocity_features,
    compute_deformation_features, compute_strain_invariants,
    classify_strain_regime_simple, validate_strain_features
)


@pytest.fixture
def sample_strain_components():
    """Create sample strain rate components."""
    # Simple 2x2 grid
    dudx = np.array([[1e-6, -2e-6], [0, 1e-6]])    # Longitudinal strain
    dvdy = np.array([[-0.5e-6, 1e-6], [0.5e-6, -1e-6]])  # Transverse strain
    dudy = np.array([[0.2e-6, 0.3e-6], [-0.1e-6, 0.4e-6]])  # Shear component
    dvdx = np.array([[0.1e-6, -0.2e-6], [0.3e-6, -0.1e-6]])  # Shear component
    
    return dudx, dvdy, dudy, dvdx


@pytest.fixture
def sample_velocity_data():
    """Create sample velocity data."""
    u = np.array([[100, 200], [150, 250]])  # m/year
    v = np.array([[50, -30], [80, -10]])    # m/year
    return u, v


def test_compute_strain_rate_tensor(sample_strain_components):
    """Test strain rate tensor computation."""
    dudx, dvdy, dudy, dvdx = sample_strain_components
    
    result = compute_strain_rate_tensor(dudx, dvdy, dudy, dvdx)
    
    # Check all expected fields are present
    expected_fields = ['epsilon_xx', 'epsilon_yy', 'epsilon_xy', 'divergence', 
                      'shear_magnitude', 'effective_strain']
    for field in expected_fields:
        assert field in result
    
    # Check basic relationships
    np.testing.assert_array_equal(result['epsilon_xx'], dudx)
    np.testing.assert_array_equal(result['epsilon_yy'], dvdy)
    
    # Check divergence calculation
    expected_divergence = dudx + dvdy
    np.testing.assert_array_equal(result['divergence'], expected_divergence)
    
    # Check shear strain calculation
    expected_shear_xy = 0.5 * (dudy + dvdx)
    np.testing.assert_array_equal(result['epsilon_xy'], expected_shear_xy)


def test_compute_velocity_features(sample_velocity_data):
    """Test velocity feature computation."""
    u, v = sample_velocity_data
    
    result = compute_velocity_features(u, v)
    
    # Check expected fields
    expected_fields = ['speed', 'direction', 'log_speed']
    for field in expected_fields:
        assert field in result
    
    # Check speed calculation
    expected_speed = np.sqrt(u**2 + v**2)
    np.testing.assert_array_equal(result['speed'], expected_speed)
    
    # Check direction calculation
    expected_direction = np.arctan2(v, u)
    np.testing.assert_array_equal(result['direction'], expected_direction)
    
    # Check log_speed is finite
    assert np.all(np.isfinite(result['log_speed']))


def test_compute_deformation_features(sample_strain_components):
    """Test advanced deformation features."""
    dudx, dvdy, dudy, dvdx = sample_strain_components
    strain_data = compute_strain_rate_tensor(dudx, dvdy, dudy, dvdx)
    
    result = compute_deformation_features(strain_data)
    
    # Check expected fields
    expected_fields = ['dilatation_rate', 'rotation_rate', 'strain_anisotropy', 'pure_shear_ratio']
    for field in expected_fields:
        assert field in result
    
    # Check dilatation rate matches divergence
    np.testing.assert_array_equal(result['dilatation_rate'], strain_data['divergence'])
    
    # Check ratios are finite and non-negative
    assert np.all(result['strain_anisotropy'] >= 0)
    assert np.all(result['pure_shear_ratio'] >= 0)
    assert np.all(np.isfinite(result['strain_anisotropy']))
    assert np.all(np.isfinite(result['pure_shear_ratio']))


def test_compute_strain_invariants(sample_strain_components):
    """Test strain rate invariant computation."""
    dudx, dvdy, dudy, dvdx = sample_strain_components
    epsilon_xy = 0.5 * (dudy + dvdx)
    
    result = compute_strain_invariants(dudx, dvdy, epsilon_xy)
    
    # Check expected fields
    expected_fields = ['first_invariant', 'second_invariant', 'effective_strain']
    for field in expected_fields:
        assert field in result
    
    # Check first invariant (trace)
    expected_first = dudx + dvdy
    np.testing.assert_array_equal(result['first_invariant'], expected_first)
    
    # Check effective strain is non-negative
    assert np.all(result['effective_strain'] >= 0)


def test_classify_strain_regime_simple():
    """Test simple strain regime classification."""
    epsilon_xx = np.array([-1e-6, -0.1e-6, 0.1e-6, 1e-6])
    threshold = 0.5e-6
    
    result = classify_strain_regime_simple(epsilon_xx, threshold)
    
    # Check expected classification
    # -1e-6 < -0.5e-6 -> compression (0)
    # -0.1e-6 in [-0.5e-6, 0.5e-6] -> transition (1)  
    # 0.1e-6 in [-0.5e-6, 0.5e-6] -> transition (1)
    # 1e-6 > 0.5e-6 -> extension (2)
    expected = np.array([0, 1, 1, 2])
    np.testing.assert_array_equal(result, expected)


def test_classify_strain_regime_simple_default_threshold():
    """Test strain regime classification with default threshold."""
    epsilon_xx = np.array([-1e-6, 0, 1e-6])
    
    result = classify_strain_regime_simple(epsilon_xx)
    
    # With threshold=0: negative->compression, zero->transition, positive->extension
    expected = np.array([0, 1, 2])
    np.testing.assert_array_equal(result, expected)


def test_validate_strain_features_good_data(sample_strain_components):
    """Test validation with good strain features."""
    dudx, dvdy, dudy, dvdx = sample_strain_components
    strain_features = compute_strain_rate_tensor(dudx, dvdy, dudy, dvdx)
    
    # Should not raise any warnings or errors
    validate_strain_features(strain_features)


def test_validate_strain_features_negative_effective_strain():
    """Test validation with invalid effective strain."""
    bad_features = {'effective_strain': np.array([1e-6, -1e-6, 2e-6])}
    
    with pytest.warns(UserWarning, match="negative values"):
        validate_strain_features(bad_features)


def test_validate_strain_features_large_strain_rates():
    """Test validation with unusually large strain rates."""
    bad_features = {'epsilon_xx': np.array([1e-6, 0.5, 2e-6])}  # 0.5 is very large
    
    with pytest.warns(UserWarning, match="unusually large"):
        validate_strain_features(bad_features)


def test_validate_strain_features_nan_values():
    """Test validation with NaN values."""
    bad_features = {'epsilon_xx': np.array([1e-6, np.nan, 2e-6])}
    
    with pytest.warns(UserWarning, match="NaN or infinite"):
        validate_strain_features(bad_features)


def test_effective_strain_calculation():
    """Test effective strain calculation against known values."""
    # Simple case with known answer
    epsilon_xx = np.array([1e-6])
    epsilon_yy = np.array([0])
    epsilon_xy = np.array([0])
    
    strain_data = compute_strain_rate_tensor(epsilon_xx, epsilon_yy, 
                                           np.zeros_like(epsilon_xx), 
                                           np.zeros_like(epsilon_xx))
    
    # Effective strain should equal |epsilon_xx| when other components are zero
    expected_effective = np.abs(epsilon_xx)
    np.testing.assert_array_almost_equal(strain_data['effective_strain'], expected_effective)


def test_shear_magnitude_calculation():
    """Test shear magnitude calculation."""
    # Case with only shear strain
    dudx = np.array([0])
    dvdy = np.array([0])  
    dudy = np.array([2e-6])  # Shear component
    dvdx = np.array([0])     # Shear component
    
    result = compute_strain_rate_tensor(dudx, dvdy, dudy, dvdx)
    
    # epsilon_xy = 0.5 * (dudy + dvdx) = 0.5 * 2e-6 = 1e-6
    # shear_magnitude = sqrt(2) * |epsilon_xy| = sqrt(2) * 1e-6
    expected_shear = np.sqrt(2) * 1e-6
    np.testing.assert_array_almost_equal(result['shear_magnitude'], expected_shear)


if __name__ == '__main__':
    pytest.main([__file__])