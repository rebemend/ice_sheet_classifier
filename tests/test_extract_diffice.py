import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.extract_diffice_amery import (
    extract_velocity_components, extract_coordinates, extract_ice_thickness,
    compute_strain_rates, validate_diffice_data
)


@pytest.fixture
def sample_diffice_data():
    """Create sample DIFFICE data dictionary."""
    # Simple 3x3 grid
    u = np.array([[100, 150, 200], [120, 180, 240], [140, 210, 280]])
    v = np.array([[50, 60, 70], [55, 65, 75], [60, 70, 80]])
    h = np.array([[100, 120, 140], [110, 130, 150], [120, 140, 160]])
    
    # 1D coordinates
    x = np.array([0, 1000, 2000])
    y = np.array([0, 1000, 2000])
    
    return {
        'u': u,
        'v': v,
        'h': h,
        'x': x,
        'y': y
    }


@pytest.fixture 
def sample_diffice_data_2d_coords():
    """Create sample DIFFICE data with 2D coordinate arrays."""
    u = np.array([[100, 150], [120, 180]])
    v = np.array([[50, 60], [55, 65]])
    h = np.array([[100, 120], [110, 130]])
    
    # 2D coordinate meshgrid
    x = np.array([[0, 1000], [0, 1000]])
    y = np.array([[0, 0], [1000, 1000]])
    
    return {
        'u': u,
        'v': v, 
        'h': h,
        'x': x,
        'y': y
    }


def test_extract_velocity_components_success(sample_diffice_data):
    """Test successful velocity extraction."""
    u, v = extract_velocity_components(sample_diffice_data)
    
    np.testing.assert_array_equal(u, sample_diffice_data['u'])
    np.testing.assert_array_equal(v, sample_diffice_data['v'])


def test_extract_velocity_components_missing():
    """Test error handling for missing velocity components."""
    bad_data = {'h': np.array([1, 2, 3])}  # Missing u and v
    
    with pytest.raises(KeyError, match="Velocity components not found"):
        extract_velocity_components(bad_data)


def test_extract_coordinates_success(sample_diffice_data):
    """Test successful coordinate extraction."""
    x, y = extract_coordinates(sample_diffice_data)
    
    np.testing.assert_array_equal(x, sample_diffice_data['x'])
    np.testing.assert_array_equal(y, sample_diffice_data['y'])


def test_extract_coordinates_missing():
    """Test error handling for missing coordinates."""
    bad_data = {'u': np.array([1, 2, 3])}  # Missing x and y
    
    with pytest.raises(KeyError, match="Coordinates not found"):
        extract_coordinates(bad_data)


def test_extract_ice_thickness_success(sample_diffice_data):
    """Test successful ice thickness extraction."""
    h = extract_ice_thickness(sample_diffice_data)
    
    np.testing.assert_array_equal(h, sample_diffice_data['h'])


def test_extract_ice_thickness_missing():
    """Test error handling for missing ice thickness."""
    bad_data = {'u': np.array([1, 2, 3])}  # Missing h
    
    with pytest.raises(KeyError, match="Ice thickness not found"):
        extract_ice_thickness(bad_data)


def test_compute_strain_rates_1d_coordinates(sample_diffice_data):
    """Test strain rate computation with 1D coordinates."""
    u = sample_diffice_data['u']
    v = sample_diffice_data['v'] 
    x = sample_diffice_data['x']
    y = sample_diffice_data['y']
    
    strain_rates = compute_strain_rates(u, v, x, y)
    
    # Check all expected fields are present
    expected_fields = ['dudx', 'dvdy', 'dudy', 'dvdx']
    for field in expected_fields:
        assert field in strain_rates
        assert strain_rates[field].shape == u.shape


def test_compute_strain_rates_2d_coordinates(sample_diffice_data_2d_coords):
    """Test strain rate computation with 2D coordinates."""
    u = sample_diffice_data_2d_coords['u']
    v = sample_diffice_data_2d_coords['v']
    x = sample_diffice_data_2d_coords['x'] 
    y = sample_diffice_data_2d_coords['y']
    
    strain_rates = compute_strain_rates(u, v, x, y)
    
    # Check all expected fields are present
    expected_fields = ['dudx', 'dvdy', 'dudy', 'dvdx']
    for field in expected_fields:
        assert field in strain_rates
        assert strain_rates[field].shape == u.shape


def test_compute_strain_rates_basic_calculation():
    """Test strain rate calculation with simple known case."""
    # Linear velocity field: u = x, v = y
    # This gives dudx = 1, dvdy = 1, dudy = 0, dvdx = 0
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    u = X.astype(float)  # u = x
    v = Y.astype(float)  # v = y
    
    strain_rates = compute_strain_rates(u, v, x, y)
    
    # Check dudx should be close to 1 (may have edge effects)
    assert np.allclose(strain_rates['dudx'], 1.0, atol=0.1)
    
    # Check dvdy should be close to 1 
    assert np.allclose(strain_rates['dvdy'], 1.0, atol=0.1)
    
    # Check dudy and dvdx should be close to 0
    assert np.allclose(strain_rates['dudy'], 0.0, atol=0.1)
    assert np.allclose(strain_rates['dvdx'], 0.0, atol=0.1)


def test_validate_diffice_data_success(sample_diffice_data):
    """Test validation with good DIFFICE data."""
    # Add strain rates for complete data
    complete_data = sample_diffice_data.copy()
    complete_data.update({
        'dudx': np.ones_like(sample_diffice_data['u']) * 1e-6,
        'speed': np.sqrt(sample_diffice_data['u']**2 + sample_diffice_data['v']**2)
    })
    
    # Should not raise any exceptions
    validate_diffice_data(complete_data)


def test_validate_diffice_data_missing_field():
    """Test validation with missing required field."""
    incomplete_data = {'u': np.array([1, 2, 3])}  # Missing required fields
    
    with pytest.raises(ValueError, match="Required field"):
        validate_diffice_data(incomplete_data)


def test_validate_diffice_data_shape_mismatch():
    """Test validation with shape mismatch."""
    bad_data = {
        'x': np.array([0, 1]),
        'y': np.array([0, 1]),
        'u': np.array([[1, 2], [3, 4]]),
        'v': np.array([1, 2, 3]),  # Wrong shape
        'h': np.array([[1, 2], [3, 4]]),
        'dudx': np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
    }
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_diffice_data(bad_data)


def test_validate_diffice_data_with_warnings(sample_diffice_data):
    """Test validation that produces warnings but doesn't fail."""
    # Add some problematic but not fatal data
    problematic_data = sample_diffice_data.copy()
    problematic_data.update({
        'dudx': np.ones_like(sample_diffice_data['u']) * 1e-6,
        'speed': np.sqrt(sample_diffice_data['u']**2 + sample_diffice_data['v']**2)
    })
    
    # Add some NaN values by converting to float first
    problematic_data['u'] = problematic_data['u'].astype(float)
    problematic_data['u'][0, 0] = np.nan
    
    # Should produce warnings but not fail
    with pytest.warns(UserWarning):
        validate_diffice_data(problematic_data)


def test_coordinate_spacing_extraction_1d():
    """Test coordinate spacing calculation for 1D arrays."""
    x = np.array([0, 1000, 2000, 3000])
    y = np.array([0, 500, 1000])
    u = np.ones((3, 4))
    v = np.ones((3, 4))
    
    strain_rates = compute_strain_rates(u, v, x, y)
    
    # Should work without errors
    assert 'dudx' in strain_rates
    assert strain_rates['dudx'].shape == (3, 4)


def test_coordinate_spacing_extraction_2d():
    """Test coordinate spacing calculation for 2D arrays."""
    x = np.array([[0, 1000], [0, 1000]])
    y = np.array([[0, 0], [500, 500]])
    u = np.ones((2, 2))
    v = np.ones((2, 2))
    
    strain_rates = compute_strain_rates(u, v, x, y)
    
    # Should work without errors
    assert 'dudx' in strain_rates
    assert strain_rates['dudx'].shape == (2, 2)


def test_single_point_coordinates():
    """Test handling of single point coordinate arrays."""
    # numpy.gradient requires at least 2 points, so test with minimal case
    x = np.array([0, 1000])  
    y = np.array([0, 1000])
    u = np.array([[100, 150], [120, 180]]) 
    v = np.array([[50, 60], [55, 65]])
    
    # Should work with minimal grid
    strain_rates = compute_strain_rates(u, v, x, y)
    
    assert 'dudx' in strain_rates
    assert strain_rates['dudx'].shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__])