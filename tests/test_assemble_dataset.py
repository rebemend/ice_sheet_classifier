import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.assemble_dataset import (
    interpolate_to_common_grid, check_grid_compatibility,
    create_feature_arrays
)


@pytest.fixture
def sample_unified_data():
    """Create sample unified data for testing."""
    # Simple 2x2 grid
    return {
        'x': np.array([[0, 1000], [0, 1000]]),
        'y': np.array([[0, 0], [1000, 1000]]),
        'u': np.array([[100, 150], [120, 180]]),
        'v': np.array([[50, 60], [55, 65]]),
        'h': np.array([[100, 120], [110, 130]]),
        'dudx': np.array([[1e-6, 2e-6], [1.5e-6, 2.5e-6]]),
        'dvdy': np.array([[-0.5e-6, 1e-6], [0.2e-6, -1e-6]]),
        'dudy': np.array([[0.1e-6, 0.2e-6], [-0.1e-6, 0.3e-6]]),
        'dvdx': np.array([[0.05e-6, -0.1e-6], [0.15e-6, -0.05e-6]]),
        'speed': np.array([[111.8, 161.55], [128.84, 191.05]]),
        'mu': np.array([[1e14, 2e14], [1.5e14, 3e14]]),
        'eta': np.array([[5e13, 8e13], [6e13, 1e14]]),
        'anisotropy': np.array([[2.0, 2.5], [2.5, 3.0]])
    }


def test_interpolate_to_common_grid():
    """Test grid interpolation functionality."""
    # Source data (1D coordinates)
    source_data = {
        'x': np.array([0, 1000]),
        'y': np.array([0, 1000]), 
        'field1': np.array([[1, 2], [3, 4]]),
        'field2': np.array([[10, 20], [30, 40]])
    }
    
    # Target grid (2D coordinates)
    target_x = np.array([[0, 500, 1000], [0, 500, 1000], [0, 500, 1000]])
    target_y = np.array([[0, 0, 0], [500, 500, 500], [1000, 1000, 1000]])
    
    fields_to_interpolate = ['field1', 'field2']
    
    result = interpolate_to_common_grid(source_data, target_x, target_y, fields_to_interpolate)
    
    # Check that interpolated fields have correct shape
    assert result['field1'].shape == target_x.shape
    assert result['field2'].shape == target_x.shape
    
    # Check that interpolation preserves corner values approximately
    np.testing.assert_allclose(result['field1'][0, 0], 1, rtol=0.1)
    np.testing.assert_allclose(result['field1'][-1, -1], 4, rtol=0.1)


def test_check_grid_compatibility_compatible():
    """Test grid compatibility check for compatible grids."""
    data1 = {
        'x': np.array([[0, 1000], [0, 1000]]),
        'y': np.array([[0, 0], [1000, 1000]])
    }
    
    data2 = {
        'x': np.array([[0, 1000], [0, 1000]]),
        'y': np.array([[0, 0], [1000, 1000]])
    }
    
    assert check_grid_compatibility(data1, data2) == True


def test_check_grid_compatibility_incompatible():
    """Test grid compatibility check for incompatible grids."""
    data1 = {
        'x': np.array([[0, 1000], [0, 1000]]),
        'y': np.array([[0, 0], [1000, 1000]])
    }
    
    data2 = {
        'x': np.array([[0, 500, 1000], [0, 500, 1000], [0, 500, 1000]]),
        'y': np.array([[0, 0, 0], [500, 500, 500], [1000, 1000, 1000]])
    }
    
    assert check_grid_compatibility(data1, data2) == False


def test_check_grid_compatibility_missing_coords():
    """Test grid compatibility when coordinates are missing."""
    data1 = {
        'x': np.array([0, 1000]),
        'y': np.array([0, 1000])
    }
    
    data2 = {
        'field': np.array([[1, 2], [3, 4]])  # No coordinates
    }
    
    assert check_grid_compatibility(data1, data2) == False


def test_create_feature_arrays_complete(sample_unified_data):
    """Test feature array creation with complete data."""
    feature_data = create_feature_arrays(sample_unified_data)
    
    # Check expected fields are present
    expected_fields = ['coordinates', 'baseline_features', 'primary_features', 
                      'mask', 'grid_shape', 'dudx', 'speed', 'mu', 'anisotropy']
    for field in expected_fields:
        assert field in feature_data
    
    # Check array shapes
    n_points = np.prod(sample_unified_data['u'].shape)
    assert feature_data['coordinates'].shape == (n_points, 2)
    assert feature_data['baseline_features'].shape == (n_points, 2)
    assert feature_data['primary_features'].shape == (n_points, 4)
    assert feature_data['mask'].shape == (n_points,)
    
    # Check that mask filters valid points
    assert np.all(feature_data['mask'])  # Should all be valid with good test data


def test_create_feature_arrays_missing_viscosity():
    """Test feature array creation without viscosity data."""
    incomplete_data = {
        'x': np.array([[0, 1000], [0, 1000]]),
        'y': np.array([[0, 0], [1000, 1000]]),
        'u': np.array([[100, 150], [120, 180]]),
        'v': np.array([[50, 60], [55, 65]]),
        'dudx': np.array([[1e-6, 2e-6], [1.5e-6, 2.5e-6]]),
        'speed': np.array([[111.8, 161.55], [128.84, 191.05]])
        # Missing mu, eta, anisotropy
    }
    
    feature_data = create_feature_arrays(incomplete_data)
    
    # Should still work but with warnings
    assert 'baseline_features' in feature_data
    assert 'primary_features' in feature_data
    
    # Primary features should be same as baseline when viscosity missing
    np.testing.assert_array_equal(
        feature_data['baseline_features'], 
        feature_data['primary_features']
    )


def test_create_feature_arrays_with_invalid_data():
    """Test feature array creation with some invalid data points."""
    invalid_data = {
        'x': np.array([[0, 1000], [0, 1000]]),
        'y': np.array([[0, 0], [1000, 1000]]),
        'u': np.array([[100, 150], [120, 180]]),
        'v': np.array([[50, 60], [55, 65]]),
        'dudx': np.array([[1e-6, np.nan], [1.5e-6, 2.5e-6]]),  # NaN value
        'speed': np.array([[111.8, 161.55], [128.84, -10]]),    # Negative speed
        'mu': np.array([[1e14, 2e14], [1.5e14, 3e14]]),
        'eta': np.array([[5e13, 8e13], [6e13, 1e14]]),
        'anisotropy': np.array([[2.0, 2.5], [2.5, 3.0]])
    }
    
    feature_data = create_feature_arrays(invalid_data)
    
    # Should have some invalid points
    assert not np.all(feature_data['mask'])
    
    # Should have fewer valid points than total
    n_total = np.prod(invalid_data['u'].shape)
    n_valid = np.sum(feature_data['mask'])
    assert n_valid < n_total


def test_create_feature_arrays_grid_shape():
    """Test that grid shape is correctly preserved."""
    data = {
        'x': np.array([[[0, 1000, 2000], [0, 1000, 2000]], 
                      [[0, 1000, 2000], [0, 1000, 2000]]]),
        'y': np.array([[[0, 0, 0], [1000, 1000, 1000]], 
                      [[2000, 2000, 2000], [3000, 3000, 3000]]]),
        'u': np.ones((2, 2, 3)) * 100,
        'v': np.ones((2, 2, 3)) * 50,
        'dudx': np.ones((2, 2, 3)) * 1e-6,
        'speed': np.ones((2, 2, 3)) * 111.8,
        'mu': np.ones((2, 2, 3)) * 1e14,
        'eta': np.ones((2, 2, 3)) * 5e13,
        'anisotropy': np.ones((2, 2, 3)) * 2.0
    }
    
    feature_data = create_feature_arrays(data)
    
    # Check grid shape is preserved
    expected_shape = (2, 2, 3)
    assert feature_data['grid_shape'] == expected_shape
    
    # Check total number of points
    n_total = np.prod(expected_shape)
    assert len(feature_data['mask']) == n_total


if __name__ == '__main__':
    pytest.main([__file__])