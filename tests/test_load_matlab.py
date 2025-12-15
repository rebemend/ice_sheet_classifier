import numpy as np
import pytest
import tempfile
import os
from scipy.io import savemat
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.load_matlab import (
    load_viscosity_data, compute_anisotropy_ratio,
    validate_viscosity_data, load_and_validate_viscosity
)


@pytest.fixture
def sample_viscosity_data():
    """Create sample viscosity data for testing."""
    mu = np.array([[1e14, 2e14], [1.5e14, 3e14]])
    eta = np.array([[5e13, 8e13], [6e13, 1e14]])
    x = np.array([0, 1000])
    y = np.array([0, 1000])
    
    return {
        'mu': mu,
        'eta': eta,
        'x': x,
        'y': y
    }


@pytest.fixture
def temp_mat_file(sample_viscosity_data):
    """Create temporary .mat file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        mat_path = f.name
    
    savemat(mat_path, sample_viscosity_data)
    
    yield mat_path
    
    # Cleanup
    os.unlink(mat_path)


def test_load_viscosity_data_success(temp_mat_file, sample_viscosity_data):
    """Test successful loading of viscosity data."""
    loaded_data = load_viscosity_data(temp_mat_file)
    
    # Check all expected fields are present
    assert 'mu' in loaded_data
    assert 'eta' in loaded_data
    assert 'x' in loaded_data
    assert 'y' in loaded_data
    
    # Check data values match
    np.testing.assert_array_equal(loaded_data['mu'], sample_viscosity_data['mu'])
    np.testing.assert_array_equal(loaded_data['eta'], sample_viscosity_data['eta'])


def test_load_viscosity_data_file_not_found():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        load_viscosity_data('/nonexistent/path.mat')


def test_compute_anisotropy_ratio():
    """Test anisotropy ratio computation."""
    mu = np.array([2.0, 4.0, 6.0])
    eta = np.array([1.0, 2.0, 3.0])
    
    expected = np.array([2.0, 2.0, 2.0])
    result = compute_anisotropy_ratio(mu, eta)
    
    np.testing.assert_array_equal(result, expected)


def test_compute_anisotropy_ratio_division_by_zero():
    """Test anisotropy ratio with near-zero denominator."""
    mu = np.array([1.0, 2.0])
    eta = np.array([1e-15, 1.0])  # Very small value
    
    result = compute_anisotropy_ratio(mu, eta)
    
    # Should not have inf or nan values
    assert np.all(np.isfinite(result))
    # First ratio should be large but finite
    assert result[0] > 1e10


def test_validate_viscosity_data_success(sample_viscosity_data):
    """Test validation with good data."""
    # Should not raise any exception
    validate_viscosity_data(sample_viscosity_data)


def test_validate_viscosity_data_missing_field():
    """Test validation with missing required field."""
    bad_data = {'mu': np.array([1, 2, 3])}  # Missing eta
    
    with pytest.raises(ValueError, match="Required field 'eta' missing"):
        validate_viscosity_data(bad_data)


def test_validate_viscosity_data_shape_mismatch():
    """Test validation with mismatched shapes."""
    bad_data = {
        'mu': np.array([[1, 2], [3, 4]]),
        'eta': np.array([1, 2, 3])  # Different shape
    }
    
    with pytest.raises(ValueError, match="shapes don't match"):
        validate_viscosity_data(bad_data)


def test_validate_viscosity_data_negative_values():
    """Test validation with non-positive viscosity."""
    bad_data = {
        'mu': np.array([1, -2, 3]),  # Negative value
        'eta': np.array([1, 2, 3])
    }
    
    with pytest.raises(ValueError, match="non-positive values"):
        validate_viscosity_data(bad_data)


def test_validate_viscosity_data_nan_values():
    """Test validation with NaN values."""
    bad_data = {
        'mu': np.array([1, np.nan, 3]),
        'eta': np.array([1, 2, 3])
    }
    
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_viscosity_data(bad_data)


def test_load_and_validate_viscosity(temp_mat_file):
    """Test complete load and validation pipeline."""
    result = load_and_validate_viscosity(temp_mat_file)
    
    # Check all expected fields
    assert 'mu' in result
    assert 'eta' in result
    assert 'anisotropy' in result
    
    # Check anisotropy is computed correctly
    expected_anisotropy = result['mu'] / result['eta']
    np.testing.assert_array_equal(result['anisotropy'], expected_anisotropy)


if __name__ == '__main__':
    pytest.main([__file__])