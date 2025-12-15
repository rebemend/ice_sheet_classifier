# Tests Directory

This directory contains unit tests for the ice shelf classifier modules. The tests ensure code reliability, validate functionality, and prevent regressions during development.

## Overview

The test suite covers all major components of the ice shelf classifier with focused unit tests that validate individual functions and classes. Tests are designed to run quickly and provide clear feedback on code quality.

## Test Structure

### **Test Organization**
Tests are organized by module, following the source code structure:

```
tests/
├── test_assemble_dataset.py      # Data loading and assembly tests
├── test_clustering.py            # Clustering algorithm tests  
├── test_extract_diffice.py       # DIFFUSE data extraction tests
├── test_load_matlab.py           # MATLAB data loading tests
├── test_strain_features.py       # Strain rate computation tests
└── test_viscosity_features.py    # Viscosity feature tests
```

### **Test Categories**

#### **Unit Tests**
- **Function-level testing**: Individual function correctness
- **Input validation**: Parameter checking and error handling
- **Edge case handling**: Boundary conditions and special cases
- **Output validation**: Result format and value correctness

#### **Integration Tests**
- **Module interaction**: Data flow between components
- **Pipeline validation**: End-to-end workflow testing
- **Data consistency**: Cross-module data integrity

#### **Performance Tests**
- **Speed validation**: Execution time within reasonable bounds
- **Memory usage**: Resource consumption monitoring
- **Scalability**: Behavior with different data sizes

## Key Test Files

### **test_assemble_dataset.py**
Tests for data loading and assembly functionality.

**Test Coverage:**
- Data loading from different sources (DIFFUSE, MATLAB)
- Coordinate system transformations and interpolation
- Data validation and quality checks
- Output format consistency

**Key Tests:**
```python
def test_load_processed_dataset():
    """Test loading of preprocessed dataset files."""
    
def test_interpolate_to_common_grid():
    """Test spatial interpolation between different grids."""
    
def test_data_validation():
    """Test input data validation and error handling."""
```

### **test_clustering.py**
Tests for k-means clustering implementation.

**Test Coverage:**
- KMeansRunner functionality
- K-selection methods (elbow, silhouette)
- Timeout and performance optimizations
- Result structure validation

**Key Tests:**
```python
def test_kmeans_runner_basic():
    """Test basic k-means clustering functionality."""
    
def test_silhouette_sampling():
    """Test sampling optimization for large datasets."""
    
def test_k_selection_methods():
    """Test k-value selection algorithms."""
```

### **test_extract_diffice.py**
Tests for DIFFUSE data extraction.

**Test Coverage:**
- DIFFUSE repository data loading
- Coordinate system handling
- Data format validation
- Error handling for missing files

### **test_load_matlab.py**
Tests for MATLAB file processing.

**Test Coverage:**
- MATLAB .mat file loading
- Structure parsing and validation
- Viscosity data extraction
- Format compatibility checks

### **test_strain_features.py**
Tests for strain rate calculations.

**Test Coverage:**
- Strain rate tensor computations
- Velocity gradient processing
- Physical value validation
- Mathematical correctness

**Key Tests:**
```python
def test_strain_rate_tensor():
    """Test strain rate tensor computation from velocity gradients."""
    
def test_effective_strain_rate():
    """Test Von Mises effective strain rate calculation."""
    
def test_physical_ranges():
    """Test that computed strain rates are within physical ranges."""
```

### **test_viscosity_features.py**
Tests for viscosity-based features.

**Test Coverage:**
- Viscosity anisotropy calculations
- Stress computations
- Rheological parameter validation
- Feature scaling and normalization

## Running Tests

### **All Tests**
```bash
# Run complete test suite
python -m pytest tests/

# With verbose output
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### **Individual Test Files**
```bash
# Test specific module
python -m pytest tests/test_clustering.py -v

# Test specific function
python -m pytest tests/test_clustering.py::test_kmeans_runner_basic -v
```

### **Test Categories**
```bash
# Run only fast tests (< 1 second)
python -m pytest tests/ -m "not slow"

# Run integration tests
python -m pytest tests/ -m "integration"

# Run performance tests
python -m pytest tests/ -m "performance"
```

## Test Data

### **Synthetic Test Data**
Tests use small, controlled synthetic datasets to ensure:
- **Predictable results**: Known expected outcomes
- **Fast execution**: Tests run quickly
- **Comprehensive coverage**: All code paths tested
- **Reproducibility**: Consistent results across runs

### **Test Data Generation**
```python
# Example synthetic data generation
import numpy as np

def generate_test_ice_data():
    """Generate synthetic ice shelf data for testing."""
    # Create simple velocity field
    x = np.linspace(0, 1000, 50)
    y = np.linspace(0, 500, 25)
    X, Y = np.meshgrid(x, y)
    
    # Simple linear velocity profile
    u = 1e-6 * X  # Increasing velocity downstream
    v = np.zeros_like(u)
    
    # Corresponding viscosity field
    mu = 1e14 * np.ones_like(u)
    eta = 1e13 * np.ones_like(u)
    
    return {
        'x': X, 'y': Y,
        'u': u, 'v': v,
        'mu': mu, 'eta': eta
    }
```

### **Real Data Testing**
Some tests use small subsets of real data for:
- **Format validation**: Ensure real data loads correctly
- **Integration testing**: Verify end-to-end workflows
- **Regression testing**: Detect changes in behavior with real data

## Test Quality Metrics

### **Code Coverage**
Target coverage levels:
- **Overall**: >90% line coverage
- **Critical paths**: 100% coverage for core algorithms
- **Error handling**: All exception paths tested

### **Test Performance**
Performance requirements:
- **Individual tests**: <1 second execution time
- **Full test suite**: <30 seconds total runtime
- **Memory usage**: <500MB peak memory usage

### **Test Reliability**
- **Deterministic**: All tests produce consistent results
- **Independent**: Tests don't depend on execution order
- **Isolated**: No side effects between tests
- **Robust**: Tests handle minor data variations

## Test Configuration

### **pytest Configuration**
```ini
# pytest.ini configuration
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts = 
    -v
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow (> 1 second)
    integration: marks tests as integration tests
    performance: marks tests as performance tests
```

### **Test Markers**
```python
import pytest

# Mark slow tests
@pytest.mark.slow
def test_large_dataset_processing():
    """Test with large dataset (may take several seconds)."""
    pass

# Mark integration tests
@pytest.mark.integration  
def test_full_pipeline():
    """Test complete analysis pipeline."""
    pass

# Mark performance tests
@pytest.mark.performance
def test_clustering_speed():
    """Test clustering performance meets requirements."""
    pass
```

## Writing New Tests

### **Test Structure Template**
```python
import pytest
import numpy as np
from src.module_name import function_to_test

class TestFunctionName:
    """Test class for function_to_test."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_data = generate_test_data()
    
    def test_basic_functionality(self):
        """Test basic function behavior."""
        result = function_to_test(self.test_data)
        assert result is not None
        assert isinstance(result, expected_type)
    
    def test_input_validation(self):
        """Test input parameter validation."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
    
    def test_edge_cases(self):
        """Test boundary conditions and edge cases."""
        # Empty input
        result = function_to_test([])
        assert len(result) == 0
        
        # Single element
        result = function_to_test([single_element])
        assert len(result) == 1
    
    def test_physical_constraints(self):
        """Test that results satisfy physical constraints."""
        result = function_to_test(self.test_data)
        
        # Check physical ranges
        assert np.all(result >= physical_min)
        assert np.all(result <= physical_max)
        
        # Check conservation laws
        assert np.abs(conservation_check(result)) < tolerance
```

### **Test Best Practices**

#### **Clear Test Names**
```python
# Good: Descriptive test names
def test_kmeans_converges_with_valid_input():
def test_strain_rate_calculation_handles_zero_gradients():
def test_feature_scaling_preserves_data_shape():

# Bad: Vague test names
def test_function():
def test_basic():
def test_edge_case():
```

#### **Isolated Tests**
```python
# Good: Each test is independent
def test_feature_computation():
    data = generate_fresh_test_data()
    result = compute_features(data)
    assert_expected_result(result)

# Bad: Tests depend on shared state
global_data = None  # Don't do this

def test_setup():
    global global_data
    global_data = load_data()

def test_computation():  # Depends on test_setup
    result = compute_features(global_data)
```

#### **Comprehensive Assertions**
```python
# Good: Test multiple aspects
def test_clustering_result():
    result = run_clustering(data, k=3)
    
    # Test result structure
    assert 'labels' in result
    assert 'centroids' in result
    
    # Test data shapes
    assert len(result['labels']) == len(data)
    assert result['centroids'].shape == (3, n_features)
    
    # Test value ranges
    assert np.all(result['labels'] >= 0)
    assert np.all(result['labels'] < 3)
```

## Continuous Integration

### **Automated Testing**
Tests run automatically on:
- **Code commits**: Validate changes don't break functionality
- **Pull requests**: Ensure code quality before merging
- **Releases**: Comprehensive testing before deployment

### **Test Reports**
- **Coverage reports**: Track test coverage over time
- **Performance reports**: Monitor test execution speed
- **Failure analysis**: Detailed debugging information for failed tests

## Dependencies

- **pytest**: Testing framework
- **numpy**: Test data generation and numerical assertions
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **hypothesis**: Property-based testing (optional)