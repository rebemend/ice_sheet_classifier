# Data Loading Module

This module handles loading and preprocessing of ice shelf data from various sources including DIFFUSE repositories and MATLAB result files.

## Overview

The data loading module provides a unified interface for accessing ice shelf data regardless of the original format. It handles coordinate transformations, data validation, and format standardization.

## Key Components

### **assemble_dataset.py**
Main data assembly pipeline that combines multiple data sources.

**Key Functions:**
- `create_complete_dataset()` - Full pipeline from raw data to processed dataset
- `load_processed_dataset()` - Load previously processed data files
- `interpolate_to_common_grid()` - Spatial interpolation between different grids

**Usage:**
```python
from data_loading.assemble_dataset import create_complete_dataset, load_processed_dataset

# Process raw data
unified_data, feature_data = create_complete_dataset(
    diffice_data_path="data/DIFFICE_jax/",
    viscosity_data_path="data/raw/results.mat",
    output_path="data/processed/dataset.npz"
)

# Load processed data
unified_data, feature_data = load_processed_dataset("data/processed/dataset.npz")
```

### **extract_diffice_amery.py**
Interface to DIFFUSE repository data for Amery ice shelf.

**Key Functions:**
- `load_and_process_diffice_amery()` - Extract velocity and coordinate data
- `validate_diffice_data()` - Data quality validation
- `get_amery_data_info()` - Metadata extraction

**Data Handling:**
- Loads velocity fields (u, v components)
- Extracts coordinate grids (x, y)
- Handles missing data and NaN values
- Performs coordinate system transformations

### **load_matlab.py**
MATLAB file processing for viscosity and derived quantities.

**Key Functions:**
- `load_and_validate_viscosity()` - Load viscosity fields from .mat files
- `extract_matlab_results()` - Parse MATLAB result structures
- `validate_matlab_data()` - Consistency checks

**Supported Formats:**
- MATLAB .mat files with 'results' structure
- Viscosity fields: mu (horizontal), eta (vertical)
- Pre-computed strain rates and derived quantities

### **load_results_mat.py**
Specialized loader for complete MATLAB analysis results.

**Key Functions:**
- `load_complete_results_data()` - Load full result dataset
- `validate_results_data()` - Comprehensive validation
- `extract_coordinate_info()` - Coordinate system handling

## Data Flow

### **Input Data Sources**
1. **DIFFUSE Repository** (`data/DIFFICE_jax/`):
   - Velocity fields from ice flow simulations
   - Coordinate grids in polar stereographic projection
   - Ice thickness data (optional)

2. **MATLAB Results** (`data/raw/results.mat`):
   - Viscosity fields (μ, η)
   - Strain rate tensors
   - Derived physical quantities

### **Processing Pipeline**
```
Raw Data → Validation → Coordinate Alignment → Interpolation → Unified Dataset
   ↓            ↓              ↓                 ↓              ↓
DIFFUSE/    Quality       Grid Matching     Common Grid    Processed .npz
MATLAB      Checks        & Transforms      Interpolation     File
```

### **Output Format**
Processed datasets are saved as NumPy .npz files with two main components:

**unified_data** (physical fields):
- `u`, `v`: Velocity components [m/s]
- `speed`: Velocity magnitude [m/s]
- `x`, `y`: Coordinate grids [m]
- `mu`, `eta`: Viscosity fields [Pa·s]
- `primary_strain`: Longitudinal strain rate [1/s]
- `anisotropy`: Viscosity ratio (μ/η)

**feature_data** (metadata):
- `coordinates`: Grid coordinate arrays
- `grid_shape`: Spatial dimensions
- `mask`: Valid data mask
- `baseline_features`: Basic feature set
- `primary_features`: Extended feature set

## Usage Patterns

### **Basic Data Loading**
```python
from data_loading.assemble_dataset import load_processed_dataset

# Load preprocessed data
unified_data, feature_data = load_processed_dataset("data/processed/dataset.npz")

# Extract key fields
velocity_u = unified_data['u']
velocity_v = unified_data['v']
viscosity = unified_data['mu']
coordinates = feature_data['coordinates']
```

### **Full Pipeline Processing**
```python
from data_loading.assemble_dataset import create_complete_dataset

# Process from raw data
unified_data, feature_data = create_complete_dataset(
    diffice_data_path="data/DIFFICE_jax/examples/real_data/data_pinns_Amery.mat",
    viscosity_data_path="data/raw/results.mat",
    output_path="data/processed/amery_dataset.npz"
)
```

### **Validation and Quality Checks**
```python
from data_loading.extract_diffice_amery import validate_diffice_data
from data_loading.load_matlab import validate_matlab_data

# Validate data sources
diffice_valid = validate_diffice_data(diffice_data)
matlab_valid = validate_matlab_data(matlab_data)
```

## Data Quality Assurance

### **Validation Checks**
- Coordinate system consistency
- Physical value ranges (velocities, viscosities, strain rates)
- Grid alignment between data sources
- Missing data handling

### **Quality Metrics**
- Data completeness percentage
- Coordinate transformation accuracy
- Interpolation error estimates
- Physical quantity validation ranges

### **Error Handling**
- Graceful handling of missing files
- Warning messages for data quality issues
- Fallback options for incomplete datasets
- Detailed error reporting for debugging

## Configuration Options

### **Interpolation Methods**
- `linear`: Fast linear interpolation (default)
- `cubic`: Smooth cubic interpolation
- `nearest`: Nearest neighbor (for discrete fields)

### **Coordinate Systems**
- Automatic detection of projection parameters
- Support for polar stereographic coordinates
- Coordinate transformation validation

### **Data Filtering**
- Configurable valid data ranges
- NaN handling strategies
- Outlier detection and removal

## Performance Considerations

- **Memory efficient**: Processes data in chunks for large datasets
- **Lazy loading**: Only loads required fields
- **Caching**: Saves intermediate results for repeated processing
- **Parallel processing**: Utilizes multiple cores for interpolation

## Dependencies

- **NumPy**: Array operations and file I/O
- **SciPy**: Interpolation and coordinate transformations
- **warnings**: Data quality reporting