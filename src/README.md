# Source Code (src/)

This directory contains the core implementation of the ice shelf classifier, organized into modular components for data loading, feature engineering, clustering, utilities, and visualization.

## Module Overview

### 📂 **Module Structure**
```
src/
├── data_loading/     # Data I/O and preprocessing
├── features/         # Feature computation and engineering  
├── clustering/       # K-means algorithms and analysis
├── utils/           # Utilities and validation
└── visualization/   # Plotting and spatial analysis
```

### 🔄 **Data Flow**
```
Raw Data → data_loading → features → clustering → visualization
    ↓            ↓            ↓          ↓            ↓
DIFFUSE/     Processed    Feature    Cluster    Plots &
MATLAB       Arrays      Vectors    Labels     Analysis
```

## Module Descriptions

### **data_loading/**
- **Purpose**: Load and preprocess ice shelf data from various sources
- **Key Functions**: DIFFUSE data extraction, MATLAB file parsing, data validation
- **Main Files**: 
  - `assemble_dataset.py` - Main data assembly pipeline
  - `extract_diffice_amery.py` - DIFFUSE repository interface
  - `load_matlab.py` - MATLAB file processing
- **Usage**: Called automatically by main scripts, can be used standalone for data preparation

### **features/**
- **Purpose**: Compute physical features from velocity and viscosity fields
- **Key Functions**: Strain rate calculations, viscosity features, feature set selection
- **Main Files**:
  - `feature_sets.py` - Feature combination definitions
  - `strain_features.py` - Strain rate computations
  - `viscosity_features.py` - Viscosity-based features
- **Usage**: Feature computation pipeline called during analysis

### **clustering/**
- **Purpose**: K-means clustering implementation with ice shelf-specific optimizations
- **Key Functions**: Optimized k-means, k-selection, feature ablation analysis
- **Main Files**:
  - `kmeans_runner.py` - Main clustering engine (with timeout fixes)
  - `k_selection.py` - Elbow method implementation
  - `ablation.py` - Feature importance analysis
- **Usage**: Core clustering functionality used by all analysis scripts

### **utils/**
- **Purpose**: Utilities for feature scaling, validation, and data quality checks
- **Key Functions**: Feature scaling, data validation, coordinate transformations
- **Main Files**:
  - `scaling.py` - Feature normalization and scaling
- **Usage**: Supporting utilities used throughout the pipeline

### **visualization/**
- **Purpose**: Spatial visualization and feature space analysis
- **Key Functions**: Cluster maps, feature distributions, PCA plots
- **Main Files**:
  - `spatial_maps.py` - Geographic cluster visualizations
  - `feature_space.py` - Feature space analysis and plotting
- **Usage**: Visualization pipeline for analysis results

## Usage Patterns

### **Import Structure**
The modules use a clean import structure:
```python
# From scripts or tests:
from data_loading.assemble_dataset import load_processed_dataset
from features.feature_sets import compute_primary_features
from clustering.kmeans_runner import KMeansRunner
from utils.scaling import scale_features_for_clustering
from visualization.spatial_maps import plot_cluster_map
```

### **Typical Workflow**
```python
# 1. Load data
unified_data, feature_data = load_processed_dataset(data_path)

# 2. Compute features
features = compute_primary_features(unified_data)

# 3. Scale features
scaled_features = scale_features_for_clustering(features)

# 4. Run clustering
runner = KMeansRunner()
results = runner.run_single_kmeans(scaled_features, k=3)

# 5. Visualize results
plot_cluster_map(results, coordinates, output_path)
```

## Key Design Principles

### **Modularity**
- Each module has a single responsibility
- Clean interfaces between components
- Can be used independently or as part of pipeline

### **Performance Optimization**
- Sampling-based silhouette computation for large datasets
- Efficient memory usage patterns
- Timeout protection for long-running operations

### **Robustness**
- Comprehensive input validation
- Error handling and warnings
- Data quality checks throughout pipeline

### **Flexibility**
- Multiple feature sets supported
- Configurable scaling methods
- Extensible architecture for new features

## Development Guidelines

### **Adding New Features**
1. Add computation logic to appropriate `features/` module
2. Update feature set definitions in `feature_sets.py`
3. Add validation checks in `utils/`
4. Update visualization if needed

### **Extending Clustering**
1. Add new algorithms to `clustering/` module
2. Follow `KMeansRunner` interface pattern
3. Include performance optimizations
4. Add appropriate tests

### **Data Source Integration**
1. Create new loader in `data_loading/`
2. Follow existing validation patterns
3. Update `assemble_dataset.py` for integration
4. Add format documentation

## Testing
Each module has corresponding unit tests in the `tests/` directory. The tests focus on:
- Function correctness
- Data validation
- Edge case handling
- Performance regression detection