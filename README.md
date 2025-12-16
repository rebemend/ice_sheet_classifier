# Ice Shelf Classifier

A machine learning pipeline for classifying ice shelf flow regimes using k-means clustering on velocity, strain rate, and viscosity data.

## Overview

This repository implements an unsupervised classification system that identifies three main ice shelf flow regimes:
- **Compression**: Areas with negative longitudinal strain (near grounding line)
- **Transition**: Low strain rate regions (rheological boundaries) 
- **Extension**: Areas with positive longitudinal strain (toward calving front)

The classifier uses k-means clustering on physical features derived from ice shelf velocity and viscosity fields.

### Repository Structure

  ice_sheet_classifier/
  ├── README.md              
  ├── requirements.txt       # Dependencies
  ├── environment.yml        # Conda environment
  ├── main.py               # Complete GUI pipeline
  ├── scripts/              # Main execution scripts
  │   ├── run_k_selection.py      # K-value selection (elbow method)
  │   └── run_optimized_kmeans.py # Main k-means classification
  ├── src/                  # Source code modules
  │   ├── clustering/       # K-means analysis
  │   ├── data_loading/     # Data processing
  │   ├── features/         # Feature engineering
  │   └── utils/            # Scaling utilities
  ├── tests/               # Unit tests
  ├── data/                # Data files
  │   ├── DIFFICE_jax/      # DIFFICE repository data
  │   ├── raw/              # Raw data files
  │   └── real_data_analysis/ # Processed dataset
  └── planning/           # Planning documents

### Setup
```bash
# Clone the repository
git clone https://github.com/rebemend/ice_sheet_classifier
cd ice_sheet_classifier

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Complete Pipeline (Recommended)
Run the entire pipeline with interactive GUI for k-selection:
```bash
python main.py
```

Or with custom paths:
```bash
python main.py \
    --diffice_data data/DIFFICE_jax \
    --viscosity_data data/raw/results.mat \
    --output_dir results/
```

This will:
1. Use existing processed dataset (data/real_data_analysis/processed_dataset.npz)
2. Show k-selection analysis plots and prompt for k value  
3. Run k-means classification with selected k
4. Display spatial classification results in GUI

**Note:** The pipeline uses the existing processed Amery ice shelf dataset. If you need to process new raw data, use the individual scripts first to create a processed dataset.

### Individual Scripts

#### 1. K-Value Selection (Elbow Method)
To determine the optimal number of clusters for your dataset:
```bash
python scripts/run_k_selection.py \
      --diffice_data dummy \
      --viscosity_data dummy \
      --processed_data data/real_data_analysis/processed_dataset.npz \
      --output_dir results/k_select/ \
      --k_range 2 8
```

#### 2. K-Means Classification
Once you've determined the optimal k value (typically k=3 for ice shelf regimes):
```bash
python scripts/run_optimized_kmeans.py \
      --processed_data data/real_data_analysis/processed_dataset.npz \
      --output_dir results/k_classify/ \
      --k 3
```


## Data Requirements

### Input Data Formats

The pipeline expects two main data sources:

1. **DIFFUSE Repository Data** (`data/DIFFICE_jax/`):
   - Ice shelf velocity fields (u, v)
   - Coordinate grids (x, y) 
   - Optional: ice thickness (h)

2. **MATLAB Results File** (`data/raw/results.mat`):
   - Viscosity fields (mu, eta)
   - Pre-computed strain rates
   - Properly scaled physical quantities

### Required Fields

The classifier requires these physical fields:
- `u`, `v`: Velocity components [m/s]
- `mu`, `eta`: Horizontal and vertical viscosity [Pa·s]  
- `x`, `y`: Spatial coordinates [m]
- `dudx`, `dvdy`, `dudy`, `dvdx`: Velocity gradients [1/s]

## Feature Sets

The pipeline supports different feature combinations:

### Primary (Recommended)
- `dudx`: Longitudinal strain rate (∂u/∂x)
- `speed`: Velocity magnitude (√(u² + v²))
- `mu`: Horizontal viscosity
- `anisotropy`: Viscosity ratio (μ/η)

### Baseline  
- `dudx`: Longitudinal strain rate
- `speed`: Velocity magnitude

### Extended
- All primary features plus:
- `effective_strain`: Von Mises strain rate
- `divergence`: Velocity divergence  
- `h`: Ice thickness

### Stress-Based
- `dudx`: Longitudinal strain rate
- `speed`: Velocity magnitude
- `deviatoric_stress`: Stress magnitude
- `stress_anisotropy`: Stress ratio
- `stress_indicator`: Shear/normal stress ratio

## Scripts Overview

### Main Pipeline

- **`main.py`**: Complete GUI-driven pipeline with k-selection and classification
- **`scripts/run_optimized_kmeans.py`**: Main clustering script with comprehensive visualizations
- **`scripts/run_k_selection.py`**: K-value selection with elbow method and silhouette analysis

### Source Modules

- **`src/data_loading/`**: Data loading and preprocessing
- **`src/features/`**: Feature computation and selection  
- **`src/utils/`**: Scaling utilities
- **`src/clustering/`**: K-means analysis and k-selection

## Output Files

### Results Directory Structure
```
results/
├── optimized_clustering_results.npz    # Main results
├── clustering_summary.json             # Analysis summary  
├── spatial_clusters.png               # Cluster map
├── spatial_features.png               # Feature distributions
└── spatial_regimes.png                # Physical interpretation
```

### Key Output Files

- **`clustering_results.npz`**: NumPy archive with:
  - `labels`: Cluster assignments
  - `centroids`: Cluster centers
  - `coordinates`: Spatial positions
  - `features_scaled`: Processed features

- **`clustering_summary.json`**: Analysis metrics:
  - Silhouette scores
  - Cluster balance ratios
  - Performance timing
  - Physical interpretation

## Performance Considerations

### Large Dataset Optimization

For datasets > 50k points, the optimized script automatically:
- Samples data for expensive silhouette computation
- Uses efficient k-means parameters
- Bypasses redundant analysis steps

### Memory Usage

- ~200k points: ~8GB RAM for full analysis
- ~100k points: ~4GB RAM recommended
- Use `--analysis_only` to skip visualizations

### Typical Runtime
- Data loading: ~0.1s
- Feature processing: ~0.5s  
- K-means clustering: ~0.2s (200k points)
- Silhouette computation: ~60s (full) vs ~2s (sampled)

## Configuration

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--k` | Number of clusters | 3 |
| `--feature_set` | Feature combination | 'primary' |
| `--scaling_method` | Feature scaling | 'standard' |
| `--analysis_only` | Skip visualizations | False |
| `--max_silhouette_size` | Sample size for silhouette | 50000 |

### Environment Variables

Set these if needed:
- `PYTHONPATH`: Include `src/` directory
- `OMP_NUM_THREADS`: Control threading for sklearn

## Troubleshooting

### Common Issues

1. **Timeout/Performance Issues**
   - Use `run_optimized_kmeans.py` for large datasets
   - Reduce `max_silhouette_size` parameter
   - Set `--analysis_only` to skip visualizations

2. **Data Loading Errors** 
   - Check file paths are absolute
   - Verify MATLAB file contains 'results' structure
   - Ensure DIFFUSE data has required fields

3. **Feature Computation Issues**
   - Check for NaN values in input data
   - Verify coordinate grid consistency
   - Use `validate_*_data()` functions

4. **Memory Issues**
   - Sample the dataset (use subset of points)
   - Reduce `n_init` parameter for k-means
   - Process data in chunks

### Debug Mode

Run with verbose output:
```bash
python -u scripts/run_optimized_kmeans.py [options] 2>&1 | tee debug.log
```

## Physical Interpretation

### Ice Shelf Regimes

The classifier identifies flow regimes based on strain rate patterns:

- **Cluster 0** (typically): Compression regime
  - Negative ∂u/∂x (longitudinal compression)
  - Near grounding line
  - High viscosity, low anisotropy

- **Cluster 1** (typically): Transition regime  
  - Low strain rates
  - Rheological boundaries
  - Moderate viscosity and anisotropy

- **Cluster 2** (typically): Extension regime
  - Positive ∂u/∂x (longitudinal extension)  
  - Toward calving front
  - Lower viscosity, higher anisotropy

### Validation

Physical consistency checks:
- Strain rates: 10⁻¹² to 10⁻⁹ s⁻¹
- Viscosity: 10¹² to 10¹⁶ Pa·s
- Velocities: 10⁻⁸ to 10⁻⁵ m/s
- Anisotropy ratios: 0.1 to 10

### Testing

Run the test suite:
```bash
python -m pytest tests/
```
