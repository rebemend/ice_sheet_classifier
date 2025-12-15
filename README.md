# Ice Shelf Classifier

A machine learning pipeline for classifying ice shelf flow regimes using k-means clustering on velocity, strain rate, and viscosity data.

## Overview

This repository implements an unsupervised classification system that identifies three main ice shelf flow regimes:
- **Compression**: Areas with negative longitudinal strain (near grounding line)
- **Transition**: Low strain rate regions (rheological boundaries) 
- **Extension**: Areas with positive longitudinal strain (toward calving front)

The classifier uses k-means clustering on physical features derived from ice shelf velocity and viscosity fields.

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy  
- scikit-learn
- matplotlib (for visualizations)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ice_sheet_classifier

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage with Preprocessed Data

If you already have processed data:

```bash
python scripts/run_optimized_kmeans.py \
    --processed_data real_data_analysis/processed_dataset.npz \
    --output_dir results/ \
    --k 3
```

### 2. Full Pipeline from Raw Data

For raw DIFFUSE data and MATLAB results:

```bash
python scripts/run_kmeans.py \
    --diffice_data data/DIFFICE_jax \
    --viscosity_data data/raw/results.mat \
    --output_dir results/ \
    --k 3 \
    --feature_set primary \
    --analysis_only
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

### Core Scripts

- **`scripts/run_optimized_kmeans.py`**: Main optimized clustering script (recommended for large datasets)
- **`scripts/run_kmeans.py`**: Full analysis pipeline with visualizations
- **`scripts/run_k_selection.py`**: Determine optimal number of clusters
- **`scripts/run_ablation.py`**: Feature importance analysis

### Data Processing

- **`src/data_loading/`**: Data loading and preprocessing modules
- **`src/features/`**: Feature computation and selection
- **`src/utils/`**: Scaling and validation utilities

### Analysis & Visualization

- **`src/clustering/`**: K-means implementation and analysis
- **`src/visualization/`**: Spatial maps and feature space plots

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

## Development

### Repository Structure
```
ice_sheet_classifier/
├── src/                    # Source code
│   ├── data_loading/      # Data I/O and preprocessing
│   ├── features/          # Feature computation
│   ├── clustering/        # K-means algorithms  
│   ├── utils/            # Utilities and validation
│   └── visualization/    # Plotting and analysis
├── scripts/              # Main execution scripts
├── tests/               # Unit tests
├── data/                # Input data directory
└── docs/                # Additional documentation
```

### Testing

Run the test suite:
```bash
python -m pytest tests/
```

### Contributing

1. Follow existing code style and structure
2. Add tests for new features
3. Update documentation
4. Use meaningful commit messages

## References

- DIFFUSE_jax repository for ice shelf modeling
- Sergienko & Hulbe (2011) for ice shelf rheology
- Arthur & Vassilvitskii (2007) for k-means++ initialization

## License

[Add appropriate license information]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ice_shelf_classifier,
  title={Ice Shelf Flow Regime Classifier},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]}
}
```
