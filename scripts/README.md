# Scripts Directory

This directory contains the main execution scripts for the ice shelf classifier. These are the primary entry points for running the clustering analysis.

## Main Scripts

### 🎯 Core Analysis Scripts

#### `run_k_selection.py`
**Purpose**: Determine optimal number of clusters using elbow method
- **When to use**: Before running classification to find best k value
- **Input**: Processed dataset (.npz file)
- **Output**: K-selection analysis plots and metrics
- **Usage**:
```bash
python scripts/run_k_selection.py \
    --processed_data data/real_data_analysis/processed_dataset.npz \
    --output_dir results/ \
    --k_range 2 8
```

#### `run_optimized_kmeans.py`
**Purpose**: Fast k-means classification (recommended for large datasets)
- **When to use**: Main classification after determining optimal k
- **Features**: Optimized for performance, sampling-based silhouette computation
- **Input**: Processed dataset and known k value
- **Output**: Cluster assignments, centroids, summary metrics
- **Usage**:
```bash
python scripts/run_optimized_kmeans.py \
    --processed_data data/real_data_analysis/processed_dataset.npz \
    --output_dir results/ \
    --k 3
```

#### `run_kmeans.py`
**Purpose**: Complete pipeline with data processing and visualizations
- **When to use**: Full analysis from raw data with detailed outputs
- **Features**: Complete workflow, extensive visualizations
- **Input**: Raw DIFFUSE data + MATLAB viscosity files
- **Output**: Full analysis with plots and detailed metrics
- **Usage**:
```bash
python scripts/run_kmeans.py \
    --diffice_data data/DIFFICE_jax \
    --viscosity_data data/raw/results.mat \
    --output_dir results/ \
    --k 3 \
    --feature_set primary
```

### 📊 Specialized Analysis Scripts

#### `run_ablation.py`
**Purpose**: Feature importance analysis through ablation study
- **When to use**: To understand which features contribute most to clustering
- **Output**: Feature importance rankings and ablation results
- **Usage**:
```bash
python scripts/run_ablation.py \
    --processed_data data/real_data_analysis/processed_dataset.npz \
    --output_dir results/ablation/
```

#### `export_amery_from_diffice.py`
**Purpose**: Extract and preprocess Amery ice shelf data from DIFFUSE repository
- **When to use**: Initial data preparation from raw DIFFUSE files
- **Output**: Processed .npz files ready for analysis
- **Usage**:
```bash
python scripts/export_amery_from_diffice.py \
    --input_dir data/DIFFICE_jax \
    --output_dir data/processed/
```

## Script Selection Guide

### For First-Time Users:
1. **Start with**: `run_k_selection.py` to find optimal k
2. **Then use**: `run_optimized_kmeans.py` for fast classification
3. **For detailed analysis**: `run_kmeans.py` with visualizations

### For Large Datasets (>50k points):
- **Recommended**: `run_optimized_kmeans.py`
- **Avoid**: `run_kmeans.py` (may timeout on visualizations)

### For Research/Analysis:
- **Feature analysis**: `run_ablation.py`
- **Full pipeline**: `run_kmeans.py`
- **Parameter tuning**: `run_k_selection.py`

## Common Parameters

### Required Parameters
- `--processed_data`: Path to processed dataset (.npz file)
- `--output_dir`: Directory for results
- `--k`: Number of clusters (typically 3 for ice shelf regimes)

### Optional Parameters
- `--feature_set`: Feature combination (`primary`, `baseline`, `extended`, `stress`)
- `--scaling_method`: Feature scaling (`standard`, `minmax`, `robust`)
- `--analysis_only`: Skip visualizations (faster execution)
- `--max_silhouette_size`: Sample size for silhouette computation

## Output Structure

Each script creates organized output directories:
```
results/
├── clustering_results.npz        # Main results data
├── clustering_summary.json       # Metrics and analysis
├── clustering_summary.txt        # Human-readable summary
├── spatial_clusters.png          # Cluster map
├── spatial_features.png          # Feature distributions
└── feature_space_*.png          # Feature space visualizations
```

## Performance Notes

### Typical Execution Times
- **K-selection**: 1-5 minutes (depends on k_range)
- **Optimized k-means**: 10-60 seconds (large datasets)
- **Full pipeline**: 2-10 minutes (includes visualizations)

### Memory Requirements
- **Small datasets** (<10k points): 2GB RAM
- **Medium datasets** (10-50k points): 4GB RAM  
- **Large datasets** (>50k points): 8GB RAM

### Troubleshooting
- **Timeout issues**: Use `run_optimized_kmeans.py`
- **Memory errors**: Reduce dataset size or use `--analysis_only`
- **File not found**: Check data paths are absolute or relative to project root