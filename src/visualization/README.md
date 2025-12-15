# Visualization Module

This module provides comprehensive visualization capabilities for ice shelf clustering analysis, including spatial maps, feature space analysis, and diagnostic plots.

## Overview

The visualization module creates publication-quality plots for interpreting clustering results and understanding ice shelf flow regimes. It handles both spatial (geographic) visualizations and feature space analysis plots.

## Key Components

### **spatial_maps.py**
Geographic visualizations of clustering results overlaid on ice shelf spatial data.

**Key Functions:**
- `plot_cluster_map()` - Main cluster assignment visualization
- `plot_feature_distributions()` - Spatial distribution of individual features
- `plot_regime_interpretation()` - Physical interpretation of clusters
- `create_spatial_analysis()` - Complete spatial visualization suite

**Spatial Visualizations:**
```python
from visualization.spatial_maps import plot_cluster_map, plot_feature_distributions

# Plot cluster assignments on geographic map
plot_cluster_map(
    labels=cluster_labels,
    coordinates=coordinates,
    title="Ice Shelf Flow Regimes",
    output_path="results/spatial_clusters.png"
)

# Show feature distributions spatially
plot_feature_distributions(
    features=feature_dict,
    coordinates=coordinates,
    output_path="results/spatial_features.png"
)
```

### **feature_space.py**
Feature space analysis and dimensionality reduction visualizations.

**Key Functions:**
- `plot_feature_space_pca()` - Principal Component Analysis visualization
- `plot_feature_space_tsne()` - t-SNE dimensionality reduction
- `plot_feature_pairs()` - Pairwise feature scatter plots
- `plot_silhouette_analysis()` - Silhouette score visualization
- `plot_cluster_centroids()` - Cluster center analysis

**Feature Space Plots:**
```python
from visualization.feature_space import plot_feature_space_pca, plot_silhouette_analysis

# PCA visualization of clusters in feature space
plot_feature_space_pca(
    features=scaled_features,
    labels=cluster_labels,
    feature_names=['dudx', 'speed', 'mu', 'anisotropy'],
    output_path="results/feature_space_pca.png"
)

# Silhouette analysis for cluster quality assessment
plot_silhouette_analysis(
    features=scaled_features,
    labels=cluster_labels,
    output_path="results/silhouette_analysis.png"
)
```

## Spatial Visualizations

### **Cluster Maps**
Geographic visualization of cluster assignments showing flow regime spatial distribution.

**Features:**
- **Color-coded clusters**: Different colors for each flow regime
- **Coordinate projection**: Proper handling of polar stereographic coordinates
- **Physical boundaries**: Ice shelf margins and grounding lines
- **Scale bars and legends**: Publication-ready formatting

```python
# Basic cluster map
plot_cluster_map(
    labels=labels,
    coordinates=coords,
    cluster_names=['Compression', 'Transition', 'Extension'],
    colormap='viridis',
    output_path='cluster_map.png'
)

# Advanced cluster map with customization
plot_cluster_map(
    labels=labels,
    coordinates=coords,
    cluster_names=['Compression', 'Transition', 'Extension'],
    colormap=['red', 'yellow', 'blue'],
    title='Amery Ice Shelf Flow Regimes',
    figsize=(12, 8),
    dpi=300,
    show_coordinates=True,
    add_scalebar=True,
    output_path='detailed_cluster_map.png'
)
```

### **Feature Distribution Maps**
Spatial visualization of individual feature values across the ice shelf.

```python
# Show all primary features spatially
plot_feature_distributions(
    features={
        'dudx': longitudinal_strain,
        'speed': velocity_magnitude, 
        'mu': viscosity,
        'anisotropy': viscosity_ratio
    },
    coordinates=coordinates,
    output_path='feature_distributions.png'
)
```

### **Physical Regime Interpretation**
Visualization that connects clusters to physical flow processes.

```python
# Physical interpretation overlay
plot_regime_interpretation(
    labels=labels,
    coordinates=coordinates,
    regime_descriptions={
        0: 'Compression (Near Grounding Line)',
        1: 'Transition (Rheological Boundary)', 
        2: 'Extension (Toward Calving Front)'
    },
    output_path='regime_interpretation.png'
)
```

## Feature Space Analysis

### **Principal Component Analysis (PCA)**
Dimensionality reduction to visualize clusters in 2D feature space.

```python
# PCA with cluster overlay
plot_feature_space_pca(
    features=scaled_features,
    labels=cluster_labels,
    feature_names=['dudx', 'speed', 'mu', 'anisotropy'],
    n_components=2,
    output_path='pca_analysis.png'
)
```

**PCA Interpretation:**
- **PC1 vs PC2**: Most informative 2D projection of feature space
- **Cluster separation**: How well clusters separate in reduced space
- **Feature loadings**: Which original features contribute most to each PC
- **Variance explained**: How much information is retained in 2D

### **t-SNE Analysis**
Non-linear dimensionality reduction for complex feature relationships.

```python
# t-SNE visualization
plot_feature_space_tsne(
    features=scaled_features,
    labels=cluster_labels,
    perplexity=50,
    random_state=42,
    output_path='tsne_analysis.png'
)
```

**t-SNE Advantages:**
- **Non-linear relationships**: Captures complex feature interactions
- **Cluster structure**: Often reveals better cluster separation than PCA
- **Local structure preservation**: Maintains neighborhood relationships

### **Feature Pair Analysis**
Pairwise scatter plots showing relationships between all feature combinations.

```python
# Pairwise feature relationships
plot_feature_pairs(
    features=scaled_features,
    labels=cluster_labels,
    feature_names=['dudx', 'speed', 'mu', 'anisotropy'],
    output_path='feature_pairs.png'
)
```

**Insights from Pair Plots:**
- **Feature correlations**: Linear/non-linear relationships between features
- **Cluster boundaries**: How features separate different regimes
- **Outlier identification**: Points that don't fit typical patterns

### **Silhouette Analysis**
Detailed visualization of cluster quality using silhouette scores.

```python
# Comprehensive silhouette analysis
plot_silhouette_analysis(
    features=scaled_features,
    labels=cluster_labels,
    cluster_names=['Compression', 'Transition', 'Extension'],
    output_path='silhouette_analysis.png'
)
```

**Silhouette Plot Components:**
- **Per-cluster silhouette distributions**: Quality of each cluster
- **Average silhouette line**: Overall clustering quality reference
- **Cluster thickness**: Relative cluster sizes
- **Score interpretation**: Values near 1 = well-separated, near 0 = overlapping

### **Centroid Analysis**
Visualization of cluster centers in feature space.

```python
# Cluster centroid comparison
plot_cluster_centroids(
    centroids=kmeans_result['centroids'],
    feature_names=['dudx', 'speed', 'mu', 'anisotropy'],
    cluster_names=['Compression', 'Transition', 'Extension'],
    output_path='centroid_analysis.png'
)
```

## Visualization Workflows

### **Complete Analysis Suite**
Generate all visualizations for a clustering result:

```python
from visualization.spatial_maps import create_spatial_analysis
from visualization.feature_space import create_feature_analysis

# Generate all spatial visualizations
create_spatial_analysis(
    labels=labels,
    features=features,
    coordinates=coordinates,
    output_dir='results/spatial/'
)

# Generate all feature space visualizations  
create_feature_analysis(
    features=scaled_features,
    labels=labels,
    centroids=centroids,
    feature_names=feature_names,
    output_dir='results/feature_space/'
)
```

### **Publication-Quality Plots**
Settings for high-quality publication figures:

```python
# High-resolution publication settings
plot_cluster_map(
    labels=labels,
    coordinates=coordinates,
    figsize=(10, 8),        # Large figure size
    dpi=300,                # High resolution
    font_size=12,           # Readable fonts
    colormap='Set1',        # Distinct colors
    add_scalebar=True,      # Geographic reference
    save_format='pdf',      # Vector format
    output_path='publication_map.pdf'
)
```

### **Interactive Analysis**
For exploratory data analysis:

```python
# Interactive mode (show plots instead of saving)
plot_feature_space_pca(
    features=scaled_features,
    labels=labels,
    feature_names=feature_names,
    interactive=True,       # Show instead of save
    hover_data=True        # Add hover information
)
```

## Customization Options

### **Color Schemes**
```python
# Predefined colormaps
colormaps = {
    'physical': ['blue', 'green', 'red'],      # Physical interpretation
    'qualitative': 'Set1',                     # Distinct colors
    'sequential': 'viridis',                   # Smooth gradients
    'diverging': 'RdBu'                        # Centered colormaps
}

# Custom colors for specific regimes
regime_colors = {
    'Compression': '#1f77b4',     # Blue
    'Transition': '#ff7f0e',      # Orange  
    'Extension': '#2ca02c'        # Green
}
```

### **Plot Styling**
```python
# Consistent styling across all plots
plot_style = {
    'figsize': (10, 8),
    'dpi': 150,
    'font_size': 11,
    'line_width': 1.5,
    'marker_size': 6,
    'alpha': 0.7,
    'grid': True,
    'spines': 'off'
}
```

### **Geographic Projections**
```python
# Different coordinate system handling
projection_options = {
    'polar_stereo': True,          # Antarctic polar stereographic
    'utm': False,                  # Universal Transverse Mercator
    'geographic': False,           # Lat/lon coordinates
    'custom_proj': None            # Custom projection string
}
```

## Output Formats

### **Supported Formats**
- **PNG**: Raster format, good for web/presentations
- **PDF**: Vector format, publication quality
- **SVG**: Vector format, editable
- **JPG**: Compressed raster format
- **EPS**: Encapsulated PostScript for publications

### **Resolution Settings**
```python
output_settings = {
    'web': {'dpi': 72, 'format': 'png'},
    'presentation': {'dpi': 150, 'format': 'png'},
    'publication': {'dpi': 300, 'format': 'pdf'},
    'poster': {'dpi': 300, 'format': 'png', 'figsize': (16, 12)}
}
```

## Performance Considerations

### **Large Dataset Optimization**
- **Point sampling**: Reduce points for scatter plots while preserving patterns
- **Binned visualizations**: Use hexbin or 2D histograms for dense point clouds
- **Progressive rendering**: Show low-resolution preview first
- **Memory management**: Process data in chunks for very large datasets

### **Rendering Speed**
- **Rasterization**: Convert complex vector graphics to raster for speed
- **Level-of-detail**: Reduce detail based on zoom level
- **Cached computations**: Store expensive calculations (PCA, t-SNE)
- **Parallel processing**: Use multiprocessing for multiple plots

## Quality Assurance

### **Validation Checks**
- **Data integrity**: Ensure labels and coordinates match
- **Color accessibility**: Check color schemes for colorblind accessibility
- **Geographic accuracy**: Validate coordinate transformations
- **Scale consistency**: Ensure consistent scaling across related plots

### **Best Practices**
- **Clear legends**: Always include informative legends and colorbars
- **Axis labels**: Proper units and descriptions
- **Title information**: Descriptive titles with key parameters
- **Consistent styling**: Uniform appearance across related figures

## Dependencies

- **matplotlib**: Core plotting functionality
- **seaborn**: Statistical visualization enhancements  
- **numpy**: Array operations and data handling
- **scipy**: Scientific computing functions
- **scikit-learn**: PCA, t-SNE implementations
- **cartopy**: Geographic projections (optional)
- **geopandas**: Geographic data handling (optional)