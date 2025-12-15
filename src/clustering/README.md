# Clustering Module

This module implements k-means clustering algorithms optimized for ice shelf flow regime classification, with performance optimizations for large datasets and specialized analysis tools.

## Overview

The clustering module provides the core machine learning functionality for the ice shelf classifier. It includes an optimized k-means implementation with timeout protection, k-value selection methods, and feature importance analysis through ablation studies.

## Key Components

### **kmeans_runner.py** ⭐
Main clustering engine with performance optimizations for large ice shelf datasets.

**Key Features:**
- **Sampling-based silhouette computation** for datasets >10k points (fixes timeout issue)
- **Timeout protection** prevents hanging on large datasets  
- **Multiple k-value analysis** with elbow method and silhouette analysis
- **Comprehensive metrics** including cluster balance and convergence analysis

**Key Functions:**
```python
from clustering.kmeans_runner import KMeansRunner

# Initialize runner with optimizations
runner = KMeansRunner(
    random_state=42,
    max_iter=300,
    silhouette_timeout=60,
    silhouette_sample_size=5000
)

# Single k-means run
result = runner.run_single_kmeans(features, n_clusters=3)

# Multiple k analysis  
k_results = runner.run_k_range(features, k_range=[2, 3, 4, 5, 6])

# Find optimal k
optimal_k, selection_info = runner.find_optimal_k(k_results, method='combined')
```

**Performance Optimizations:**
- **Large dataset handling**: Automatically samples 5,000 points for silhouette computation when dataset >10,000 points
- **Timeout protection**: Prevents hanging on expensive operations
- **Memory efficiency**: Minimal data copying, efficient array operations
- **Progress reporting**: Real-time feedback on long-running operations

### **k_selection.py**
K-value selection using elbow method and silhouette analysis.

**Key Functions:**
- `run_k_selection_analysis()` - Complete k-selection pipeline
- `compute_elbow_scores()` - Elbow method implementation
- `compute_silhouette_scores()` - Silhouette analysis across k values
- `plot_k_selection_results()` - Visualization of k-selection metrics

**Selection Methods:**
```python
from clustering.k_selection import run_k_selection_analysis

# Complete k-selection analysis
results = run_k_selection_analysis(
    features=scaled_features,
    k_range=range(2, 9),
    method='combined',  # 'elbow', 'silhouette', or 'combined'
    output_dir='results/'
)

# Extract recommended k
optimal_k = results['optimal_k']
```

### **ablation.py**
Feature importance analysis through systematic feature removal.

**Key Functions:**
- `run_feature_ablation()` - Complete ablation study
- `compute_ablation_metrics()` - Performance impact of feature removal
- `analyze_feature_importance()` - Ranking and importance scores
- `suggest_minimal_features()` - Reduced feature set recommendations

**Ablation Analysis:**
```python
from clustering.ablation import run_feature_ablation

# Complete ablation study
ablation_results = run_feature_ablation(
    features=feature_dict,
    feature_names=['dudx', 'speed', 'mu', 'anisotropy'],
    n_clusters=3,
    output_dir='results/ablation/'
)

# Feature importance ranking
importance = ablation_results['feature_importance']
```

## Core Algorithms

### **K-Means Implementation**
Enhanced scikit-learn k-means with ice shelf-specific optimizations:

**Algorithm Settings:**
- **Initialization**: k-means++ (default) for better cluster initialization
- **n_init**: 10 runs with different initializations (for stability)
- **max_iter**: 300 iterations (sufficient for convergence)
- **tol**: 1e-4 convergence tolerance

**Optimization Features:**
- **Silhouette sampling**: For datasets >10k points, samples 5k points for silhouette computation
- **Timeout handling**: Graceful handling of expensive operations
- **Memory management**: Efficient memory usage for large datasets
- **Progress tracking**: Real-time updates on clustering progress

### **K-Selection Methods**

#### **Elbow Method**
Finds the "elbow" in the inertia vs k curve using second derivative analysis.
```python
# Inertia (within-cluster sum of squares)
inertias = []
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

# Find elbow using second derivative
elbow_k = find_elbow_point(k_range, inertias)
```

#### **Silhouette Method**
Selects k that maximizes average silhouette score.
```python
# Silhouette scores
silhouettes = []
for k in k_range:
    labels = KMeans(n_clusters=k).fit_predict(features)
    score = silhouette_score(features, labels)
    silhouettes.append(score)

# Select k with maximum silhouette
optimal_k = k_range[np.argmax(silhouettes)]
```

#### **Combined Method**
Integrates elbow and silhouette methods for robust k-selection.

## Performance Characteristics

### **Scalability**
| Dataset Size | K-means Time | Silhouette Time | Memory Usage |
|--------------|-------------|-----------------|--------------|
| 1,000 points | 0.02s | 0.01s | 50 MB |
| 10,000 points | 0.06s | 0.5s | 200 MB |
| 50,000 points | 0.12s | **12s → 0.15s*** | 1 GB |
| 100,000 points | 0.20s | **50s → 0.15s*** | 2 GB |
| 200,000 points | 0.40s | **>60s → 0.15s*** | 4 GB |

*\*With sampling optimization*

### **Timeout Protection**
The module includes comprehensive timeout protection:
- **Silhouette computation**: Automatic sampling for large datasets
- **K-range analysis**: Progress tracking and early termination options
- **Feature ablation**: Configurable timeouts for extensive analysis

## Usage Patterns

### **Standard Clustering**
```python
from clustering.kmeans_runner import KMeansRunner

# Basic clustering
runner = KMeansRunner(random_state=42)
result = runner.run_single_kmeans(scaled_features, n_clusters=3)

# Extract results
labels = result['labels']
centroids = result['centroids']
silhouette_score = result['silhouette_avg']
```

### **K-Selection Workflow**
```python
from clustering.k_selection import run_k_selection_analysis

# Determine optimal k
k_analysis = run_k_selection_analysis(
    features=scaled_features,
    k_range=range(2, 8),
    output_dir='results/k_selection/'
)

# Use recommended k for final clustering
optimal_k = k_analysis['optimal_k']
```

### **Feature Importance Analysis**
```python
from clustering.ablation import run_feature_ablation

# Analyze feature importance
ablation = run_feature_ablation(
    features=feature_dict,
    feature_names=['dudx', 'speed', 'mu', 'anisotropy'],
    n_clusters=3
)

# Get feature rankings
importance_ranking = ablation['importance_ranking']
```

### **Large Dataset Handling**
```python
# For datasets > 100k points
runner = KMeansRunner(
    max_iter=100,  # Reduce iterations for speed
    silhouette_sample_size=2000  # Smaller sample for even faster silhouette
)

# Run with analysis_only mode (skip expensive visualizations)
result = runner.run_single_kmeans(large_dataset, n_clusters=3)
```

## Result Structures

### **Single K-means Result**
```python
result = {
    'n_clusters': 3,
    'labels': array([0, 1, 2, ...]),           # Cluster assignments
    'centroids': array([[...], [...], [...]]), # Cluster centers
    'inertia': 1234.56,                        # Within-cluster sum of squares
    'silhouette_avg': 0.45,                    # Average silhouette score
    'silhouette_samples': array([...]),        # Per-point silhouette scores
    'cluster_sizes': array([1000, 800, 1200]), # Points per cluster
    'converged': True,                         # Convergence status
    'n_iter': 15                               # Iterations to convergence
}
```

### **K-Selection Result**
```python
k_analysis = {
    'optimal_k': 3,
    'method': 'combined',
    'k_range': [2, 3, 4, 5, 6],
    'inertias': array([...]),
    'silhouettes': array([...]),
    'elbow_k': 3,
    'silhouette_k': 3,
    'selection_reasoning': 'Both methods agree on k=3'
}
```

### **Ablation Result**
```python
ablation = {
    'feature_importance': {
        'dudx': 0.85,      # Most important
        'mu': 0.72,
        'anisotropy': 0.43,
        'speed': 0.21      # Least important
    },
    'importance_ranking': ['dudx', 'mu', 'anisotropy', 'speed'],
    'minimal_features': ['dudx', 'mu'],  # Sufficient for good clustering
    'ablation_results': {...}  # Detailed per-feature results
}
```

## Quality Metrics

### **Clustering Quality**
- **Silhouette Score**: Measures cluster separation (0 to 1, higher better)
- **Inertia**: Within-cluster sum of squares (lower better)
- **Cluster Balance**: Distribution of points across clusters
- **Convergence**: Whether algorithm converged within max_iter

### **Physical Interpretation**
- **Regime Identification**: How well clusters correspond to physical flow regimes
- **Spatial Coherence**: Geographic continuity of cluster assignments  
- **Feature Significance**: Physical meaningfulness of cluster centers

### **Validation Checks**
- **Cluster Stability**: Consistency across multiple runs
- **Feature Scaling**: Proper normalization of input features
- **Outlier Impact**: Robustness to extreme values

## Configuration Options

### **KMeansRunner Parameters**
```python
runner = KMeansRunner(
    random_state=42,              # Reproducibility
    max_iter=300,                 # Maximum iterations
    silhouette_timeout=60,        # Silhouette computation timeout (seconds)
    silhouette_sample_size=5000   # Sample size for large datasets
)
```

### **Analysis Options**
- **K-selection methods**: 'elbow', 'silhouette', 'combined'
- **Feature sets**: Different combinations for ablation analysis
- **Sampling strategies**: For handling large datasets efficiently

## Error Handling

### **Common Issues**
1. **Timeout on large datasets**: Automatic sampling kicks in
2. **Convergence failures**: Increased max_iter or different initialization
3. **Memory errors**: Reduced sample sizes or chunked processing
4. **Numerical instabilities**: Feature scaling and validation

### **Debugging Tools**
- **Verbose logging**: Detailed progress and timing information
- **Convergence monitoring**: Track algorithm convergence
- **Quality checks**: Validate results and identify issues
- **Performance profiling**: Identify bottlenecks in processing

## Dependencies

- **scikit-learn**: K-means implementation and metrics
- **NumPy**: Array operations and mathematical functions
- **matplotlib**: Visualization of results (optional)
- **warnings**: Issue reporting and user feedback