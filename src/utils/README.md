# Utils Module

This module provides utility functions for feature scaling, data validation, and supporting operations used throughout the ice shelf classifier pipeline.

## Overview

The utils module contains helper functions and utilities that support the main analysis pipeline. Currently focused on feature scaling and normalization, it can be extended with additional utilities as needed.

## Key Components

### **scaling.py**
Feature scaling and normalization functions for preprocessing data before clustering.

**Key Functions:**
- `scale_features_for_clustering()` - Main scaling function with multiple methods
- `FeatureScaler` - Class-based scaler with fit/transform pattern
- `get_scaling_summary()` - Statistics about scaling transformations
- `validate_scaled_features()` - Quality checks for scaled data

## Feature Scaling

### **Why Scaling is Critical**
K-means clustering is sensitive to feature scales because it uses Euclidean distance. Without proper scaling:
- Features with larger numerical ranges dominate the distance calculation
- Physical features have vastly different scales (velocity: 10⁻⁸ m/s, viscosity: 10¹⁵ Pa·s)
- Clustering results become biased toward high-magnitude features

### **Scaling Methods**

#### **Standard Scaling** (Recommended)
Transforms features to have zero mean and unit variance: `(x - μ)/σ`

**Advantages:**
- Preserves the shape of the original distribution
- Handles outliers reasonably well
- Most commonly used and well-understood
- Works well with normal/Gaussian-like distributions

**Usage:**
```python
from utils.scaling import scale_features_for_clustering

scaled_features = scale_features_for_clustering(
    features, 
    method='standard'
)
```

#### **Min-Max Scaling**
Transforms features to a fixed range [0, 1]: `(x - min)/(max - min)`

**Advantages:**
- Bounded output range
- Preserves zero values
- Good for features with known bounds

**Disadvantages:**
- Sensitive to outliers
- May compress most data if outliers are present

**Usage:**
```python
scaled_features = scale_features_for_clustering(
    features, 
    method='minmax'
)
```

#### **Robust Scaling**
Uses median and interquartile range: `(x - median)/IQR`

**Advantages:**
- Robust to outliers
- Good for skewed distributions
- Less affected by extreme values

**Disadvantages:**
- May not center data at zero
- Less intuitive interpretation

**Usage:**
```python
scaled_features = scale_features_for_clustering(
    features, 
    method='robust'
)
```

### **Class-Based Scaling**
For more control and to maintain scaling parameters:

```python
from utils.scaling import FeatureScaler

# Initialize and fit scaler
scaler = FeatureScaler(method='standard')
scaler.fit(training_features)

# Transform data
scaled_features = scaler.transform(features)

# Get scaling statistics
stats = scaler.get_scaling_stats()
```

## Usage Patterns

### **Basic Feature Scaling**
```python
import numpy as np
from utils.scaling import scale_features_for_clustering

# Example ice shelf features (different scales)
features = {
    'dudx': np.array([1e-10, 2e-10, -1e-10, ...]),      # strain rate [s⁻¹]
    'speed': np.array([1e-7, 5e-7, 2e-7, ...]),         # velocity [m/s] 
    'mu': np.array([1e14, 5e14, 2e14, ...]),            # viscosity [Pa·s]
    'anisotropy': np.array([0.5, 2.1, 1.2, ...])        # ratio [dimensionless]
}

# Scale for clustering
scaled_features = scale_features_for_clustering(features, method='standard')

# Result: all features have mean ≈ 0, std ≈ 1
```

### **Feature Array Preparation**
```python
# Convert feature dictionary to array for clustering
feature_names = ['dudx', 'speed', 'mu', 'anisotropy']
feature_array = np.column_stack([
    scaled_features[name] for name in feature_names
])

# Now ready for k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(feature_array)
```

### **Scaling with Validation**
```python
from utils.scaling import scale_features_for_clustering, validate_scaled_features

# Scale features
scaled_features = scale_features_for_clustering(features)

# Validate scaling quality
validation = validate_scaled_features(scaled_features)
if not validation['valid']:
    print(f"Scaling issues: {validation['warnings']}")
```

### **Advanced Scaling Control**
```python
from utils.scaling import FeatureScaler

# Initialize scaler with custom parameters
scaler = FeatureScaler(
    method='standard',
    handle_constant_features='warn',  # Warn about constant features
    clip_outliers=True,               # Clip extreme outliers
    outlier_threshold=3.0             # 3-sigma outlier threshold
)

# Fit and transform
scaler.fit(features)
scaled_features = scaler.transform(features)

# Get detailed statistics
stats = scaler.get_scaling_stats()
print(f"Feature means: {stats['means']}")
print(f"Feature stds: {stats['stds']}")
print(f"Outliers clipped: {stats['outliers_clipped']}")
```

## Data Validation

### **Scaling Quality Checks**
The module includes comprehensive validation of scaled features:

```python
validation_results = validate_scaled_features(scaled_features)

# Check results
if validation_results['valid']:
    print("Scaling successful!")
else:
    print(f"Issues found: {validation_results['errors']}")
    print(f"Warnings: {validation_results['warnings']}")
```

**Validation Checks:**
- **NaN/Inf detection**: Identifies invalid values after scaling
- **Scale consistency**: Ensures all features have similar scales
- **Distribution checks**: Warns about highly skewed distributions
- **Outlier detection**: Identifies extreme values that may affect clustering

### **Input Validation**
Before scaling, the module validates input data:

```python
# Automatic validation in scaling functions
try:
    scaled_features = scale_features_for_clustering(features)
except ValueError as e:
    print(f"Input validation failed: {e}")
```

**Input Checks:**
- Feature completeness (no missing features)
- Data type consistency (numeric arrays)
- Shape compatibility across features
- Physical value range validation

## Scaling Statistics

### **Get Scaling Summary**
```python
from utils.scaling import get_scaling_summary

summary = get_scaling_summary(original_features, scaled_features)

print(f"Scaling method: {summary['method']}")
print(f"Features scaled: {summary['n_features']}")
print(f"Data points: {summary['n_samples']}")

# Per-feature statistics
for feature_name in summary['features']:
    stats = summary['features'][feature_name]
    print(f"{feature_name}:")
    print(f"  Original range: [{stats['original_min']:.2e}, {stats['original_max']:.2e}]")
    print(f"  Scaled range: [{stats['scaled_min']:.2f}, {stats['scaled_max']:.2f}]")
    print(f"  Scale factor: {stats['scale_factor']:.2e}")
```

### **Feature Scale Comparison**
```python
# Compare feature scales before and after scaling
def compare_feature_scales(original, scaled):
    for name in original.keys():
        orig_range = np.ptp(original[name])  # peak-to-peak
        scaled_range = np.ptp(scaled[name])
        print(f"{name}: {orig_range:.2e} → {scaled_range:.2f}")

compare_feature_scales(features, scaled_features)
```

## Configuration Options

### **Scaling Parameters**
```python
scaled_features = scale_features_for_clustering(
    features,
    method='standard',           # 'standard', 'minmax', 'robust'
    handle_nans='remove',        # 'remove', 'interpolate', 'raise'
    clip_outliers=False,         # Clip extreme outliers
    outlier_threshold=3.0,       # Sigma threshold for outliers
    validate_input=True,         # Perform input validation
    return_scaler=False          # Return scaler object
)
```

### **FeatureScaler Configuration**
```python
scaler = FeatureScaler(
    method='standard',
    copy=True,                   # Don't modify input data
    handle_constant_features='warn',  # 'ignore', 'warn', 'raise'
    feature_range=(0, 1),        # For minmax scaling
    quantile_range=(25.0, 75.0)  # For robust scaling
)
```

## Physical Feature Considerations

### **Typical Ice Shelf Feature Ranges**
Understanding the physical ranges helps choose appropriate scaling:

| Feature | Typical Range | Units | Scale Factor |
|---------|---------------|-------|--------------|
| dudx (strain rate) | -1×10⁻⁹ to +1×10⁻⁹ | s⁻¹ | ~10⁻⁹ |
| speed (velocity) | 1×10⁻⁸ to 1×10⁻⁵ | m/s | ~10⁻⁶ |
| mu (viscosity) | 1×10¹² to 1×10¹⁶ | Pa·s | ~10¹⁴ |
| anisotropy (ratio) | 0.1 to 10 | - | ~1 |

**Without scaling**: viscosity dominates clustering due to its large magnitude (~10¹⁴)
**With scaling**: all features contribute equally to clustering decisions

### **Physical Interpretation After Scaling**
- **Scaled values near 0**: Close to typical/mean values for that feature
- **Scaled values > 2**: Unusually high values (>2σ above mean)
- **Scaled values < -2**: Unusually low values (>2σ below mean)
- **Extreme values (|scaled| > 3)**: Potential outliers requiring investigation

## Performance Considerations

### **Memory Efficiency**
- **In-place operations**: Option to modify input arrays directly
- **Chunked processing**: For very large datasets
- **Lazy evaluation**: Compute statistics only when needed

### **Computational Complexity**
- **Scaling**: O(n·f) where n = samples, f = features
- **Validation**: O(n·f) for quality checks
- **Statistics**: O(n·f) for summary computations

## Error Handling

### **Common Issues**
1. **Constant features**: Features with zero variance (can't be scaled)
2. **Missing data**: NaN values in input features
3. **Infinite values**: Result from division by zero or overflow
4. **Incompatible shapes**: Mismatched feature array dimensions

### **Error Recovery**
- **Automatic handling**: Remove or interpolate missing values
- **Graceful degradation**: Skip problematic features with warnings
- **Detailed diagnostics**: Identify specific issues and solutions

## Dependencies

- **NumPy**: Array operations and statistical functions
- **scikit-learn**: Scaling implementations (StandardScaler, MinMaxScaler, RobustScaler)
- **warnings**: Issue reporting to users