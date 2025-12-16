# Features Module

This module computes physical features from ice shelf velocity and viscosity data for use in k-means clustering. Features are derived from fundamental glaciological quantities and represent different aspects of ice flow dynamics.

## Overview

The features module transforms raw ice shelf data (velocities, viscosities, coordinates) into meaningful physical features that capture ice flow regimes. Different feature sets are available depending on analysis requirements.

## Key Components

### **feature_sets.py**
Central feature set definitions and computation pipeline.

**Key Functions:**
- `compute_primary_features()` - Recommended feature set for general analysis
- `compute_baseline_features()` - Minimal feature set for quick analysis
- `compute_extended_features()` - Comprehensive feature set with additional quantities
- `compute_stress_features()` - Stress-based features for mechanical analysis
- `get_feature_info()` - Metadata about feature definitions

**Feature Set Definitions:**
```python
# Primary (Recommended)
features = compute_primary_features(unified_data)
# Returns: [dudx, speed, mu, anisotropy]

# Baseline (Fast)
features = compute_baseline_features(unified_data) 
# Returns: [dudx, speed]

# Extended (Comprehensive)
features = compute_extended_features(unified_data)
# Returns: [dudx, speed, mu, anisotropy, effective_strain, divergence, h]

# Stress-based (Mechanical)
features = compute_stress_features(unified_data)
# Returns: [dudx, speed, deviatoric_stress, stress_anisotropy, stress_indicator]
```

### **strain_features.py**
Strain rate tensor computations and derived quantities.

**Key Functions:**
- `compute_strain_rate_tensor()` - Full strain rate tensor from velocity gradients
- `compute_effective_strain_rate()` - Von Mises equivalent strain rate
- `compute_velocity_divergence()` - ∇ · v (mass conservation)
- `compute_longitudinal_strain()` - ∂u/∂x (primary flow direction)
- `compute_shear_strain()` - Shear strain components

**Physical Definitions:**
```python
# Strain rate tensor: ε̇ᵢⱼ = ½(∂vᵢ/∂xⱼ + ∂vⱼ/∂xᵢ)
strain_tensor = compute_strain_rate_tensor(dudx, dudy, dvdx, dvdy)

# Effective strain rate: ε̇ₑ = √(ε̇ᵢⱼε̇ᵢⱼ) 
effective_strain = compute_effective_strain_rate(dudx, dudy, dvdx, dvdy)

# Velocity divergence: ∇ · v = ∂u/∂x + ∂v/∂y
divergence = compute_velocity_divergence(dudx, dvdy)
```

### **viscosity_features.py**
Viscosity-based features and rheological properties.

**Key Functions:**
- `compute_anisotropy_ratio()` - Viscosity anisotropy (η/μ)
- `compute_deviatoric_stress()` - Deviatoric stress magnitude
- `compute_stress_ratios()` - Stress anisotropy measures
- `compute_rheological_indicator()` - Flow regime indicators

**Rheological Features:**
```python
# Viscosity anisotropy: ratio of horizontal to vertical viscosity
anisotropy = compute_anisotropy_ratio(mu, eta)

# Deviatoric stress magnitude: τ = μ * ε̇
stress = compute_deviatoric_stress(mu, strain_rate)

# Stress anisotropy ratios
stress_ratios = compute_stress_ratios(stress_tensor, mu, eta)
```

## Feature Definitions

### **Primary Features** (Recommended)

#### **dudx** - Longitudinal Strain Rate
- **Definition**: ∂u/∂x (rate of stretching in flow direction)
- **Physical meaning**: Compression (negative) vs extension (positive)
- **Typical range**: -1×10⁻⁹ to +1×10⁻⁹ s⁻¹
- **Units**: s⁻¹

#### **speed** - Velocity Magnitude  
- **Definition**: √(u² + v²) 
- **Physical meaning**: Ice flow speed
- **Typical range**: 1×10⁻⁸ to 1×10⁻⁵ m/s (1-300 m/year)
- **Units**: m/s

#### **mu** - Horizontal Viscosity
- **Definition**: Effective viscosity in horizontal plane
- **Physical meaning**: Resistance to horizontal deformation
- **Typical range**: 1×10¹² to 1×10¹⁶ Pa·s
- **Units**: Pa·s

#### **anisotropy** - Viscosity Ratio
- **Definition**: η/μ (vertical/horizontal viscosity ratio)
- **Physical meaning**: Degree of flow anisotropy
- **Typical range**: 0.1 to 10
- **Units**: Dimensionless

### **Extended Features**

#### **effective_strain** - Von Mises Strain Rate
- **Definition**: √(½ε̇ᵢⱼε̇ᵢⱼ) 
- **Physical meaning**: Total deformation rate magnitude
- **Units**: s⁻¹

#### **divergence** - Velocity Divergence
- **Definition**: ∇ · v = ∂u/∂x + ∂v/∂y
- **Physical meaning**: Volume change rate (mass conservation)
- **Units**: s⁻¹

#### **h** - Ice Thickness
- **Definition**: Ice shelf thickness
- **Physical meaning**: Structural constraint on flow
- **Units**: m

### **Stress Features**

#### **deviatoric_stress** - Stress Magnitude
- **Definition**: |τ| where τᵢⱼ = μ(∂vᵢ/∂xⱼ + ∂vⱼ/∂xᵢ)
- **Physical meaning**: Magnitude of driving stress
- **Units**: Pa

#### **stress_anisotropy** - Stress Ratio
- **Definition**: Ratio of stress components
- **Physical meaning**: Stress field directionality
- **Units**: Dimensionless

## Usage Patterns

### **Standard Analysis**
```python
from features.feature_sets import compute_primary_features

# Compute recommended feature set
features = compute_primary_features(unified_data)

# Features is a dictionary with keys: ['dudx', 'speed', 'mu', 'anisotropy']
dudx = features['dudx']
speed = features['speed']
mu = features['mu'] 
anisotropy = features['anisotropy']
```

### **Custom Feature Selection**
```python
from features.strain_features import compute_longitudinal_strain
from features.viscosity_features import compute_anisotropy_ratio

# Compute individual features
longitudinal_strain = compute_longitudinal_strain(dudx)
viscosity_anisotropy = compute_anisotropy_ratio(mu, eta)

# Combine into custom feature set
custom_features = {
    'longitudinal_strain': longitudinal_strain,
    'anisotropy': viscosity_anisotropy
}
```

### **Feature Validation**
```python
from features.feature_sets import validate_features

# Validate computed features
validation_results = validate_features(features)
if not validation_results['valid']:
    print(f"Feature validation failed: {validation_results['errors']}")
```

## Physical Interpretation

### **Ice Flow Regimes**
Features are designed to distinguish three main ice shelf flow regimes:

#### **Compression Regime**
- **dudx < 0**: Negative longitudinal strain (compression)
- **High mu**: High viscosity (near grounding line)
- **Low anisotropy**: More isotropic flow
- **Location**: Near grounding line, lateral margins

#### **Transition Regime**
- **dudx ≈ 0**: Low strain rates
- **Moderate mu**: Intermediate viscosity
- **Variable anisotropy**: Transitional rheology
- **Location**: Rheological boundaries, flow transitions

#### **Extension Regime**
- **dudx > 0**: Positive longitudinal strain (extension)
- **Lower mu**: Reduced effective viscosity
- **Higher anisotropy**: More directional flow
- **Location**: Toward calving front, flow acceleration zones

### **Feature Scaling**
All features require scaling before clustering:
- **Standard scaling**: (x - μ)/σ (recommended)
- **Min-max scaling**: (x - min)/(max - min)
- **Robust scaling**: (x - median)/IQR

## Quality Control

### **Data Validation**
- Physical value range checks
- NaN and infinite value handling
- Grid consistency verification
- Units and scaling validation

### **Feature Quality Metrics**
- Feature variance and distribution checks
- Correlation analysis between features
- Outlier detection and handling
- Missing data statistics

### **Physical Consistency**
- Strain rate magnitude validation
- Viscosity range verification
- Stress-strain relationship checks
- Conservation law compliance

## Performance Optimization

### **Computational Efficiency**
- Vectorized numpy operations
- Memory-efficient array processing
- Minimal data copying
- Lazy evaluation where possible

### **Large Dataset Handling**
- Chunked processing for large grids
- Memory usage monitoring
- Progress tracking for long computations
- Efficient storage formats

## Dependencies

- **NumPy**: Array operations and mathematical functions
- **SciPy**: Advanced mathematical computations
- **warnings**: Data quality reporting