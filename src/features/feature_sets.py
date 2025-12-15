import numpy as np
from typing import Dict, List, Tuple, Optional
from .strain_features import (
    compute_strain_rate_tensor, compute_velocity_features,
    compute_deformation_features, validate_strain_features
)
from .viscosity_features import (
    compute_viscosity_features, compute_stress_features,
    validate_viscosity_features
)


class FeatureSetDefinitions:
    """
    Define standard feature sets for ice shelf classification.
    
    This class centralizes the definition of different feature combinations
    used for k-means clustering analysis.
    """
    
    @staticmethod
    def get_baseline_features() -> List[str]:
        """
        Core deformation features (Baseline configuration).
        
        Returns
        -------
        List[str]
            Feature names for baseline clustering
        """
        return ['dudx', 'speed']
    
    @staticmethod
    def get_primary_features() -> List[str]:
        """
        Velocity + Viscosity features (Primary configuration).
        
        Returns
        -------
        List[str]
            Feature names for primary clustering
        """
        return ['dudx', 'speed', 'mu', 'anisotropy']
    
    @staticmethod
    def get_extended_features() -> List[str]:
        """
        Extended physics features (Diagnostic configuration).
        
        Returns
        -------
        List[str]
            Feature names for extended clustering
        """
        return [
            'dudx', 'speed', 'mu', 'anisotropy',
            'effective_strain', 'divergence', 'h'
        ]
    
    @staticmethod
    def get_stress_features() -> List[str]:
        """
        Stress-based features.
        
        Returns
        -------
        List[str]
            Feature names for stress-based clustering
        """
        return [
            'dudx', 'speed', 'deviatoric_stress', 
            'stress_anisotropy', 'stress_indicator'
        ]
    
    @staticmethod
    def get_ablation_features() -> Dict[str, List[str]]:
        """
        Feature sets for ablation study.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping ablation test names to feature lists
        """
        primary = FeatureSetDefinitions.get_primary_features()
        
        return {
            'no_strain': ['speed', 'mu', 'anisotropy'],
            'no_velocity': ['dudx', 'mu', 'anisotropy'],
            'no_viscosity': ['dudx', 'speed'],
            'no_anisotropy': ['dudx', 'speed', 'mu'],
            'full': primary
        }


def extract_features_from_data(unified_data: Dict[str, np.ndarray],
                              feature_names: List[str]) -> np.ndarray:
    """
    Extract specified features from unified dataset.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset containing all computed fields
    feature_names : List[str]
        List of feature names to extract
        
    Returns
    -------
    np.ndarray
        Feature array of shape (n_points, n_features)
        
    Raises
    ------
    KeyError
        If a required feature is not found in the data
    """
    n_points = unified_data['u'].size  # Total grid points
    n_features = len(feature_names)
    
    # Initialize feature array
    features = np.zeros((n_points, n_features))
    
    # Extract and flatten each feature
    for i, feature_name in enumerate(feature_names):
        if feature_name not in unified_data:
            raise KeyError(f"Feature '{feature_name}' not found in unified data")
        
        feature_data = unified_data[feature_name]
        features[:, i] = feature_data.flatten()
    
    return features


def compute_all_features(unified_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute all possible features from unified dataset.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset from data loading
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with all computed features added to original data
    """
    # Copy original data
    all_features = unified_data.copy()
    
    # Get reference shape for consistency
    if 'x' in unified_data:
        reference_shape = unified_data['x'].shape
    elif 'u' in unified_data:
        reference_shape = unified_data['u'].shape
    else:
        reference_shape = None
    
    # Ensure all arrays have consistent shapes (either all 2D or all flattened)
    if reference_shape is not None:
        for key, arr in all_features.items():
            if isinstance(arr, np.ndarray) and arr.size == np.prod(reference_shape):
                if arr.shape != reference_shape:
                    # Reshape to reference shape if needed
                    all_features[key] = arr.reshape(reference_shape)
    
    # Handle strain field compatibility: create 'dudx' alias for 'primary_strain' if needed
    if 'primary_strain' in unified_data and 'dudx' not in unified_data:
        all_features['dudx'] = unified_data['primary_strain']
        print("Using primary_strain as dudx for feature compatibility")
    
    # Extract basic fields
    u = all_features['u']
    v = all_features['v']
    
    # Compute velocity features
    velocity_features = compute_velocity_features(u, v)
    all_features.update(velocity_features)
    
    # Compute strain rate features if gradients are available
    if all(key in all_features for key in ['dudx', 'dvdy', 'dudy', 'dvdx']):
        # Check if strain rate tensors have already been computed and are consistent
        if all(key in all_features for key in ['epsilon_xx', 'epsilon_yy', 'epsilon_xy', 'effective_strain']):
            # Use existing computed values
            pass
        else:
            # Compute strain rate tensor
            strain_tensor = compute_strain_rate_tensor(
                all_features['dudx'], all_features['dvdy'],
                all_features['dudy'], all_features['dvdx']
            )
            all_features.update(strain_tensor)
            
            # Advanced deformation features
            deformation_features = compute_deformation_features(strain_tensor)
            all_features.update(deformation_features)
            
            validate_strain_features(strain_tensor)
    
    # Compute viscosity features if available
    if 'mu' in all_features and 'eta' in all_features:
        mu = all_features['mu']
        eta = all_features['eta']
        h = all_features.get('h')
        
        # Basic viscosity features
        viscosity_features = compute_viscosity_features(mu, eta, h)
        all_features.update(viscosity_features)
        
        # Stress features if strain data is available
        if 'effective_strain' in all_features:
            stress_features = compute_stress_features(mu, eta, all_features)
            all_features.update(stress_features)
        
        validate_viscosity_features(viscosity_features)
    
    return all_features


def create_feature_set(unified_data: Dict[str, np.ndarray],
                      feature_set_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Create a specific feature set for clustering.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset
    feature_set_name : str
        Name of feature set ('baseline', 'primary', 'extended', 'stress')
        
    Returns
    -------
    Tuple[np.ndarray, List[str], np.ndarray]
        - Feature array (n_valid_points, n_features)
        - Feature names
        - Valid point mask
    """
    # Compute all features
    all_features = compute_all_features(unified_data)
    
    # Get feature names for this set
    if feature_set_name == 'baseline':
        feature_names = FeatureSetDefinitions.get_baseline_features()
    elif feature_set_name == 'primary':
        feature_names = FeatureSetDefinitions.get_primary_features()
    elif feature_set_name == 'extended':
        feature_names = FeatureSetDefinitions.get_extended_features()
    elif feature_set_name == 'stress':
        feature_names = FeatureSetDefinitions.get_stress_features()
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    
    # Extract features
    features = extract_features_from_data(all_features, feature_names)
    
    # Create validity mask
    valid_mask = np.all(np.isfinite(features), axis=1)
    
    # Remove invalid points
    valid_features = features[valid_mask]
    
    return valid_features, feature_names, valid_mask


def create_ablation_feature_sets(unified_data: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, List[str], np.ndarray]]:
    """
    Create all feature sets for ablation study.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset
        
    Returns
    -------
    Dict[str, Tuple[np.ndarray, List[str], np.ndarray]]
        Dictionary mapping ablation names to (features, names, mask) tuples
    """
    # Compute all features
    all_features = compute_all_features(unified_data)
    
    # Get ablation feature definitions
    ablation_defs = FeatureSetDefinitions.get_ablation_features()
    
    ablation_sets = {}
    
    for ablation_name, feature_names in ablation_defs.items():
        try:
            features = extract_features_from_data(all_features, feature_names)
            valid_mask = np.all(np.isfinite(features), axis=1)
            valid_features = features[valid_mask]
            
            ablation_sets[ablation_name] = (valid_features, feature_names, valid_mask)
        except KeyError as e:
            print(f"Warning: Skipping ablation '{ablation_name}': {e}")
    
    return ablation_sets


def get_spatial_coordinates(unified_data: Dict[str, np.ndarray],
                           valid_mask: np.ndarray) -> np.ndarray:
    """
    Get spatial coordinates for valid points.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset
    valid_mask : np.ndarray
        Boolean mask of valid points
        
    Returns
    -------
    np.ndarray
        Coordinates array (n_valid_points, 2) with [x, y] for each point
    """
    x = unified_data['x'].flatten()
    y = unified_data['y'].flatten()
    
    coordinates = np.column_stack([x, y])
    valid_coordinates = coordinates[valid_mask]
    
    return valid_coordinates


def create_feature_summary(feature_set_name: str,
                          features: np.ndarray,
                          feature_names: List[str],
                          valid_mask: np.ndarray) -> str:
    """
    Create summary of a feature set.
    
    Parameters
    ----------
    feature_set_name : str
        Name of the feature set
    features : np.ndarray
        Feature array
    feature_names : List[str]
        Feature names
    valid_mask : np.ndarray
        Valid point mask
        
    Returns
    -------
    str
        Summary string
    """
    summary_lines = [f"Feature Set: {feature_set_name}"]
    summary_lines.append("-" * 50)
    summary_lines.append(f"Number of features: {len(feature_names)}")
    summary_lines.append(f"Total points: {len(valid_mask)}")
    summary_lines.append(f"Valid points: {np.sum(valid_mask)} ({100*np.mean(valid_mask):.1f}%)")
    summary_lines.append(f"Feature names: {feature_names}")
    summary_lines.append("")
    
    # Feature statistics
    summary_lines.append("Feature Statistics:")
    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        summary_lines.append(
            f"  {name}: "
            f"mean={np.mean(feat_vals):.3e}, "
            f"std={np.std(feat_vals):.3e}, "
            f"range=[{np.min(feat_vals):.3e}, {np.max(feat_vals):.3e}]"
        )
    
    return "\n".join(summary_lines)


def validate_feature_set(features: np.ndarray, feature_names: List[str]) -> None:
    """
    Validate a feature set for clustering.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array
    feature_names : List[str]
        Feature names
        
    Raises
    ------
    ValueError
        If features are not suitable for clustering
    """
    if features.size == 0:
        raise ValueError("Feature array is empty")
    
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"Number of feature columns ({features.shape[1]}) doesn't match "
            f"number of feature names ({len(feature_names)})"
        )
    
    # Check for constant features (no variance)
    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        if np.std(feat_vals) < 1e-12:
            raise ValueError(f"Feature '{name}' has no variance (constant values)")
    
    # Check for very high correlation between features
    if features.shape[1] > 1:
        corr_matrix = np.corrcoef(features.T)
        high_corr = np.abs(corr_matrix) > 0.99
        # Remove diagonal
        np.fill_diagonal(high_corr, False)
        
        if np.any(high_corr):
            pairs = np.where(high_corr)
            for i, j in zip(pairs[0], pairs[1]):
                if i < j:  # Avoid duplicate warnings
                    corr_val = corr_matrix[i, j]
                    print(f"Warning: High correlation ({corr_val:.3f}) between "
                          f"'{feature_names[i]}' and '{feature_names[j]}'")


def get_feature_set_description(feature_set_name: str) -> str:
    """
    Get description of a feature set.
    
    Parameters
    ----------
    feature_set_name : str
        Name of the feature set
        
    Returns
    -------
    str
        Description of the feature set
    """
    descriptions = {
        'baseline': (
            "Core deformation features: longitudinal strain rate and velocity magnitude. "
            "Tests if deformation alone can recover ice flow regimes."
        ),
        'primary': (
            "Velocity + viscosity features: strain, speed, horizontal viscosity, and "
            "anisotropy ratio. This is the main production configuration."
        ),
        'extended': (
            "Extended physics features including ice thickness and additional strain "
            "rate measures for comprehensive regime identification."
        ),
        'stress': (
            "Stress-based features using viscosity-strain coupling to characterize "
            "the mechanical state of the ice shelf."
        )
    }
    
    return descriptions.get(feature_set_name, f"Feature set: {feature_set_name}")


def recommend_feature_set(data_availability: Dict[str, bool]) -> str:
    """
    Recommend appropriate feature set based on data availability.
    
    Parameters
    ---------- 
    data_availability : Dict[str, bool]
        Dictionary indicating which data types are available
        
    Returns
    -------
    str
        Recommended feature set name
    """
    has_viscosity = data_availability.get('viscosity', False)
    has_thickness = data_availability.get('thickness', False)
    has_strain = data_availability.get('strain', False)
    
    if has_viscosity and has_strain:
        return 'primary'  # Best option
    elif has_strain:
        return 'baseline'  # Deformation only
    else:
        raise ValueError("Insufficient data: need at least velocity and strain rate data")