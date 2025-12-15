import numpy as np
from typing import Dict, Tuple, Optional
from scipy.interpolate import griddata
import warnings
from .extract_diffice_amery import load_and_process_diffice_amery, validate_diffice_data
from .load_matlab import load_and_validate_viscosity


def interpolate_to_common_grid(source_data: Dict[str, np.ndarray], 
                              target_x: np.ndarray, target_y: np.ndarray,
                              fields_to_interpolate: list) -> Dict[str, np.ndarray]:
    """
    Interpolate data fields to a common spatial grid.
    
    Parameters
    ----------
    source_data : Dict[str, np.ndarray]
        Source data dictionary containing fields and coordinates
    target_x, target_y : np.ndarray
        Target grid coordinates (2D arrays)
    fields_to_interpolate : list
        List of field names to interpolate
        
    Returns
    -------
    Dict[str, np.ndarray]
        Interpolated data on target grid
    """
    if 'x' not in source_data or 'y' not in source_data:
        raise ValueError("Source data must contain 'x' and 'y' coordinate arrays")
    
    source_x = source_data['x']
    source_y = source_data['y']
    
    # Create coordinate pairs for interpolation
    if source_x.ndim == 1 and source_y.ndim == 1:
        # 1D coordinate vectors - create meshgrid
        source_x_grid, source_y_grid = np.meshgrid(source_x, source_y)
    else:
        # Already 2D grids
        source_x_grid, source_y_grid = source_x, source_y
    
    # Flatten source coordinates
    source_points = np.column_stack([
        source_x_grid.flatten(),
        source_y_grid.flatten()
    ])
    
    # Target grid points
    target_points = np.column_stack([
        target_x.flatten(),
        target_y.flatten()
    ])
    
    interpolated_data = {}
    
    for field in fields_to_interpolate:
        if field not in source_data:
            warnings.warn(f"Field '{field}' not found in source data, skipping")
            continue
        
        source_field = source_data[field]
        
        # Interpolate using griddata with linear interpolation
        interpolated_field = griddata(
            source_points,
            source_field.flatten(),
            target_points,
            method='linear',
            fill_value=np.nan
        )
        
        # Reshape to target grid shape
        interpolated_data[field] = interpolated_field.reshape(target_x.shape)
    
    return interpolated_data


def check_grid_compatibility(diffice_data: Dict[str, np.ndarray], 
                           viscosity_data: Dict[str, np.ndarray]) -> bool:
    """
    Check if DIFFICE and viscosity data grids are compatible.
    
    Parameters
    ----------
    diffice_data : Dict[str, np.ndarray]
        DIFFICE data dictionary
    viscosity_data : Dict[str, np.ndarray]  
        Viscosity data dictionary
        
    Returns
    -------
    bool
        True if grids are compatible, False otherwise
    """
    # Check if both datasets have coordinate information
    if not all(key in diffice_data for key in ['x', 'y']):
        return False
    
    if not all(key in viscosity_data for key in ['x', 'y']):
        return False
    
    # Check if grids are approximately the same
    diffice_x, diffice_y = diffice_data['x'], diffice_data['y']
    viscosity_x, viscosity_y = viscosity_data['x'], viscosity_data['y']
    
    # Compare grid shapes and coordinate ranges
    if diffice_x.shape != viscosity_x.shape:
        return False
    
    if diffice_y.shape != viscosity_y.shape:
        return False
    
    # Check coordinate similarity (within 1% tolerance)
    x_close = np.allclose(diffice_x, viscosity_x, rtol=0.01)
    y_close = np.allclose(diffice_y, viscosity_y, rtol=0.01)
    
    return x_close and y_close


def create_unified_dataset(diffice_data_path: str, 
                          viscosity_mat_path: str,
                          force_interpolation: bool = False) -> Dict[str, np.ndarray]:
    """
    Create unified dataset by combining DIFFICE and viscosity data.
    
    Parameters
    ----------
    diffice_data_path : str
        Path to DIFFICE Amery data
    viscosity_mat_path : str
        Path to viscosity MATLAB file
    force_interpolation : bool
        If True, force interpolation even if grids appear compatible
        
    Returns
    -------
    Dict[str, np.ndarray]
        Unified dataset containing all required fields on common grid
    """
    # Load individual datasets
    print("Loading DIFFICE data...")
    diffice_data = load_and_process_diffice_amery(diffice_data_path)
    validate_diffice_data(diffice_data)
    
    print("Loading viscosity data...")
    viscosity_data = load_and_validate_viscosity(viscosity_mat_path)
    
    # Check grid compatibility
    grids_compatible = check_grid_compatibility(diffice_data, viscosity_data)
    
    if grids_compatible and not force_interpolation:
        print("Grids are compatible, merging directly...")
        # Direct merge
        unified_data = {**diffice_data, **viscosity_data}
    else:
        print("Grid interpolation required...")
        # Use DIFFICE grid as reference (higher resolution expected)
        target_x = diffice_data['x']
        target_y = diffice_data['y']
        
        # Interpolate viscosity data to DIFFICE grid
        viscosity_fields = ['mu', 'eta', 'anisotropy']
        interpolated_viscosity = interpolate_to_common_grid(
            viscosity_data, target_x, target_y, viscosity_fields
        )
        
        # Combine datasets
        unified_data = {**diffice_data, **interpolated_viscosity}
    
    return unified_data


def create_feature_arrays(unified_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Create feature arrays suitable for k-means clustering.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset from create_unified_dataset
        
    Returns
    -------
    Dict[str, np.ndarray]
        Feature arrays flattened and ready for clustering:
        - 'coordinates': (N, 2) array of x, y positions
        - 'baseline_features': (N, 2) array of dudx, speed
        - 'primary_features': (N, 4) array of dudx, speed, mu, anisotropy
        - 'mask': (N,) boolean mask of valid points
    """
    # Get spatial dimensions
    reference_shape = unified_data['u'].shape
    n_points = np.prod(reference_shape)
    
    # Create coordinate array
    x_flat = unified_data['x'].flatten()
    y_flat = unified_data['y'].flatten()
    coordinates = np.column_stack([x_flat, y_flat])
    
    # Extract and flatten feature fields
    dudx = unified_data['dudx'].flatten()
    speed = unified_data['speed'].flatten()
    
    # Check if viscosity fields are available
    if 'mu' in unified_data and 'anisotropy' in unified_data:
        mu = unified_data['mu'].flatten()
        anisotropy = unified_data['anisotropy'].flatten()
        has_viscosity = True
    else:
        warnings.warn("Viscosity data not available, using baseline features only")
        mu = np.full_like(dudx, np.nan)
        anisotropy = np.full_like(dudx, np.nan)
        has_viscosity = False
    
    # Create validity mask (exclude NaN and infinite values)
    valid_mask = (np.isfinite(dudx) & 
                  np.isfinite(speed) & 
                  (speed > 0))  # Speed should be non-negative
    
    if has_viscosity:
        valid_mask = (valid_mask & 
                     np.isfinite(mu) & 
                     np.isfinite(anisotropy) &
                     (mu > 0))  # Viscosity should be positive
    
    # Create feature arrays
    baseline_features = np.column_stack([dudx, speed])
    
    if has_viscosity:
        primary_features = np.column_stack([dudx, speed, mu, anisotropy])
    else:
        primary_features = baseline_features
    
    feature_data = {
        'coordinates': coordinates,
        'baseline_features': baseline_features,
        'primary_features': primary_features,
        'mask': valid_mask,
        'grid_shape': reference_shape
    }
    
    # Add individual feature arrays for analysis
    feature_data.update({
        'dudx': dudx,
        'speed': speed,
        'mu': mu,
        'anisotropy': anisotropy
    })
    
    return feature_data


def save_processed_dataset(unified_data: Dict[str, np.ndarray],
                          feature_data: Dict[str, np.ndarray],
                          output_path: str) -> None:
    """
    Save processed dataset to file for reuse.
    
    Parameters
    ----------
    unified_data : Dict[str, np.ndarray]
        Unified dataset with all fields
    feature_data : Dict[str, np.ndarray]  
        Processed feature arrays
    output_path : str
        Output file path (will save as .npz)
    """
    # Combine all data for saving
    save_data = {**unified_data, **feature_data}
    
    # Save as compressed numpy archive
    np.savez_compressed(output_path, **save_data)
    print(f"Processed dataset saved to: {output_path}")


def load_processed_dataset(dataset_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load previously processed dataset.
    
    Parameters
    ---------- 
    dataset_path : str
        Path to processed dataset file
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        Unified data and feature data dictionaries
    """
    loaded = np.load(dataset_path)
    
    # Split back into unified_data and feature_data
    feature_keys = ['coordinates', 'baseline_features', 'primary_features', 
                   'mask', 'grid_shape', 'dudx', 'speed', 'mu', 'anisotropy']
    
    feature_data = {key: loaded[key] for key in feature_keys if key in loaded}
    unified_data = {key: loaded[key] for key in loaded.keys() if key not in feature_keys}
    
    return unified_data, feature_data


def create_complete_dataset(diffice_data_path: str,
                           viscosity_mat_path: str, 
                           output_path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Complete pipeline to create dataset ready for k-means clustering.
    
    Parameters
    ----------
    diffice_data_path : str
        Path to DIFFICE data
    viscosity_mat_path : str  
        Path to viscosity MATLAB file
    output_path : Optional[str]
        If provided, save processed dataset to this path
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        Unified data and feature arrays ready for clustering
    """
    print("Creating unified dataset...")
    unified_data = create_unified_dataset(diffice_data_path, viscosity_mat_path)
    
    print("Creating feature arrays...")
    feature_data = create_feature_arrays(unified_data)
    
    print(f"Dataset summary:")
    print(f"  Total grid points: {len(feature_data['mask'])}")
    print(f"  Valid points: {np.sum(feature_data['mask'])}")
    print(f"  Grid shape: {feature_data['grid_shape']}")
    
    if output_path:
        save_processed_dataset(unified_data, feature_data, output_path)
    
    return unified_data, feature_data