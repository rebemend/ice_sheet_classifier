import numpy as np
from typing import Dict, Tuple, Optional
from scipy.interpolate import griddata
import warnings
from .extract_diffice_amery import load_and_process_diffice_amery, validate_diffice_data
from .load_matlab import load_and_validate_viscosity
from .load_results_mat import load_complete_results_data, validate_results_data


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
    
    # Flatten source coordinates and remove NaN points
    source_x_flat = source_x_grid.flatten()
    source_y_flat = source_y_grid.flatten()
    
    # Find finite coordinate points
    finite_mask = np.isfinite(source_x_flat) & np.isfinite(source_y_flat)
    
    if np.sum(finite_mask) == 0:
        raise ValueError("No finite coordinate points found in source data")
    
    source_points = np.column_stack([
        source_x_flat[finite_mask],
        source_y_flat[finite_mask]
    ])
    
    # Target grid points - also filter NaN
    target_x_flat = target_x.flatten()
    target_y_flat = target_y.flatten()
    target_finite_mask = np.isfinite(target_x_flat) & np.isfinite(target_y_flat)
    
    if np.sum(target_finite_mask) == 0:
        raise ValueError("No finite coordinate points found in target data")
    
    target_points_finite = np.column_stack([
        target_x_flat[target_finite_mask],
        target_y_flat[target_finite_mask]
    ])
    
    interpolated_data = {}
    
    for field in fields_to_interpolate:
        if field not in source_data:
            warnings.warn(f"Field '{field}' not found in source data, skipping")
            continue
        
        source_field = source_data[field]
        source_field_flat = source_field.flatten()
        
        # Only use finite source field values that correspond to finite coordinates
        source_values = source_field_flat[finite_mask]
        
        # Further filter to remove NaN values in the field itself
        field_finite_mask = np.isfinite(source_values)
        if np.sum(field_finite_mask) == 0:
            warnings.warn(f"Field '{field}' has no finite values, filling with NaN")
            interpolated_data[field] = np.full(target_x.shape, np.nan)
            continue
        
        final_source_points = source_points[field_finite_mask]
        final_source_values = source_values[field_finite_mask]
        
        # Create full target grid for output
        target_points_full = np.column_stack([
            target_x_flat,
            target_y_flat
        ])
        
        # Interpolate using griddata with linear interpolation  
        interpolated_field = griddata(
            final_source_points,
            final_source_values,
            target_points_full,
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
    
    # Check coordinate similarity (within 1% tolerance, handling NaN values)
    # Only compare where both coordinates are finite
    finite_d = np.isfinite(diffice_x) & np.isfinite(diffice_y)
    finite_v = np.isfinite(viscosity_x) & np.isfinite(viscosity_y)
    both_finite = finite_d & finite_v
    
    if np.sum(both_finite) == 0:
        # No overlapping finite points - assume incompatible
        return False
    
    x_close = np.allclose(diffice_x[both_finite], viscosity_x[both_finite], rtol=0.01, atol=1e-6)
    y_close = np.allclose(diffice_y[both_finite], viscosity_y[both_finite], rtol=0.01, atol=1e-6)
    
    return x_close and y_close


def create_unified_dataset_from_results(results_mat_path: str) -> Dict[str, np.ndarray]:
    """
    Create unified dataset using results.mat as primary source for consistency.
    
    This approach uses the results.mat file as the primary data source since it
    contains all required fields on a consistent grid with proper scaling and
    derived quantities already computed.
    
    Parameters
    ----------
    results_mat_path : str
        Path to the results.mat file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Complete unified dataset with all fields
    """
    print("Loading complete dataset from results.mat...")
    unified_data = load_complete_results_data(results_mat_path)
    validate_results_data(unified_data)
    
    print(f"Unified dataset created successfully:")
    print(f"  Grid shape: {unified_data['x'].shape}")
    print(f"  Total fields: {len(unified_data)}")
    print(f"  Key fields: {list(unified_data.keys())}")
    
    return unified_data


def create_unified_dataset(diffice_data_path: str, 
                          viscosity_mat_path: str,
                          force_interpolation: bool = False,
                          use_results_only: bool = True) -> Dict[str, np.ndarray]:
    """
    Create unified dataset by combining DIFFICE and viscosity data.
    
    Parameters
    ----------
    diffice_data_path : str
        Path to DIFFICE Amery data
    viscosity_mat_path : str
        Path to viscosity MATLAB file (results.mat)
    force_interpolation : bool
        If True, force interpolation even if grids appear compatible
    use_results_only : bool
        If True, use results.mat as primary source (recommended for consistency)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Unified dataset containing all required fields on common grid
    """
    # Use results.mat only for consistency (recommended approach)
    if use_results_only:
        print("Using results.mat as primary data source for consistency...")
        return create_unified_dataset_from_results(viscosity_mat_path)
    
    # Legacy approach: combine DIFFICE and viscosity data (may have scaling issues)
    print("Loading viscosity data first to extract scaling factors...")
    viscosity_data = load_and_validate_viscosity(viscosity_mat_path)
    
    # Extract velocity scaling factors from viscosity data if available
    velocity_scaling = None
    if 'scale_u0' in viscosity_data and 'scale_v0' in viscosity_data:
        velocity_scaling = (viscosity_data['scale_u0'], viscosity_data['scale_v0'])
        print(f"Found velocity scaling factors: u0={velocity_scaling[0]:.6e}, v0={velocity_scaling[1]:.6e}")
    
    print("Loading DIFFICE data...")
    diffice_data = load_and_process_diffice_amery(diffice_data_path, velocity_scaling)
    validate_diffice_data(diffice_data)
    
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
    # Use effective_strain from results.mat if available, otherwise dudx
    if 'effective_strain' in unified_data:
        primary_strain = unified_data['effective_strain'].flatten()
        print("Using effective_strain from viscosity data")
    else:
        primary_strain = unified_data['dudx'].flatten()
        warnings.warn("Using computed dudx (may be near-zero for DIFFUSE data)")
    
    speed = unified_data['speed'].flatten()
    
    # Check if viscosity fields are available
    if 'mu' in unified_data:
        mu = unified_data['mu'].flatten()
        has_viscosity = True
        
        # Compute anisotropy if not already present
        if 'anisotropy' in unified_data:
            anisotropy = unified_data['anisotropy'].flatten()
        elif 'eta' in unified_data:
            eta = unified_data['eta'].flatten()
            anisotropy = eta / mu
            print("Computed anisotropy from eta/mu")
        else:
            anisotropy = np.full_like(mu, np.nan)
            warnings.warn("Cannot compute anisotropy: eta field missing")
    else:
        warnings.warn("Viscosity data not available, using strain-based features only")
        mu = np.full_like(primary_strain, np.nan)
        anisotropy = np.full_like(primary_strain, np.nan)
        has_viscosity = False
    
    # Create validity mask (exclude NaN and infinite values)
    # For very small strain values, just check if finite (not necessarily > 0)
    valid_mask = (np.isfinite(primary_strain) & 
                  np.isfinite(speed))  # Allow zero or very small strain values
    
    if has_viscosity:
        valid_mask = (valid_mask & 
                     np.isfinite(mu) & 
                     np.isfinite(anisotropy) &
                     (mu > 0))  # Viscosity should be positive
    
    # Create feature arrays
    baseline_features = np.column_stack([primary_strain, speed])
    
    if has_viscosity:
        primary_features = np.column_stack([primary_strain, speed, mu, anisotropy])
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
        'primary_strain': primary_strain,  # This is either effective_strain or dudx
        'speed': speed,
        'mu': mu,
        'anisotropy': anisotropy
    })
    
    # Also add the strain field name for reference
    strain_field_name = 'effective_strain' if 'effective_strain' in unified_data else 'dudx'
    feature_data['strain_field_name'] = strain_field_name
    
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
    
    # Keys that belong to feature_data specifically
    feature_only_keys = ['coordinates', 'baseline_features', 'primary_features', 
                        'mask', 'grid_shape']
    
    # Extract feature-specific data
    feature_data = {key: loaded[key] for key in feature_only_keys if key in loaded}
    
    # All other keys go to unified_data (including computed features like dudx, speed, etc.)
    unified_data = {key: loaded[key] for key in loaded.keys() if key not in feature_only_keys}
    
    # Fix shape inconsistency issues in loaded data
    if 'x' in unified_data:
        reference_shape = unified_data['x'].shape
        total_elements = np.prod(reference_shape)
        
        # Ensure all arrays that should have the same spatial extent are reshaped consistently
        spatial_fields = ['x', 'y', 'u', 'v', 'h', 'speed', 'dudx', 'dvdy', 'dudy', 'dvdx',
                         'mu', 'eta', 'effective_strain', 'anisotropy', 'primary_strain',
                         'epsilon_xx', 'epsilon_yy', 'epsilon_xy']
        
        for field_name in spatial_fields:
            if field_name in unified_data:
                field_data = unified_data[field_name]
                if isinstance(field_data, np.ndarray) and field_data.size == total_elements:
                    if field_data.shape != reference_shape:
                        unified_data[field_name] = field_data.reshape(reference_shape)
                        print(f"Reshaped {field_name} from {field_data.shape} to {reference_shape}")
    
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