import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings


def load_complete_results_data(mat_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load complete dataset from results.mat file.
    
    This function loads all available fields from the results.mat file which contains
    the complete ice shelf analysis results including velocities, strain rates, 
    viscosity, and all derived quantities on a consistent grid.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the results.mat file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing all loaded fields:
        - Coordinates: 'x', 'y'
        - Velocities: 'u', 'v', 'speed'
        - Thickness: 'h' 
        - Strain rates: 'dudx', 'dudy', 'dvdx', 'dvdy', 'effective_strain'
        - Viscosity: 'mu', 'eta', 'anisotropy'
        - Strain tensor: 'epsilon_xx', 'epsilon_yy', 'epsilon_xy'
        
    Raises
    ------
    FileNotFoundError
        If the mat file is not found
    KeyError
        If expected fields are not found in the file
    """
    mat_path = Path(mat_file_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Results mat file not found: {mat_file_path}")
    
    # Load MATLAB data
    try:
        mat_data = loadmat(str(mat_path))
    except Exception as e:
        raise ValueError(f"Failed to load results mat file {mat_file_path}: {e}")
    
    if 'results' not in mat_data:
        raise KeyError("'results' field not found in mat file")
    
    results = mat_data['results']
    if not hasattr(results, 'dtype') or not results.dtype.names:
        raise KeyError("Results field does not contain expected structure")
    
    # Extract all available fields
    data = {}
    
    # Extract fields from the struct
    for field_name in results.dtype.names:
        try:
            field_data = results[field_name][0, 0]
            data[field_name] = field_data
        except Exception as e:
            warnings.warn(f"Failed to extract field '{field_name}': {e}")
    
    # Validate that we have the essential fields
    required_fields = ['x', 'y', 'u', 'v', 'str', 'mu', 'eta']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise KeyError(f"Required fields missing from results: {missing_fields}")
    
    # Process and standardize field names for compatibility
    processed_data = process_results_fields(data)
    
    return processed_data


def process_results_fields(raw_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Process and standardize field names from results.mat for compatibility.
    
    Parameters
    ----------
    raw_data : Dict[str, np.ndarray]
        Raw data from results.mat
        
    Returns
    -------
    Dict[str, np.ndarray]
        Processed data with standardized field names
    """
    processed = {}
    
    # Copy coordinates directly
    processed['x'] = raw_data['x']
    processed['y'] = raw_data['y']
    
    # Copy velocities and compute speed
    processed['u'] = raw_data['u']  
    processed['v'] = raw_data['v']
    processed['speed'] = np.sqrt(raw_data['u']**2 + raw_data['v']**2)
    
    # Copy thickness
    if 'h' in raw_data:
        processed['h'] = raw_data['h']
    else:
        warnings.warn("Thickness field 'h' not found in results")
        processed['h'] = np.ones_like(processed['u'])
    
    # Process strain rate components - these are the key fields!
    # u_x = ∂u/∂x, u_y = ∂u/∂y, v_x = ∂v/∂x, v_y = ∂v/∂y
    if all(field in raw_data for field in ['u_x', 'u_y', 'v_x', 'v_y']):
        processed['dudx'] = raw_data['u_x']  # ε_xx component
        processed['dudy'] = raw_data['u_y']  # ε_xy component (part)
        processed['dvdx'] = raw_data['v_x']  # ε_xy component (part)  
        processed['dvdy'] = raw_data['v_y']  # ε_yy component
        
        # Compute strain tensor components
        processed['epsilon_xx'] = raw_data['u_x']  # Longitudinal strain
        processed['epsilon_yy'] = raw_data['v_y']  # Transverse strain
        processed['epsilon_xy'] = 0.5 * (raw_data['u_y'] + raw_data['v_x'])  # Shear strain
        
        print("Extracted strain rate tensor components from velocity gradients")
    else:
        missing = [f for f in ['u_x', 'u_y', 'v_x', 'v_y'] if f not in raw_data]
        warnings.warn(f"Missing velocity gradient fields for strain computation: {missing}")
        # Create zero placeholders
        for field in ['dudx', 'dudy', 'dvdx', 'dvdy', 'epsilon_xx', 'epsilon_yy', 'epsilon_xy']:
            processed[field] = np.zeros_like(processed['u'])
    
    # Copy effective strain rate
    processed['effective_strain'] = raw_data['str']
    
    # Copy viscosity fields
    processed['mu'] = raw_data['mu']
    processed['eta'] = raw_data['eta']
    processed['anisotropy'] = raw_data['mu'] / raw_data['eta']
    
    # Additional derived quantities
    processed['divergence'] = processed['epsilon_xx'] + processed['epsilon_yy']
    processed['shear_magnitude'] = np.sqrt(processed['epsilon_xy']**2)
    
    # Log validation
    finite_mask = np.isfinite(processed['x']) & np.isfinite(processed['y'])
    valid_count = np.sum(finite_mask)
    total_count = processed['x'].size
    
    print(f"Processed results.mat data:")
    print(f"  Grid shape: {processed['x'].shape}")
    print(f"  Valid points: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    print(f"  Effective strain range: [{processed['effective_strain'][finite_mask].min():.3e}, {processed['effective_strain'][finite_mask].max():.3e}] s⁻¹")
    print(f"  Epsilon_xx range: [{processed['epsilon_xx'][finite_mask].min():.3e}, {processed['epsilon_xx'][finite_mask].max():.3e}] s⁻¹")
    
    return processed


def validate_results_data(data: Dict[str, np.ndarray]) -> None:
    """
    Validate the processed results data for consistency.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Processed results data
        
    Raises
    ------
    ValueError
        If data fails validation checks
    """
    # Check that all arrays have the same shape
    reference_shape = data['x'].shape
    for field_name, field_data in data.items():
        if hasattr(field_data, 'shape') and field_data.shape != reference_shape:
            raise ValueError(f"Field '{field_name}' has shape {field_data.shape}, expected {reference_shape}")
    
    # Check for reasonable finite value counts
    finite_mask = np.isfinite(data['x']) & np.isfinite(data['y'])
    finite_count = np.sum(finite_mask)
    
    if finite_count == 0:
        raise ValueError("No finite coordinate values found")
    
    # Validate strain rates are reasonable
    strain_fields = ['effective_strain', 'epsilon_xx', 'epsilon_yy', 'epsilon_xy']
    for field in strain_fields:
        if field in data:
            field_finite = data[field][finite_mask]
            field_finite = field_finite[np.isfinite(field_finite)]
            if len(field_finite) == 0:
                warnings.warn(f"No finite values in strain field '{field}'")
            else:
                # Convert to /year for validation
                field_per_year = field_finite * (365.25 * 24 * 3600)
                if np.abs(field_per_year).max() > 1.0:  # > 1/year is extremely high
                    warnings.warn(f"Very high strain rates in '{field}': max {np.abs(field_per_year).max():.3f} /year")
    
    # Validate viscosity values
    visc_fields = ['mu', 'eta']
    for field in visc_fields:
        if field in data:
            field_finite = data[field][finite_mask]
            field_finite = field_finite[np.isfinite(field_finite)]
            if len(field_finite) > 0:
                if np.any(field_finite <= 0):
                    raise ValueError(f"Non-positive viscosity values found in '{field}'")
                
                # Check against typical ice viscosity range (1e10 - 1e16 Pa·s)
                if field_finite.min() < 1e9 or field_finite.max() > 1e17:
                    warnings.warn(f"Unusual viscosity range in '{field}': [{field_finite.min():.2e}, {field_finite.max():.2e}] Pa·s")
    
    print("Results data validation completed successfully")