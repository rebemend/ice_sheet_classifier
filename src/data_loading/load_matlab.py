import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings


def load_viscosity_data(mat_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load anisotropic viscosity data from MATLAB .mat file.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the results.mat file containing viscosity data
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing viscosity fields:
        - 'mu': horizontal viscosity μ(x,y) 
        - 'eta': vertical viscosity η(x,y)
        - 'x': x-coordinates (if available)
        - 'y': y-coordinates (if available)
    
    Raises
    ------
    FileNotFoundError
        If the specified mat file does not exist
    KeyError
        If expected viscosity fields are not found in the mat file
    """
    mat_path = Path(mat_file_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_file_path}")
    
    # Load MATLAB data
    try:
        mat_data = loadmat(str(mat_path))
    except Exception as e:
        raise ValueError(f"Failed to load MATLAB file {mat_file_path}: {e}")
    
    # Extract viscosity fields - adapt these keys based on actual file structure
    viscosity_data = {}
    
    # Check if data is nested in a 'results' structure
    working_data = mat_data
    if 'results' in mat_data:
        results = mat_data['results']
        # Handle MATLAB struct array
        if hasattr(results, 'dtype') and results.dtype.names:
            # Extract fields from the struct
            working_data = {}
            for field_name in results.dtype.names:
                working_data[field_name] = results[field_name][0, 0]
        else:
            working_data = results
    
    # Common possible variable names in MATLAB files
    possible_mu_names = ['mu', 'mu_h', 'horizontal_viscosity', 'viscosity_h']
    possible_eta_names = ['eta', 'eta_v', 'vertical_viscosity', 'viscosity_v']
    possible_x_names = ['x', 'X', 'x_coord', 'longitude']
    possible_y_names = ['y', 'Y', 'y_coord', 'latitude']
    possible_strain_names = ['str', 'strain', 'strain_rate', 'effective_strain']
    
    # Find horizontal viscosity
    mu_key = None
    for name in possible_mu_names:
        if name in working_data:
            mu_key = name
            break
    
    if mu_key is None:
        available_keys = [k for k in working_data.keys() if not k.startswith('__')]
        raise KeyError(f"Horizontal viscosity not found. Available keys: {available_keys}")
    
    # Find vertical viscosity
    eta_key = None
    for name in possible_eta_names:
        if name in working_data:
            eta_key = name
            break
    
    if eta_key is None:
        available_keys = [k for k in working_data.keys() if not k.startswith('__')]
        raise KeyError(f"Vertical viscosity not found. Available keys: {available_keys}")
    
    viscosity_data['mu'] = working_data[mu_key]
    viscosity_data['eta'] = working_data[eta_key]
    
    # Try to extract strain field (for DIFFUSE data where velocities are near-zero)
    strain_key = None
    for name in possible_strain_names:
        if name in working_data:
            strain_key = name
            break
    
    if strain_key is not None:
        viscosity_data['effective_strain'] = working_data[strain_key]
        print(f"Found strain field: {strain_key}")
    else:
        warnings.warn("No strain field found in viscosity data")
    
    # Try to extract coordinate information if available
    for name in possible_x_names:
        if name in working_data:
            viscosity_data['x'] = working_data[name]
            break
    
    for name in possible_y_names:
        if name in working_data:
            viscosity_data['y'] = working_data[name]
            break
    
    # Extract scaling factors if available (for DIFFUSE velocity scaling)
    # First check if we're in the nested results structure
    if 'results' in mat_data and hasattr(mat_data['results'], 'dtype'):
        results = mat_data['results']
        if 'scale' in results.dtype.names:
            scale_struct = results['scale'][0,0]
            if hasattr(scale_struct, 'dtype') and scale_struct.dtype.names:
                if 'u0' in scale_struct.dtype.names and 'v0' in scale_struct.dtype.names:
                    viscosity_data['scale_u0'] = scale_struct['u0'][0,0][0,0]
                    viscosity_data['scale_v0'] = scale_struct['v0'][0,0][0,0]
                    print(f"Extracted velocity scales: u0={viscosity_data['scale_u0']:.6e}, v0={viscosity_data['scale_v0']:.6e}")
    # Fallback to direct scale field
    elif 'scale' in mat_data:
        scale = mat_data['scale']
        if hasattr(scale, 'dtype') and scale.dtype.names and 'u0' in scale.dtype.names and 'v0' in scale.dtype.names:
            viscosity_data['scale_u0'] = scale['u0'][0,0][0,0]
            viscosity_data['scale_v0'] = scale['v0'][0,0][0,0]
            print(f"Extracted velocity scales (direct): u0={viscosity_data['scale_u0']:.6e}, v0={viscosity_data['scale_v0']:.6e}")
    
    return viscosity_data


def compute_anisotropy_ratio(mu: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    Compute anisotropy ratio μ/η from horizontal and vertical viscosity.
    
    Parameters
    ----------
    mu : np.ndarray
        Horizontal viscosity array
    eta : np.ndarray  
        Vertical viscosity array
        
    Returns
    -------
    np.ndarray
        Anisotropy ratio μ/η, with small values clipped to avoid division issues
    """
    # Avoid division by very small numbers
    eta_safe = np.where(np.abs(eta) < 1e-12, 1e-12, eta)
    return mu / eta_safe


def validate_viscosity_data(viscosity_data: Dict[str, np.ndarray]) -> None:
    """
    Validate loaded viscosity data for consistency and expected properties.
    
    Parameters
    ----------
    viscosity_data : Dict[str, np.ndarray]
        Viscosity data dictionary from load_viscosity_data()
        
    Raises
    ------
    ValueError
        If data validation fails
    """
    required_fields = ['mu', 'eta']
    for field in required_fields:
        if field not in viscosity_data:
            raise ValueError(f"Required field '{field}' missing from viscosity data")
    
    mu = viscosity_data['mu']
    eta = viscosity_data['eta']
    
    # Check shapes match
    if mu.shape != eta.shape:
        raise ValueError(f"Viscosity field shapes don't match: mu {mu.shape} vs eta {eta.shape}")
    
    # Check for reasonable values (viscosity should be positive where finite)
    mu_finite = mu[np.isfinite(mu)]
    eta_finite = eta[np.isfinite(eta)]
    
    if len(mu_finite) == 0:
        raise ValueError("Horizontal viscosity contains no finite values")
    if len(eta_finite) == 0:
        raise ValueError("Vertical viscosity contains no finite values")
        
    if np.any(mu_finite <= 0):
        raise ValueError("Horizontal viscosity contains non-positive finite values")
    if np.any(eta_finite <= 0):
        raise ValueError("Vertical viscosity contains non-positive finite values")
    
    # Warn about NaN values but don't fail (expected for domain masks)
    mu_nan_count = np.sum(~np.isfinite(mu))
    eta_nan_count = np.sum(~np.isfinite(eta))
    
    if mu_nan_count > 0:
        warnings.warn(f"Horizontal viscosity contains {mu_nan_count} NaN/infinite values (likely domain mask)")
    if eta_nan_count > 0:
        warnings.warn(f"Vertical viscosity contains {eta_nan_count} NaN/infinite values (likely domain mask)")


def load_and_validate_viscosity(mat_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load and validate viscosity data from MATLAB file.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the results.mat file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Validated viscosity data including anisotropy ratio
    """
    # Load raw data
    viscosity_data = load_viscosity_data(mat_file_path)
    
    # Validate
    validate_viscosity_data(viscosity_data)
    
    # Compute anisotropy ratio
    viscosity_data['anisotropy'] = compute_anisotropy_ratio(
        viscosity_data['mu'], viscosity_data['eta']
    )
    
    return viscosity_data