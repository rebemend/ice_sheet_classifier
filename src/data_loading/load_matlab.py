import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Tuple, Optional


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
    
    # Common possible variable names in MATLAB files
    possible_mu_names = ['mu', 'mu_h', 'horizontal_viscosity', 'viscosity_h']
    possible_eta_names = ['eta', 'eta_v', 'vertical_viscosity', 'viscosity_v']
    possible_x_names = ['x', 'X', 'x_coord', 'longitude']
    possible_y_names = ['y', 'Y', 'y_coord', 'latitude']
    
    # Find horizontal viscosity
    mu_key = None
    for name in possible_mu_names:
        if name in mat_data:
            mu_key = name
            break
    
    if mu_key is None:
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raise KeyError(f"Horizontal viscosity not found. Available keys: {available_keys}")
    
    # Find vertical viscosity
    eta_key = None
    for name in possible_eta_names:
        if name in mat_data:
            eta_key = name
            break
    
    if eta_key is None:
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raise KeyError(f"Vertical viscosity not found. Available keys: {available_keys}")
    
    viscosity_data['mu'] = mat_data[mu_key]
    viscosity_data['eta'] = mat_data[eta_key]
    
    # Try to extract coordinate information if available
    for name in possible_x_names:
        if name in mat_data:
            viscosity_data['x'] = mat_data[name]
            break
    
    for name in possible_y_names:
        if name in mat_data:
            viscosity_data['y'] = mat_data[name]
            break
    
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
    
    # Check for reasonable values (viscosity should be positive)
    if np.any(mu <= 0):
        raise ValueError("Horizontal viscosity contains non-positive values")
    if np.any(eta <= 0):
        raise ValueError("Vertical viscosity contains non-positive values")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(mu)):
        raise ValueError("Horizontal viscosity contains NaN or infinite values")
    if np.any(~np.isfinite(eta)):
        raise ValueError("Vertical viscosity contains NaN or infinite values")


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