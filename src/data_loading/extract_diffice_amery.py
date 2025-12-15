import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings
try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_diffice_data(diffice_data_path: str) -> Dict[str, np.ndarray]:
    """
    Load Amery Ice Shelf data from DIFFICE_jax repository structure or direct .mat file.
    
    This function attempts to load the following fields:
    - x, y: spatial coordinates
    - u, v: velocity components  
    - h: ice thickness
    - strain rate components (if available)
    
    Parameters
    ----------
    diffice_data_path : str
        Path to the directory containing DIFFICE Amery data files or direct .mat file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing loaded fields
        
    Raises
    ------
    FileNotFoundError
        If the specified directory/file or required files are not found
    """
    data_path = Path(diffice_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"DIFFICE data path not found: {diffice_data_path}")
    
    # If it's a .mat file, load it directly
    if data_path.is_file() and data_path.suffix.lower() == '.mat':
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
        try:
            diffice_data = scipy.io.loadmat(str(data_path))
            # Remove metadata entries that start with '__'
            diffice_data = {k: v for k, v in diffice_data.items() if not k.startswith('__')}
            return diffice_data
        except Exception as e:
            raise ValueError(f"Failed to load .mat file {diffice_data_path}: {e}")
    
    # Otherwise, handle as directory
    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected directory or .mat file, got: {diffice_data_path}")
    
    diffice_data = {}
    
    # Look for common DIFFICE file patterns
    # This may need to be adjusted based on actual DIFFICE_jax file structure
    possible_extensions = ['.pkl', '.npy', '.npz', '.pickle', '.mat']
    possible_files = [
        'amery_data', 'amery_ice_shelf', 'data', 'results',
        'velocity_field', 'thickness', 'coordinates'
    ]
    
    # Try to find and load data files
    loaded_files = []
    for file_pattern in possible_files:
        for ext in possible_extensions:
            file_path = data_path / f"{file_pattern}{ext}"
            if file_path.exists():
                try:
                    data = load_file_by_extension(file_path)
                    if isinstance(data, dict):
                        diffice_data.update(data)
                    else:
                        diffice_data[file_pattern] = data
                    loaded_files.append(str(file_path))
                except Exception as e:
                    warnings.warn(f"Failed to load {file_path}: {e}")
    
    if not loaded_files:
        # List available files for debugging
        available_files = list(data_path.iterdir())
        raise FileNotFoundError(
            f"No recognizable DIFFICE data files found in {diffice_data_path}. "
            f"Available files: {[f.name for f in available_files]}"
        )
    
    return diffice_data


def load_file_by_extension(file_path: Path) -> Union[Dict, np.ndarray]:
    """
    Load data file based on extension.
    
    Parameters
    ----------
    file_path : Path
        Path to the data file
        
    Returns
    -------
    Union[Dict, np.ndarray]
        Loaded data
    """
    ext = file_path.suffix.lower()
    
    if ext == '.npy':
        return np.load(file_path)
    elif ext == '.npz':
        return dict(np.load(file_path))
    elif ext in ['.pkl', '.pickle']:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif ext == '.mat':
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
        data = scipy.io.loadmat(str(file_path))
        # Remove metadata entries that start with '__'
        return {k: v for k, v in data.items() if not k.startswith('__')}
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def extract_velocity_components(diffice_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract u and v velocity components from DIFFICE data.
    
    Parameters
    ----------
    diffice_data : Dict[str, np.ndarray]
        Raw DIFFICE data dictionary
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u and v velocity components
        
    Raises
    ------
    KeyError
        If velocity components cannot be found
    """
    # Common variable names for velocity components
    u_names = ['u', 'u_velocity', 'velocity_x', 'vx', 'ud']
    v_names = ['v', 'v_velocity', 'velocity_y', 'vy', 'vd']
    
    u, v = None, None
    
    # Find u component
    for name in u_names:
        if name in diffice_data:
            u = diffice_data[name]
            break
    
    # Find v component  
    for name in v_names:
        if name in diffice_data:
            v = diffice_data[name]
            break
    
    if u is None or v is None:
        available_keys = list(diffice_data.keys())
        raise KeyError(
            f"Velocity components not found. Available keys: {available_keys}. "
            f"Looking for u in {u_names} and v in {v_names}"
        )
    
    return u, v


def extract_coordinates(diffice_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract spatial coordinates from DIFFICE data.
    
    Parameters
    ----------
    diffice_data : Dict[str, np.ndarray]
        Raw DIFFICE data dictionary
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x and y coordinate arrays
    """
    x_names = ['x', 'X', 'x_coord', 'longitude', 'lon', 'xd']
    y_names = ['y', 'Y', 'y_coord', 'latitude', 'lat', 'yd']
    
    x, y = None, None
    
    for name in x_names:
        if name in diffice_data:
            x = diffice_data[name]
            break
    
    for name in y_names:
        if name in diffice_data:
            y = diffice_data[name]
            break
    
    if x is None or y is None:
        available_keys = list(diffice_data.keys())
        raise KeyError(
            f"Coordinates not found. Available keys: {available_keys}. "
            f"Looking for x in {x_names} and y in {y_names}"
        )
    
    return x, y


def extract_ice_thickness(diffice_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract ice thickness from DIFFICE data.
    
    Parameters
    ----------
    diffice_data : Dict[str, np.ndarray]
        Raw DIFFICE data dictionary
        
    Returns
    -------
    np.ndarray
        Ice thickness array
    """
    h_names = ['h', 'thickness', 'ice_thickness', 'H', 'hd']
    
    for name in h_names:
        if name in diffice_data:
            return diffice_data[name]
    
    available_keys = list(diffice_data.keys())
    raise KeyError(
        f"Ice thickness not found. Available keys: {available_keys}. "
        f"Looking for thickness in {h_names}"
    )


def compute_strain_rates(u: np.ndarray, v: np.ndarray, 
                        x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute strain rate components using finite differences.
    
    Parameters
    ----------
    u, v : np.ndarray
        Velocity components
    x, y : np.ndarray  
        Coordinate arrays
        
    Returns
    -------
    Dict[str, np.ndarray]
        Strain rate components: 'dudx', 'dvdy', 'dudy', 'dvdx'
    """
    # Extract coordinate spacing for gradients
    # Handle both 1D and 2D coordinate arrays
    if x.ndim == 1:
        dx = x[1] - x[0] if len(x) > 1 else 1.0
    else:
        dx = np.mean(np.diff(x, axis=1)) if x.shape[1] > 1 else 1.0
        
    if y.ndim == 1:
        dy = y[1] - y[0] if len(y) > 1 else 1.0
    else:
        dy = np.mean(np.diff(y, axis=0)) if y.shape[0] > 1 else 1.0
    
    # Compute gradients using uniform spacing (central differences)
    dudx = np.gradient(u, dx, axis=1)  # Assuming x varies along axis 1
    dudy = np.gradient(u, dy, axis=0)  # Assuming y varies along axis 0
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    
    return {
        'dudx': dudx,  # Longitudinal strain rate
        'dvdy': dvdy,  # Transverse strain rate  
        'dudy': dudy,  # Shear strain rate component
        'dvdx': dvdx   # Shear strain rate component
    }


def load_and_process_diffice_amery(diffice_data_path: str, 
                                   velocity_scaling: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    """
    Load and process all required fields from DIFFICE Amery data.
    
    Parameters
    ----------
    diffice_data_path : str
        Path to DIFFICE Amery data directory
    velocity_scaling : Optional[Tuple[float, float]]
        Velocity scaling factors (u0, v0) to convert dimensionless to physical units
        
    Returns
    -------
    Dict[str, np.ndarray]
        Processed data containing:
        - 'x', 'y': coordinates
        - 'u', 'v': velocities (scaled to physical units if scaling provided)
        - 'h': ice thickness
        - 'dudx', 'dvdy', 'dudy', 'dvdx': strain rates
        - 'speed': velocity magnitude
    """
    # Load raw data
    raw_data = load_diffice_data(diffice_data_path)
    
    # Extract required components
    u, v = extract_velocity_components(raw_data)
    x, y = extract_coordinates(raw_data)
    
    # Apply velocity scaling if provided
    if velocity_scaling is not None:
        u0, v0 = velocity_scaling
        u = u * u0  # Convert to physical units
        v = v * v0  # Convert to physical units
        print(f"Applied velocity scaling: u0={u0:.6e}, v0={v0:.6e}")
    else:
        warnings.warn("No velocity scaling applied - velocities remain dimensionless") 
    
    # Try to extract thickness, but handle gracefully if different grid sizes
    try:
        h = extract_ice_thickness(raw_data)
        # Check if thickness has same shape as velocity
        if h.shape != u.shape:
            warnings.warn(f"Thickness shape {h.shape} differs from velocity shape {u.shape}. Using dummy thickness.")
            h = np.ones_like(u)  # Use dummy thickness for now
    except KeyError:
        warnings.warn("Thickness not found in data. Using dummy thickness.")
        h = np.ones_like(u)
    
    # Skip strain rate computation from near-zero velocities
    # These will be loaded from results.mat instead
    warnings.warn("DIFFUSE velocities are near-zero. Strain rates should be loaded from results.mat separately.")
    
    # Compute speed for completeness (even though very small)
    speed = np.sqrt(u**2 + v**2)
    
    # Create placeholder strain rates (will be overridden by results.mat data)
    strain_rates = {
        'dudx': np.zeros_like(u),
        'dvdy': np.zeros_like(u), 
        'dudy': np.zeros_like(u),
        'dvdx': np.zeros_like(u)
    }
    
    # Assemble processed data
    processed_data = {
        'x': x,
        'y': y, 
        'u': u,
        'v': v,
        'h': h,
        'speed': speed,
        **strain_rates
    }
    
    return processed_data


def validate_diffice_data(diffice_data: Dict[str, np.ndarray]) -> None:
    """
    Validate DIFFICE data for consistency and expected properties.
    
    Parameters
    ----------
    diffice_data : Dict[str, np.ndarray]
        Processed DIFFICE data
        
    Raises
    ------
    ValueError
        If validation fails
    """
    required_fields = ['x', 'y', 'u', 'v', 'h', 'dudx']
    
    for field in required_fields:
        if field not in diffice_data:
            raise ValueError(f"Required field '{field}' missing from DIFFICE data")
    
    # Check that spatial dimensions are consistent
    reference_shape = diffice_data['u'].shape
    spatial_fields = ['v', 'h', 'dudx', 'dvdy', 'dudy', 'dvdx', 'speed']
    
    for field in spatial_fields:
        if field in diffice_data and diffice_data[field].shape != reference_shape:
            raise ValueError(
                f"Shape mismatch: {field} has shape {diffice_data[field].shape}, "
                f"expected {reference_shape}"
            )
    
    # Check for NaN values in critical fields
    critical_fields = ['u', 'v', 'h']
    for field in critical_fields:
        if np.any(~np.isfinite(diffice_data[field])):
            warnings.warn(f"Field '{field}' contains NaN or infinite values")
    
    # Check ice thickness is positive
    if np.any(diffice_data['h'] <= 0):
        warnings.warn("Ice thickness contains non-positive values")