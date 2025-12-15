import numpy as np
from typing import Dict, Tuple
import warnings


def compute_strain_rate_tensor(dudx: np.ndarray, dvdy: np.ndarray,
                              dudy: np.ndarray, dvdx: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute strain rate tensor components and derived quantities.
    
    The strain rate tensor is:
    ε_ij = 0.5 * (∂v_i/∂x_j + ∂v_j/∂x_i)
    
    Parameters
    ----------
    dudx, dvdy : np.ndarray
        Normal strain rate components
    dudy, dvdx : np.ndarray
        Shear strain rate components
        
    Returns
    -------
    Dict[str, np.ndarray]
        Strain rate tensor components and derived quantities:
        - 'epsilon_xx': ∂u/∂x (longitudinal strain rate)
        - 'epsilon_yy': ∂v/∂y (transverse strain rate) 
        - 'epsilon_xy': 0.5*(∂u/∂y + ∂v/∂x) (shear strain rate)
        - 'divergence': ∇·v (velocity divergence)
        - 'shear_magnitude': |γ| (shear strain rate magnitude)
        - 'effective_strain': ε_eff (effective strain rate)
    """
    # Principal strain rate components
    epsilon_xx = dudx  # Longitudinal strain rate
    epsilon_yy = dvdy  # Transverse strain rate
    epsilon_xy = 0.5 * (dudy + dvdx)  # Shear strain rate
    
    # Velocity divergence (trace of strain rate tensor)
    divergence = epsilon_xx + epsilon_yy
    
    # Shear strain rate magnitude
    shear_magnitude = np.sqrt(2) * np.abs(epsilon_xy)
    
    # Effective strain rate (second invariant of strain rate tensor)
    effective_strain = np.sqrt(epsilon_xx**2 + epsilon_yy**2 + 2*epsilon_xy**2)
    
    return {
        'epsilon_xx': epsilon_xx,
        'epsilon_yy': epsilon_yy,
        'epsilon_xy': epsilon_xy,
        'divergence': divergence,
        'shear_magnitude': shear_magnitude,
        'effective_strain': effective_strain
    }


def classify_strain_regime_simple(epsilon_xx: np.ndarray, 
                                 threshold: float = 0.0) -> np.ndarray:
    """
    Simple strain-based regime classification using longitudinal strain rate.
    
    Parameters
    ----------
    epsilon_xx : np.ndarray
        Longitudinal strain rate (∂u/∂x)
    threshold : float
        Threshold for distinguishing compression/extension (default: 0)
        
    Returns
    -------
    np.ndarray
        Integer array where:
        0 = compression (ε_xx < -threshold)
        1 = transition (-threshold ≤ ε_xx ≤ threshold) 
        2 = extension (ε_xx > threshold)
    """
    regime = np.full_like(epsilon_xx, 1, dtype=int)  # Default to transition
    regime[epsilon_xx < -threshold] = 0  # Compression
    regime[epsilon_xx > threshold] = 2   # Extension
    
    return regime


def compute_velocity_features(u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute velocity-based features for clustering.
    
    Parameters
    ----------
    u, v : np.ndarray
        Velocity components
        
    Returns
    -------
    Dict[str, np.ndarray]
        Velocity features:
        - 'speed': |v| (velocity magnitude)
        - 'direction': arctan2(v, u) (velocity direction)
        - 'log_speed': log(|v|) (log velocity magnitude)
    """
    # Velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    
    # Velocity direction (angle in radians)
    direction = np.arctan2(v, u)
    
    # Log velocity magnitude (for better scaling)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    log_speed = np.log(speed + epsilon)
    
    return {
        'speed': speed,
        'direction': direction, 
        'log_speed': log_speed
    }


def compute_deformation_features(strain_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute advanced deformation features from strain rate tensor.
    
    Parameters
    ----------
    strain_data : Dict[str, np.ndarray]
        Output from compute_strain_rate_tensor()
        
    Returns
    -------
    Dict[str, np.ndarray]
        Advanced deformation features:
        - 'dilatation_rate': ∇·v (same as divergence)
        - 'rotation_rate': ω = 0.5*(∂v/∂x - ∂u/∂y) (vorticity)
        - 'strain_anisotropy': |ε_xx - ε_yy| / ε_eff (strain anisotropy)
        - 'pure_shear_ratio': |ε_xy| / ε_eff (pure shear contribution)
    """
    epsilon_xx = strain_data['epsilon_xx']
    epsilon_yy = strain_data['epsilon_yy'] 
    epsilon_xy = strain_data['epsilon_xy']
    effective_strain = strain_data['effective_strain']
    
    # Dilatation rate (same as divergence)
    dilatation_rate = strain_data['divergence']
    
    # Note: We would need dvdx and dudy separately to compute rotation
    # For now, approximate using available data
    rotation_rate = np.zeros_like(epsilon_xx)  # Placeholder
    
    # Strain anisotropy: difference between normal strain rates
    strain_anisotropy = np.abs(epsilon_xx - epsilon_yy) / (effective_strain + 1e-10)
    
    # Pure shear contribution
    pure_shear_ratio = np.abs(epsilon_xy) / (effective_strain + 1e-10)
    
    return {
        'dilatation_rate': dilatation_rate,
        'rotation_rate': rotation_rate,
        'strain_anisotropy': strain_anisotropy,
        'pure_shear_ratio': pure_shear_ratio
    }


def compute_strain_invariants(epsilon_xx: np.ndarray, epsilon_yy: np.ndarray,
                             epsilon_xy: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute strain rate tensor invariants.
    
    Parameters
    ----------
    epsilon_xx, epsilon_yy : np.ndarray
        Normal strain rate components
    epsilon_xy : np.ndarray
        Shear strain rate component
        
    Returns
    -------
    Dict[str, np.ndarray]
        Strain rate invariants:
        - 'first_invariant': I₁ = tr(ε) = ε_xx + ε_yy
        - 'second_invariant': I₂ = det(ε) = ε_xx*ε_yy - ε_xy²
        - 'effective_strain': ε_eff = √(ε_xx² + ε_yy² + 2*ε_xy²)
    """
    # First invariant (trace)
    first_invariant = epsilon_xx + epsilon_yy
    
    # Second invariant (determinant)
    second_invariant = epsilon_xx * epsilon_yy - epsilon_xy**2
    
    # Effective strain rate (von Mises equivalent)
    effective_strain = np.sqrt(epsilon_xx**2 + epsilon_yy**2 + 2*epsilon_xy**2)
    
    return {
        'first_invariant': first_invariant,
        'second_invariant': second_invariant,
        'effective_strain': effective_strain
    }


def compute_flow_regime_indicators(strain_data: Dict[str, np.ndarray],
                                  velocity_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute indicators for different ice flow regimes.
    
    Parameters
    ----------
    strain_data : Dict[str, np.ndarray]
        Strain rate tensor data
    velocity_data : Dict[str, np.ndarray]
        Velocity field data
        
    Returns
    -------
    Dict[str, np.ndarray]
        Flow regime indicators:
        - 'compression_indicator': Measure of compressive flow
        - 'extension_indicator': Measure of extensional flow
        - 'shear_indicator': Measure of shear-dominated flow
    """
    epsilon_xx = strain_data['epsilon_xx']
    epsilon_yy = strain_data['epsilon_yy']
    shear_magnitude = strain_data['shear_magnitude']
    effective_strain = strain_data['effective_strain']
    speed = velocity_data['speed']
    
    # Compression indicator: negative longitudinal strain + high speed
    compression_indicator = -epsilon_xx * speed / (effective_strain + 1e-10)
    compression_indicator = np.maximum(compression_indicator, 0)  # Only positive values
    
    # Extension indicator: positive longitudinal strain + high speed  
    extension_indicator = epsilon_xx * speed / (effective_strain + 1e-10)
    extension_indicator = np.maximum(extension_indicator, 0)
    
    # Shear indicator: high shear relative to normal strain
    shear_indicator = shear_magnitude / (np.abs(epsilon_xx) + np.abs(epsilon_yy) + 1e-10)
    
    return {
        'compression_indicator': compression_indicator,
        'extension_indicator': extension_indicator,
        'shear_indicator': shear_indicator
    }


def validate_strain_features(strain_features: Dict[str, np.ndarray]) -> None:
    """
    Validate computed strain features for physical consistency.
    
    Parameters
    ----------
    strain_features : Dict[str, np.ndarray]
        Dictionary of computed strain features
        
    Raises
    ------
    Warning
        If features have unexpected properties
    """
    # Check effective strain is non-negative
    if 'effective_strain' in strain_features:
        eff_strain = strain_features['effective_strain']
        if np.any(eff_strain < 0):
            warnings.warn("Effective strain rate contains negative values")
    
    # Check for reasonable strain rate magnitudes (typical range: 1e-6 to 1e-2 /year)
    if 'epsilon_xx' in strain_features:
        epsilon_xx = strain_features['epsilon_xx']
        if np.any(np.abs(epsilon_xx) > 1e-1):
            warnings.warn("Longitudinal strain rates appear unusually large")
        
    # Check for NaN or infinite values
    for name, feature in strain_features.items():
        if np.any(~np.isfinite(feature)):
            warnings.warn(f"Feature '{name}' contains NaN or infinite values")


def create_strain_feature_summary(strain_features: Dict[str, np.ndarray]) -> str:
    """
    Create summary statistics for strain features.
    
    Parameters
    ----------
    strain_features : Dict[str, np.ndarray]
        Dictionary of strain features
        
    Returns
    -------
    str
        Summary string with statistics
    """
    summary_lines = ["Strain Feature Summary:"]
    summary_lines.append("-" * 40)
    
    for name, feature in strain_features.items():
        finite_mask = np.isfinite(feature)
        if not np.any(finite_mask):
            summary_lines.append(f"{name}: All NaN values")
            continue
            
        finite_vals = feature[finite_mask]
        summary_lines.append(
            f"{name}: "
            f"mean={np.mean(finite_vals):.2e}, "
            f"std={np.std(finite_vals):.2e}, "
            f"range=[{np.min(finite_vals):.2e}, {np.max(finite_vals):.2e}]"
        )
    
    return "\n".join(summary_lines)