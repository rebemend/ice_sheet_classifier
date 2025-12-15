import numpy as np
from typing import Dict, Tuple, Optional
import warnings


def compute_viscosity_features(mu: np.ndarray, eta: np.ndarray,
                              ice_thickness: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute viscosity-based features for ice shelf classification.
    
    Parameters
    ----------
    mu : np.ndarray
        Horizontal viscosity
    eta : np.ndarray
        Vertical viscosity
    ice_thickness : Optional[np.ndarray]
        Ice thickness for normalization (if available)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Viscosity features:
        - 'anisotropy': μ/η (anisotropy ratio)
        - 'log_mu': log(μ) (log horizontal viscosity)
        - 'log_eta': log(η) (log vertical viscosity)
        - 'log_anisotropy': log(μ/η) (log anisotropy ratio)
        - 'viscosity_contrast': (μ-η)/(μ+η) (normalized contrast)
        - 'mean_viscosity': (μ+η)/2 (average viscosity)
        - 'thickness_normalized_mu': μ/h (if thickness provided)
    """
    # Basic anisotropy ratio
    anisotropy = mu / (eta + 1e-12)  # Avoid division by zero
    
    # Log-transformed viscosities for better scaling
    # Handle negative and zero values properly
    epsilon = 1e-12
    log_mu = np.log(np.maximum(mu, epsilon))
    log_eta = np.log(np.maximum(eta, epsilon))
    log_anisotropy = np.log(np.maximum(anisotropy, epsilon))
    
    # Viscosity contrast (normalized difference)
    viscosity_contrast = (mu - eta) / (mu + eta + 1e-12)
    
    # Mean viscosity
    mean_viscosity = 0.5 * (mu + eta)
    
    features = {
        'anisotropy': anisotropy,
        'log_mu': log_mu,
        'log_eta': log_eta,
        'log_anisotropy': log_anisotropy,
        'viscosity_contrast': viscosity_contrast,
        'mean_viscosity': mean_viscosity
    }
    
    # Thickness-normalized features if thickness is provided
    if ice_thickness is not None:
        features['thickness_normalized_mu'] = mu / (ice_thickness + 1e-12)
        features['thickness_normalized_eta'] = eta / (ice_thickness + 1e-12)
    
    return features


def compute_viscosity_gradients(mu: np.ndarray, eta: np.ndarray,
                               x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute spatial gradients of viscosity fields.
    
    Parameters
    ----------
    mu, eta : np.ndarray
        Viscosity fields
    x, y : np.ndarray
        Coordinate arrays
        
    Returns
    -------
    Dict[str, np.ndarray]
        Viscosity gradient features:
        - 'dmu_dx', 'dmu_dy': Horizontal viscosity gradients
        - 'deta_dx', 'deta_dy': Vertical viscosity gradients
        - 'mu_gradient_magnitude': |∇μ|
        - 'eta_gradient_magnitude': |∇η|
        - 'anisotropy_gradient': ∇(μ/η)
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
    
    # Compute viscosity gradients using uniform spacing
    dmu_dx = np.gradient(mu, dx, axis=1)
    dmu_dy = np.gradient(mu, dy, axis=0)
    deta_dx = np.gradient(eta, dx, axis=1)
    deta_dy = np.gradient(eta, dy, axis=0)
    
    # Gradient magnitudes
    mu_gradient_magnitude = np.sqrt(dmu_dx**2 + dmu_dy**2)
    eta_gradient_magnitude = np.sqrt(deta_dx**2 + deta_dy**2)
    
    # Anisotropy gradient
    anisotropy = mu / (eta + 1e-12)
    daniso_dx = np.gradient(anisotropy, dx, axis=1)
    daniso_dy = np.gradient(anisotropy, dy, axis=0)
    anisotropy_gradient = np.sqrt(daniso_dx**2 + daniso_dy**2)
    
    return {
        'dmu_dx': dmu_dx,
        'dmu_dy': dmu_dy,
        'deta_dx': deta_dx,
        'deta_dy': deta_dy,
        'mu_gradient_magnitude': mu_gradient_magnitude,
        'eta_gradient_magnitude': eta_gradient_magnitude,
        'anisotropy_gradient': anisotropy_gradient
    }


def classify_rheology_regime(anisotropy: np.ndarray,
                           mu: np.ndarray,
                           anisotropy_threshold: float = 1.5,
                           mu_threshold: Optional[float] = None) -> np.ndarray:
    """
    Classify rheological regime based on viscosity characteristics.
    
    Parameters
    ----------
    anisotropy : np.ndarray
        Anisotropy ratio μ/η
    mu : np.ndarray
        Horizontal viscosity
    anisotropy_threshold : float
        Threshold for isotropic vs anisotropic classification
    mu_threshold : Optional[float]
        Threshold for high vs low viscosity (if None, use median)
        
    Returns
    -------
    np.ndarray
        Integer array where:
        0 = isotropic, low viscosity
        1 = isotropic, high viscosity  
        2 = anisotropic, low viscosity
        3 = anisotropic, high viscosity
    """
    if mu_threshold is None:
        mu_threshold = np.median(mu[np.isfinite(mu)])
    
    # Classify based on anisotropy and viscosity magnitude
    regime = np.zeros_like(anisotropy, dtype=int)
    
    # Isotropic regimes (anisotropy close to 1)
    isotropic_mask = np.abs(anisotropy - 1.0) < (anisotropy_threshold - 1.0)
    regime[isotropic_mask & (mu < mu_threshold)] = 0  # Isotropic, low viscosity
    regime[isotropic_mask & (mu >= mu_threshold)] = 1  # Isotropic, high viscosity
    
    # Anisotropic regimes
    anisotropic_mask = ~isotropic_mask
    regime[anisotropic_mask & (mu < mu_threshold)] = 2  # Anisotropic, low viscosity
    regime[anisotropic_mask & (mu >= mu_threshold)] = 3  # Anisotropic, high viscosity
    
    return regime


def compute_stress_features(mu: np.ndarray, eta: np.ndarray,
                          strain_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute stress-related features using viscosity and strain rate.
    
    Parameters
    ----------
    mu, eta : np.ndarray
        Viscosity fields
    strain_data : Dict[str, np.ndarray]
        Strain rate data (from strain_features module)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Stress features:
        - 'deviatoric_stress': σ' = 2μ*ε_eff (deviatoric stress magnitude)
        - 'stress_anisotropy': σ_xx/σ_yy ratio
        - 'stress_indicator': τ/σ_n (shear to normal stress ratio)
    """
    epsilon_xx = strain_data['epsilon_xx']
    epsilon_yy = strain_data['epsilon_yy'] 
    epsilon_xy = strain_data['epsilon_xy']
    effective_strain = strain_data['effective_strain']
    
    # Deviatoric stress magnitude (using horizontal viscosity)
    deviatoric_stress = 2 * mu * effective_strain
    
    # Stress components (simplified, assuming isotropic for principal stresses)
    sigma_xx = 2 * mu * epsilon_xx
    sigma_yy = 2 * eta * epsilon_yy  # Use vertical viscosity for transverse stress
    tau_xy = mu * epsilon_xy  # Shear stress
    
    # Stress anisotropy ratio
    stress_anisotropy = sigma_xx / (sigma_yy + 1e-12)
    
    # Stress indicator: ratio of shear to normal stress
    normal_stress_mag = np.sqrt(sigma_xx**2 + sigma_yy**2)
    stress_indicator = np.abs(tau_xy) / (normal_stress_mag + 1e-12)
    
    return {
        'deviatoric_stress': deviatoric_stress,
        'stress_anisotropy': stress_anisotropy, 
        'stress_indicator': stress_indicator,
        'sigma_xx': sigma_xx,
        'sigma_yy': sigma_yy,
        'tau_xy': tau_xy
    }


def compute_flow_law_parameters(mu: np.ndarray, eta: np.ndarray,
                               effective_strain: np.ndarray,
                               temperature: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute parameters related to ice flow law behavior.
    
    Parameters
    ----------
    mu, eta : np.ndarray
        Viscosity fields
    effective_strain : np.ndarray
        Effective strain rate
    temperature : Optional[np.ndarray]
        Ice temperature (if available)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Flow law parameters:
        - 'effective_viscosity': Effective viscosity for Glen's law
        - 'flow_enhancement': E = μ_ref/μ (enhancement factor)
        - 'strain_softening': Measure of strain rate dependence
    """
    # Effective viscosity (geometric mean of horizontal and vertical)
    effective_viscosity = np.sqrt(mu * eta)
    
    # Reference viscosity (median of effective viscosity)
    mu_ref = np.median(effective_viscosity[np.isfinite(effective_viscosity)])
    
    # Flow enhancement factor (deviation from reference)
    flow_enhancement = mu_ref / (effective_viscosity + 1e-12)
    
    # Strain softening indicator (high strain rate -> low viscosity)
    strain_softening = effective_strain / (effective_viscosity + 1e-12)
    
    features = {
        'effective_viscosity': effective_viscosity,
        'flow_enhancement': flow_enhancement,
        'strain_softening': strain_softening
    }
    
    # Temperature-dependent features if temperature is available
    if temperature is not None:
        # Temperature enhancement (Arrhenius relationship)
        T_ref = 263.15  # Reference temperature (K)
        Q = 139000  # Activation energy for ice (J/mol)
        R = 8.314   # Gas constant
        
        temp_enhancement = np.exp(-Q/R * (1/temperature - 1/T_ref))
        features['temperature_enhancement'] = temp_enhancement
    
    return features


def validate_viscosity_features(viscosity_features: Dict[str, np.ndarray]) -> None:
    """
    Validate computed viscosity features for physical consistency.
    
    Parameters
    ----------
    viscosity_features : Dict[str, np.ndarray]
        Dictionary of viscosity features
        
    Raises
    ------
    Warning
        If features have unexpected properties
    """
    # Check anisotropy ratio is positive
    if 'anisotropy' in viscosity_features:
        anisotropy = viscosity_features['anisotropy']
        if np.any(anisotropy <= 0):
            warnings.warn("Anisotropy ratio contains non-positive values")
        if np.any(anisotropy > 100):
            warnings.warn("Anisotropy ratio contains very large values (>100)")
    
    # Check viscosity values are reasonable (typical range: 1e12 to 1e16 Pa·s)
    for name in ['log_mu', 'log_eta']:
        if name in viscosity_features:
            log_visc = viscosity_features[name]
            # log(1e12) ≈ 27.6, log(1e16) ≈ 36.8
            if np.any(log_visc < 20) or np.any(log_visc > 40):
                warnings.warn(f"Feature '{name}' has values outside expected range")
    
    # Check for NaN or infinite values
    for name, feature in viscosity_features.items():
        if np.any(~np.isfinite(feature)):
            warnings.warn(f"Feature '{name}' contains NaN or infinite values")


def compute_viscosity_clustering_features(mu: np.ndarray, eta: np.ndarray,
                                        strain_data: Dict[str, np.ndarray],
                                        feature_set: str = 'standard') -> Dict[str, np.ndarray]:
    """
    Compute viscosity features optimized for k-means clustering.
    
    Parameters
    ----------
    mu, eta : np.ndarray
        Viscosity fields
    strain_data : Dict[str, np.ndarray]
        Strain rate data
    feature_set : str
        Feature set to compute ('standard', 'extended', 'minimal')
        
    Returns
    -------
    Dict[str, np.ndarray]
        Selected viscosity features for clustering
    """
    # Core viscosity features
    core_features = compute_viscosity_features(mu, eta)
    
    if feature_set == 'minimal':
        # Only essential features for clustering
        return {
            'anisotropy': core_features['anisotropy'],
            'log_mu': core_features['log_mu']
        }
    
    elif feature_set == 'standard':
        # Standard feature set for primary clustering
        return {
            'anisotropy': core_features['anisotropy'],
            'log_mu': core_features['log_mu'],
            'viscosity_contrast': core_features['viscosity_contrast'],
            'mean_viscosity': core_features['mean_viscosity']
        }
    
    elif feature_set == 'extended':
        # Extended feature set with stress information
        stress_features = compute_stress_features(mu, eta, strain_data)
        
        return {
            **core_features,
            'deviatoric_stress': stress_features['deviatoric_stress'],
            'stress_anisotropy': stress_features['stress_anisotropy'],
            'stress_indicator': stress_features['stress_indicator']
        }
    
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def create_viscosity_feature_summary(viscosity_features: Dict[str, np.ndarray]) -> str:
    """
    Create summary statistics for viscosity features.
    
    Parameters
    ----------
    viscosity_features : Dict[str, np.ndarray]
        Dictionary of viscosity features
        
    Returns
    -------
    str
        Summary string with statistics
    """
    summary_lines = ["Viscosity Feature Summary:"]
    summary_lines.append("-" * 40)
    
    for name, feature in viscosity_features.items():
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