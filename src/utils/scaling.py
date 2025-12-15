import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


class FeatureScaler:
    """
    Feature scaling utilities for k-means clustering preprocessing.
    
    Provides different scaling methods and handles invalid values properly.
    """
    
    def __init__(self, method: str = 'standard', handle_invalid: str = 'remove'):
        """
        Initialize feature scaler.
        
        Parameters
        ----------
        method : str
            Scaling method: 'standard', 'minmax', 'robust', or 'none'
        handle_invalid : str
            How to handle invalid values: 'remove', 'impute', or 'error'
        """
        self.method = method
        self.handle_invalid = handle_invalid
        self.scaler = None
        self.feature_names = None
        self.valid_mask = None
        self.is_fitted = False
        
        # Initialize scaler based on method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def _validate_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and handle invalid feature values.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array (n_samples, n_features)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cleaned features and boolean mask of valid samples
        """
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")
        
        # Find invalid values (NaN, inf, -inf)
        valid_mask = np.all(np.isfinite(features), axis=1)
        
        if self.handle_invalid == 'error' and not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            raise ValueError(f"Found {n_invalid} samples with invalid values")
        
        elif self.handle_invalid == 'remove':
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                warnings.warn(f"Removing {n_invalid} samples with invalid values")
            return features[valid_mask], valid_mask
        
        elif self.handle_invalid == 'impute':
            # Simple median imputation
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                warnings.warn(f"Imputing {n_invalid} samples with invalid values using median")
                
                features_clean = features.copy()
                for i in range(features.shape[1]):
                    col_valid = np.isfinite(features[:, i])
                    if np.any(col_valid):
                        median_val = np.median(features[col_valid, i])
                        features_clean[~col_valid, i] = median_val
                
                return features_clean, np.ones(features.shape[0], dtype=bool)
            
            return features, valid_mask
        
        else:
            raise ValueError(f"Unknown invalid handling method: {self.handle_invalid}")
    
    def fit(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureScaler':
        """
        Fit the scaler to the feature data.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array (n_samples, n_features)
        feature_names : Optional[List[str]]
            Names of features for reference
            
        Returns
        -------
        FeatureScaler
            Fitted scaler instance
        """
        # Validate and clean features
        features_clean, valid_mask = self._validate_features(features)
        
        if features_clean.shape[0] == 0:
            raise ValueError("No valid samples remaining after cleaning")
        
        # Store information
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        self.valid_mask = valid_mask
        
        # Fit scaler if using scaling
        if self.scaler is not None:
            self.scaler.fit(features_clean)
        
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array to transform
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Validate features
        features_clean, _ = self._validate_features(features)
        
        if features_clean.shape[0] == 0:
            warnings.warn("No valid samples to transform")
            return np.array([]).reshape(0, features.shape[1])
        
        # Apply scaling
        if self.scaler is not None:
            return self.scaler.transform(features_clean)
        else:
            return features_clean
    
    def fit_transform(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit scaler and transform features in one step.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array (n_samples, n_features)
        feature_names : Optional[List[str]]
            Names of features
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        return self.fit(features, feature_names).transform(features)
    
    def inverse_transform(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Parameters
        ----------
        features_scaled : np.ndarray
            Scaled feature array
            
        Returns
        -------
        np.ndarray
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        if self.scaler is not None:
            return self.scaler.inverse_transform(features_scaled)
        else:
            return features_scaled
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the fitted scaler.
        
        Returns
        -------
        Dict
            Scaler information including feature statistics
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        info = {
            "method": self.method,
            "handle_invalid": self.handle_invalid,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "feature_names": self.feature_names,
            "n_valid_samples": np.sum(self.valid_mask) if self.valid_mask is not None else None
        }
        
        # Add scaler-specific information
        if self.scaler is not None and hasattr(self.scaler, 'mean_'):
            # StandardScaler
            if hasattr(self.scaler, 'mean_'):
                info['means'] = self.scaler.mean_
                info['scales'] = self.scaler.scale_
        
        return info


def create_scaler(method: str = 'standard', **kwargs) -> FeatureScaler:
    """
    Create a feature scaler with specified method.
    
    Parameters
    ----------
    method : str
        Scaling method
    **kwargs
        Additional arguments for FeatureScaler
        
    Returns
    -------
    FeatureScaler
        Configured scaler instance
    """
    return FeatureScaler(method=method, **kwargs)


def scale_features_for_clustering(features: np.ndarray, 
                                 feature_names: Optional[List[str]] = None,
                                 method: str = 'standard') -> Tuple[np.ndarray, FeatureScaler, np.ndarray]:
    """
    Scale features for k-means clustering with validation.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array (n_samples, n_features)
    feature_names : Optional[List[str]]
        Feature names
    method : str
        Scaling method
        
    Returns
    -------
    Tuple[np.ndarray, FeatureScaler, np.ndarray]
        Scaled features, fitted scaler, and valid sample mask
    """
    # Create and fit scaler
    scaler = FeatureScaler(method=method, handle_invalid='remove')
    
    # Get valid mask before transformation
    _, valid_mask = scaler._validate_features(features)
    
    # Fit and transform
    features_scaled = scaler.fit_transform(features, feature_names)
    
    return features_scaled, scaler, valid_mask


def validate_scaled_features(features_scaled: np.ndarray, 
                           original_features: np.ndarray,
                           scaler: FeatureScaler) -> Dict[str, bool]:
    """
    Validate that feature scaling was successful.
    
    Parameters
    ----------
    features_scaled : np.ndarray
        Scaled features
    original_features : np.ndarray
        Original features
    scaler : FeatureScaler
        Fitted scaler
        
    Returns
    -------
    Dict[str, bool]
        Validation results
    """
    results = {}
    
    # Check no NaN or inf values
    results['no_invalid_values'] = np.all(np.isfinite(features_scaled))
    
    # Check feature scaling worked (for standard scaling)
    if scaler.method == 'standard' and features_scaled.shape[0] > 1:
        means = np.mean(features_scaled, axis=0)
        stds = np.std(features_scaled, axis=0)
        
        # Should have approximately zero mean and unit variance
        results['zero_mean'] = np.allclose(means, 0, atol=1e-10)
        results['unit_variance'] = np.allclose(stds, 1, atol=1e-10)
    else:
        results['zero_mean'] = True
        results['unit_variance'] = True
    
    # Check shapes are consistent
    valid_mask = scaler.valid_mask if scaler.valid_mask is not None else np.ones(original_features.shape[0], dtype=bool)
    expected_rows = np.sum(valid_mask)
    results['correct_shape'] = features_scaled.shape[0] == expected_rows
    
    # Check no constant features (zero variance)
    if features_scaled.shape[0] > 1:
        variances = np.var(features_scaled, axis=0)
        results['no_constant_features'] = np.all(variances > 1e-12)
    else:
        results['no_constant_features'] = True
    
    return results


def get_scaling_summary(features_original: np.ndarray,
                       features_scaled: np.ndarray,
                       feature_names: Optional[List[str]] = None) -> str:
    """
    Create summary of scaling transformation.
    
    Parameters
    ----------
    features_original : np.ndarray
        Original features
    features_scaled : np.ndarray
        Scaled features
    feature_names : Optional[List[str]]
        Feature names
        
    Returns
    -------
    str
        Summary string
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(features_original.shape[1])]
    
    lines = ["Feature Scaling Summary:"]
    lines.append("-" * 50)
    lines.append(f"Original shape: {features_original.shape}")
    lines.append(f"Scaled shape: {features_scaled.shape}")
    lines.append("")
    
    # Per-feature statistics
    lines.append("Feature Statistics:")
    lines.append("Name".ljust(20) + "Original Mean".ljust(15) + "Original Std".ljust(15) + 
                "Scaled Mean".ljust(15) + "Scaled Std".ljust(15))
    lines.append("-" * 80)
    
    for i, name in enumerate(feature_names):
        if i < features_original.shape[1] and i < features_scaled.shape[1]:
            orig_mean = np.mean(features_original[:, i])
            orig_std = np.std(features_original[:, i])
            
            if features_scaled.shape[0] > 0:
                scaled_mean = np.mean(features_scaled[:, i])
                scaled_std = np.std(features_scaled[:, i])
            else:
                scaled_mean = scaled_std = np.nan
            
            lines.append(
                f"{name[:19]:20s} "
                f"{orig_mean:14.2e} "
                f"{orig_std:14.2e} "
                f"{scaled_mean:14.2e} "
                f"{scaled_std:14.2e}"
            )
    
    return "\n".join(lines)


def recommend_scaling_method(features: np.ndarray, 
                           feature_names: Optional[List[str]] = None) -> str:
    """
    Recommend appropriate scaling method based on feature characteristics.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array
    feature_names : Optional[List[str]]
        Feature names
        
    Returns
    -------
    str
        Recommended scaling method
    """
    # Remove invalid values for analysis
    valid_mask = np.all(np.isfinite(features), axis=1)
    if not np.any(valid_mask):
        return 'none'  # No valid data
    
    features_clean = features[valid_mask]
    
    # Analyze feature distributions
    means = np.mean(features_clean, axis=0)
    stds = np.std(features_clean, axis=0)
    ranges = np.ptp(features_clean, axis=0)  # peak-to-peak range
    
    # Check for very different scales
    scale_ratios = np.max(ranges) / (np.min(ranges) + 1e-12)
    
    # Check for potential outliers (high standard deviation relative to range)
    outlier_indicators = stds / (ranges + 1e-12)
    high_outlier_risk = np.any(outlier_indicators > 0.3)
    
    if scale_ratios > 1000:
        # Very different scales - standard scaling recommended
        if high_outlier_risk:
            return 'robust'  # Robust scaling if outliers detected
        else:
            return 'standard'
    elif scale_ratios > 10:
        # Moderate scale differences
        return 'standard'
    else:
        # Similar scales - minimal scaling needed
        return 'minmax'