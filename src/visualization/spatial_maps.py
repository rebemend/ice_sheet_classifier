import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple, Optional, List
import warnings


class SpatialMapVisualizer:
    """
    Create spatial maps for ice shelf clustering results.
    
    Visualizes cluster labels, features, and analysis results
    on the original spatial grid.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize spatial map visualizer.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Default figure size for plots
        """
        self.figsize = figsize
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_cluster_map(self, coordinates: np.ndarray, labels: np.ndarray,
                        grid_shape: Tuple[int, int], title: str = "Ice Shelf Cluster Map") -> plt.Figure:
        """
        Plot spatial map of cluster labels.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Spatial coordinates (n_points, 2) with [x, y]
        labels : np.ndarray
            Cluster labels (n_points,)
        grid_shape : Tuple[int, int]
            Original grid shape (ny, nx)
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Reshape labels to grid
        label_grid = self._reshape_to_grid(labels, coordinates, grid_shape)
        
        # Create discrete colormap
        n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude invalid labels
        colors = self.default_colors[:n_clusters] if n_clusters <= len(self.default_colors) else plt.cm.tab10.colors[:n_clusters]
        cmap = mcolors.ListedColormap(colors)
        
        # Plot cluster map
        im = ax.imshow(label_grid, cmap=cmap, aspect='equal', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cluster Label', fontsize=12)
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # Add grid coordinates if available
        if coordinates.shape[0] > 0:
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            x_extent = [np.min(x_coords), np.max(x_coords)]
            y_extent = [np.min(y_coords), np.max(y_coords)]
            im.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
        
        plt.tight_layout()
        return fig
    
    def plot_feature_map(self, coordinates: np.ndarray, feature_values: np.ndarray,
                        grid_shape: Tuple[int, int], feature_name: str,
                        cmap: str = 'viridis') -> plt.Figure:
        """
        Plot spatial map of feature values.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Spatial coordinates (n_points, 2)
        feature_values : np.ndarray
            Feature values to plot (n_points,)
        grid_shape : Tuple[int, int]
            Original grid shape
        feature_name : str
            Name of the feature for labeling
        cmap : str
            Colormap name
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Reshape feature values to grid
        feature_grid = self._reshape_to_grid(feature_values, coordinates, grid_shape)
        
        # Plot feature map
        im = ax.imshow(feature_grid, cmap=cmap, aspect='equal', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(feature_name, fontsize=12)
        
        # Customize plot
        ax.set_title(f'{feature_name} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # Set extent
        if coordinates.shape[0] > 0:
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            x_extent = [np.min(x_coords), np.max(x_coords)]
            y_extent = [np.min(y_coords), np.max(y_coords)]
            im.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_overlay(self, coordinates: np.ndarray, labels: np.ndarray,
                            feature_values: np.ndarray, grid_shape: Tuple[int, int],
                            feature_name: str, alpha: float = 0.7) -> plt.Figure:
        """
        Plot cluster boundaries overlaid on feature map.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Spatial coordinates
        labels : np.ndarray
            Cluster labels
        feature_values : np.ndarray
            Background feature values
        grid_shape : Tuple[int, int]
            Grid shape
        feature_name : str
            Feature name for background
        alpha : float
            Transparency of cluster boundaries
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot feature background
        feature_grid = self._reshape_to_grid(feature_values, coordinates, grid_shape)
        im1 = ax.imshow(feature_grid, cmap='viridis', aspect='equal', origin='lower', alpha=0.8)
        
        # Plot cluster boundaries
        label_grid = self._reshape_to_grid(labels, coordinates, grid_shape)
        
        # Create boundary mask
        boundary_mask = self._detect_cluster_boundaries(label_grid)
        
        # Overlay boundaries
        ax.contour(boundary_mask, levels=[0.5], colors='white', linewidths=2, alpha=alpha)
        
        # Add colorbar for features
        cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
        cbar1.set_label(feature_name, fontsize=12)
        
        # Customize plot
        ax.set_title(f'Cluster Boundaries on {feature_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # Set extent
        if coordinates.shape[0] > 0:
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            x_extent = [np.min(x_coords), np.max(x_coords)]
            y_extent = [np.min(y_coords), np.max(y_coords)]
            im1.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_grid(self, coordinates: np.ndarray, data_dict: Dict[str, np.ndarray],
                           grid_shape: Tuple[int, int], ncols: int = 3) -> plt.Figure:
        """
        Plot multiple maps in a grid for comparison.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Spatial coordinates
        data_dict : Dict[str, np.ndarray]
            Dictionary mapping names to data arrays
        grid_shape : Tuple[int, int]
            Grid shape
        ncols : int
            Number of columns in subplot grid
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        n_plots = len(data_dict)
        nrows = (n_plots + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        
        # Handle single row case
        if nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        for name, data in data_dict.items():
            row = plot_idx // ncols
            col = plot_idx % ncols
            ax = axes[row, col]
            
            # Reshape data to grid
            data_grid = self._reshape_to_grid(data, coordinates, grid_shape)
            
            # Determine colormap
            if 'cluster' in name.lower() or 'label' in name.lower():
                # Discrete colormap for clusters
                n_unique = len(np.unique(data[data >= 0]))
                colors = self.default_colors[:n_unique] if n_unique <= len(self.default_colors) else plt.cm.tab10.colors[:n_unique]
                cmap = mcolors.ListedColormap(colors)
            else:
                # Continuous colormap for features
                cmap = 'viridis'
            
            # Plot
            im = ax.imshow(data_grid, cmap=cmap, aspect='equal', origin='lower')
            ax.set_title(name, fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Set extent
            if coordinates.shape[0] > 0:
                x_coords = coordinates[:, 0]
                y_coords = coordinates[:, 1]
                x_extent = [np.min(x_coords), np.max(x_coords)]
                y_extent = [np.min(y_coords), np.max(y_coords)]
                im.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _reshape_to_grid(self, values: np.ndarray, coordinates: np.ndarray,
                        grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reshape 1D values array back to 2D grid using coordinates.
        
        Parameters
        ----------
        values : np.ndarray
            1D array of values
        coordinates : np.ndarray
            Spatial coordinates (n_points, 2)
        grid_shape : Tuple[int, int]
            Target grid shape (ny, nx)
            
        Returns
        -------
        np.ndarray
            2D grid with values
        """
        ny, nx = grid_shape
        
        if len(values) != len(coordinates):
            raise ValueError("Values and coordinates must have same length")
        
        # Create empty grid with NaN
        grid = np.full((ny, nx), np.nan)
        
        if len(coordinates) == 0:
            return grid
        
        # Get coordinate ranges
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]
        
        if len(np.unique(x_coords)) == 1 and len(np.unique(y_coords)) == 1:
            # Single point case
            grid[0, 0] = values[0] if len(values) > 0 else np.nan
            return grid
        
        # Map coordinates to grid indices
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        if x_max == x_min:
            x_indices = np.zeros(len(x_coords), dtype=int)
        else:
            x_indices = np.round((x_coords - x_min) / (x_max - x_min) * (nx - 1)).astype(int)
            x_indices = np.clip(x_indices, 0, nx - 1)
        
        if y_max == y_min:
            y_indices = np.zeros(len(y_coords), dtype=int)
        else:
            y_indices = np.round((y_coords - y_min) / (y_max - y_min) * (ny - 1)).astype(int)
            y_indices = np.clip(y_indices, 0, ny - 1)
        
        # Fill grid
        grid[y_indices, x_indices] = values
        
        return grid
    
    def _detect_cluster_boundaries(self, label_grid: np.ndarray) -> np.ndarray:
        """
        Detect cluster boundaries in label grid.
        
        Parameters
        ----------
        label_grid : np.ndarray
            2D array of cluster labels
            
        Returns
        -------
        np.ndarray
            Binary mask of boundaries
        """
        # Use gradient to detect boundaries
        grad_y, grad_x = np.gradient(label_grid)
        boundary_mask = (np.abs(grad_x) > 0) | (np.abs(grad_y) > 0)
        
        return boundary_mask.astype(float)
    
    def create_regime_interpretation_plot(self, coordinates: np.ndarray, labels: np.ndarray,
                                        strain_values: np.ndarray, grid_shape: Tuple[int, int]) -> plt.Figure:
        """
        Create interpretation plot showing ice flow regimes.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Spatial coordinates
        labels : np.ndarray
            Cluster labels
        strain_values : np.ndarray
            Longitudinal strain rate values (∂u/∂x)
        grid_shape : Tuple[int, int]
            Grid shape
            
        Returns
        -------
        plt.Figure
            Interpretation figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Cluster map with regime labels
        label_grid = self._reshape_to_grid(labels, coordinates, grid_shape)
        
        # Assume k=3 with physical interpretation
        regime_colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
        regime_labels = ['Compression', 'Transition', 'Extension']
        cmap = mcolors.ListedColormap(regime_colors)
        
        im1 = ax1.imshow(label_grid, cmap=cmap, aspect='equal', origin='lower')
        ax1.set_title('Ice Flow Regimes', fontsize=14, fontweight='bold')
        
        # Custom colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_ticks([0, 1, 2])
        cbar1.set_ticklabels(regime_labels)
        
        # Plot 2: Strain rate field
        strain_grid = self._reshape_to_grid(strain_values, coordinates, grid_shape)
        im2 = ax2.imshow(strain_grid, cmap='RdBu_r', aspect='equal', origin='lower')
        ax2.set_title('Longitudinal Strain Rate (∂u/∂x)', fontsize=14, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Strain Rate (s⁻¹)', fontsize=12)
        
        # Plot 3: Overlay
        ax3.imshow(strain_grid, cmap='RdBu_r', aspect='equal', origin='lower', alpha=0.7)
        boundary_mask = self._detect_cluster_boundaries(label_grid)
        ax3.contour(boundary_mask, levels=[0.5], colors='black', linewidths=3)
        ax3.set_title('Regime Boundaries on Strain Field', fontsize=14, fontweight='bold')
        
        # Set extents for all plots
        if coordinates.shape[0] > 0:
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            x_extent = [np.min(x_coords), np.max(x_coords)]
            y_extent = [np.min(y_coords), np.max(y_coords)]
            
            for im in [im1, im2]:
                im.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
        
        # Common formatting
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
        
        plt.tight_layout()
        return fig


def create_spatial_analysis_summary(coordinates: np.ndarray, labels: np.ndarray,
                                  features_dict: Dict[str, np.ndarray], 
                                  grid_shape: Tuple[int, int],
                                  save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Create comprehensive spatial analysis summary.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Spatial coordinates
    labels : np.ndarray
        Cluster labels
    features_dict : Dict[str, np.ndarray]
        Dictionary of feature arrays
    grid_shape : Tuple[int, int]
        Grid shape
    save_path : Optional[str]
        Base path for saving plots
        
    Returns
    -------
    List[plt.Figure]
        List of created figures
    """
    visualizer = SpatialMapVisualizer()
    figures = []
    
    # Main cluster map
    print("Creating cluster map...")
    fig1 = visualizer.plot_cluster_map(coordinates, labels, grid_shape, 
                                      "Ice Shelf Cluster Classification")
    figures.append(fig1)
    if save_path:
        fig1.savefig(f"{save_path}_clusters.png", dpi=300, bbox_inches='tight')
    
    # Feature maps
    key_features = ['dudx', 'speed', 'mu', 'anisotropy']
    available_features = {k: v for k, v in features_dict.items() if k in key_features}
    
    if available_features:
        print("Creating feature comparison...")
        fig2 = visualizer.plot_comparison_grid(coordinates, available_features, grid_shape)
        figures.append(fig2)
        if save_path:
            fig2.savefig(f"{save_path}_features.png", dpi=300, bbox_inches='tight')
    
    # Physical interpretation (if strain rate available)
    if 'dudx' in features_dict:
        print("Creating regime interpretation...")
        fig3 = visualizer.create_regime_interpretation_plot(
            coordinates, labels, features_dict['dudx'], grid_shape
        )
        figures.append(fig3)
        if save_path:
            fig3.savefig(f"{save_path}_regimes.png", dpi=300, bbox_inches='tight')
    
    return figures