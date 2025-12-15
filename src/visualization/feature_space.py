import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings


class FeatureSpaceVisualizer:
    """
    Visualize clustering results in feature space.
    
    Creates scatter plots, PCA projections, and other feature space
    visualizations to understand cluster structure and separation.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize feature space visualizer.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_feature_pairs(self, features: np.ndarray, labels: np.ndarray,
                          feature_names: List[str], max_pairs: int = 6) -> plt.Figure:
        """
        Plot pairwise feature scatter plots colored by clusters.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array (n_samples, n_features)
        labels : np.ndarray
            Cluster labels (n_samples,)
        feature_names : List[str]
            Names of features
        max_pairs : int
            Maximum number of feature pairs to plot
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        n_features = features.shape[1]
        
        if n_features < 2:
            raise ValueError("Need at least 2 features for pairwise plots")
        
        # Select most interesting feature pairs
        if n_features > max_pairs:
            # Use PCA to select most informative features
            pca = PCA(n_components=min(max_pairs, n_features))
            pca.fit(features)
            
            # Select features with highest variance contributions
            feature_importance = np.sum(np.abs(pca.components_), axis=0)
            top_indices = np.argsort(feature_importance)[-max_pairs:]
        else:
            top_indices = list(range(n_features))
        
        selected_features = features[:, top_indices]
        selected_names = [feature_names[i] for i in top_indices]
        n_selected = len(selected_names)
        
        # Create pairplot
        fig, axes = plt.subplots(n_selected, n_selected, figsize=(3*n_selected, 3*n_selected))
        
        if n_selected == 1:
            axes = np.array([[axes]])
        elif n_selected == 2:
            if axes.ndim == 1:
                axes = axes.reshape(2, 1)
        
        unique_labels = np.unique(labels[labels >= 0])  # Exclude invalid labels
        colors = self.default_colors[:len(unique_labels)]
        
        for i in range(n_selected):
            for j in range(n_selected):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograms
                    for label, color in zip(unique_labels, colors):
                        mask = labels == label
                        ax.hist(selected_features[mask, i], alpha=0.6, color=color, 
                               bins=20, label=f'Cluster {label}')
                    ax.set_xlabel(selected_names[i])
                    ax.set_ylabel('Frequency')
                    if i == 0:
                        ax.legend()
                
                else:
                    # Off-diagonal: scatter plots
                    for label, color in zip(unique_labels, colors):
                        mask = labels == label
                        ax.scatter(selected_features[mask, j], selected_features[mask, i],
                                 alpha=0.6, color=color, s=20, label=f'Cluster {label}')
                    
                    ax.set_xlabel(selected_names[j])
                    ax.set_ylabel(selected_names[i])
                    
                    if i == 0 and j == n_selected - 1:
                        ax.legend()
        
        plt.suptitle('Feature Space Clustering Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_pca_projection(self, features: np.ndarray, labels: np.ndarray,
                           feature_names: List[str], n_components: int = 2) -> plt.Figure:
        """
        Plot PCA projection of features colored by clusters.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array
        labels : np.ndarray
            Cluster labels
        feature_names : List[str]
            Feature names
        n_components : int
            Number of PCA components (2 or 3)
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        
        unique_labels = np.unique(labels[labels >= 0])
        colors = self.default_colors[:len(unique_labels)]
        
        if n_components == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot PCA scatter
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax1.scatter(features_pca[mask, 0], features_pca[mask, 1],
                           alpha=0.6, color=color, s=30, label=f'Cluster {label}')
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax1.set_title('PCA Projection of Clusters')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot explained variance
            pc_numbers = np.arange(1, len(pca.explained_variance_ratio_) + 1)
            ax2.bar(pc_numbers, pca.explained_variance_ratio_, alpha=0.7)
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Explained Variance Ratio')
            ax2.set_title('PCA Explained Variance')
            ax2.grid(True, alpha=0.3)
            
        else:  # 3D case
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax.scatter(features_pca[mask, 0], features_pca[mask, 1], features_pca[mask, 2],
                          alpha=0.6, color=color, s=30, label=f'Cluster {label}')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title('3D PCA Projection of Clusters')
            ax.legend()
        
        # Add component interpretation
        self._add_pca_interpretation(fig, pca, feature_names)
        
        plt.tight_layout()
        return fig
    
    def plot_tsne_projection(self, features: np.ndarray, labels: np.ndarray,
                            perplexity: float = 30, random_state: int = 42) -> plt.Figure:
        """
        Plot t-SNE projection of features colored by clusters.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array
        labels : np.ndarray
            Cluster labels
        perplexity : float
            t-SNE perplexity parameter
        random_state : int
            Random state for reproducibility
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        if features.shape[0] < perplexity * 3:
            perplexity = max(5, features.shape[0] // 3)
            warnings.warn(f"Reduced perplexity to {perplexity} due to small sample size")
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        features_tsne = tsne.fit_transform(features)
        
        unique_labels = np.unique(labels[labels >= 0])
        colors = self.default_colors[:len(unique_labels)]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                      alpha=0.6, color=color, s=30, label=f'Cluster {label}')
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(f't-SNE Projection (perplexity={perplexity})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_centroids(self, centroids: np.ndarray, feature_names: List[str]) -> plt.Figure:
        """
        Plot cluster centroids in feature space.
        
        Parameters
        ----------
        centroids : np.ndarray
            Cluster centroids (n_clusters, n_features)
        feature_names : List[str]
            Feature names
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        n_clusters, n_features = centroids.shape
        
        if n_features == 1:
            # 1D case: bar plot
            fig, ax = plt.subplots(figsize=(8, 6))
            cluster_labels = [f'Cluster {i}' for i in range(n_clusters)]
            bars = ax.bar(cluster_labels, centroids[:, 0], 
                         color=self.default_colors[:n_clusters], alpha=0.7)
            ax.set_ylabel(feature_names[0])
            ax.set_title('Cluster Centroids')
            
            # Add value labels on bars
            for bar, value in zip(bars, centroids[:, 0]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{value:.2e}', ha='center', va='bottom')
        
        elif n_features == 2:
            # 2D case: scatter plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            colors = self.default_colors[:n_clusters]
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
            
            # Add cluster labels
            for i, (x, y) in enumerate(centroids):
                ax.annotate(f'C{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title('Cluster Centroids in Feature Space')
            ax.grid(True, alpha=0.3)
        
        else:
            # Multi-dimensional: parallel coordinates plot
            fig, ax = plt.subplots(figsize=(max(8, n_features * 1.5), 6))
            
            x_positions = np.arange(n_features)
            colors = self.default_colors[:n_clusters]
            
            for i, centroid in enumerate(centroids):
                ax.plot(x_positions, centroid, 'o-', color=colors[i], 
                       linewidth=2, markersize=8, alpha=0.8, label=f'Cluster {i}')
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(feature_names, rotation=45)
            ax.set_ylabel('Feature Value (Standardized)')
            ax.set_title('Cluster Centroids - Parallel Coordinates')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette_analysis(self, features: np.ndarray, labels: np.ndarray,
                                silhouette_samples: np.ndarray) -> plt.Figure:
        """
        Plot silhouette analysis for cluster evaluation.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array
        labels : np.ndarray
            Cluster labels
        silhouette_samples : np.ndarray
            Per-sample silhouette scores
            
        Returns
        -------
        plt.Figure
            Created figure
        """
        from sklearn.metrics import silhouette_score
        
        n_clusters = len(np.unique(labels[labels >= 0]))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Silhouette plot
        y_lower = 10
        silhouette_avg = silhouette_score(features, labels)
        
        for i in range(n_clusters):
            cluster_silhouette_values = silhouette_samples[labels == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = self.default_colors[i % len(self.default_colors)]
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label clusters
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette Coefficient Values')
        ax1.set_ylabel('Cluster Label')
        ax1.set_title('Silhouette Plot for Individual Clusters')
        
        # Add average silhouette line
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        ax1.legend()
        
        # Plot 2: Cluster scatter colored by silhouette score
        if features.shape[1] >= 2:
            scatter = ax2.scatter(features[:, 0], features[:, 1], 
                                c=silhouette_samples, cmap='viridis', alpha=0.7)
            ax2.set_xlabel('Feature 1')
            ax2.set_ylabel('Feature 2') 
            ax2.set_title('Silhouette Scores in Feature Space')
            plt.colorbar(scatter, ax=ax2, label='Silhouette Score')
        else:
            # 1D case: histogram
            ax2.hist(silhouette_samples, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(silhouette_avg, color='red', linestyle='--', 
                       label=f'Average: {silhouette_avg:.3f}')
            ax2.set_xlabel('Silhouette Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Silhouette Scores')
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def _add_pca_interpretation(self, fig: plt.Figure, pca: PCA, 
                               feature_names: List[str]) -> None:
        """
        Add PCA component interpretation to figure.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to add interpretation to
        pca : PCA
            Fitted PCA object
        feature_names : List[str]
            Feature names
        """
        # Create interpretation text
        interpretation_lines = ["PCA Component Interpretation:"]
        
        for i, component in enumerate(pca.components_):
            var_explained = pca.explained_variance_ratio_[i]
            interpretation_lines.append(f"PC{i+1} ({var_explained:.1%} variance):")
            
            # Find most important features for this component
            abs_loadings = np.abs(component)
            important_indices = np.argsort(abs_loadings)[-3:][::-1]  # Top 3
            
            for j in important_indices:
                loading = component[j]
                sign = "+" if loading > 0 else "-"
                interpretation_lines.append(f"  {sign} {feature_names[j]} ({loading:.2f})")
        
        # Add text to figure
        interpretation_text = "\n".join(interpretation_lines)
        fig.text(0.02, 0.02, interpretation_text, fontsize=8, 
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.8))


def create_feature_space_summary(features: np.ndarray, labels: np.ndarray,
                                feature_names: List[str], centroids: np.ndarray,
                                silhouette_samples: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Create comprehensive feature space analysis summary.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array (scaled)
    labels : np.ndarray
        Cluster labels
    feature_names : List[str]
        Feature names
    centroids : np.ndarray
        Cluster centroids
    silhouette_samples : Optional[np.ndarray]
        Per-sample silhouette scores
    save_path : Optional[str]
        Base path for saving plots
        
    Returns
    -------
    List[plt.Figure]
        List of created figures
    """
    visualizer = FeatureSpaceVisualizer()
    figures = []
    
    # Feature pairs plot
    if features.shape[1] >= 2:
        print("Creating feature pairs plot...")
        fig1 = visualizer.plot_feature_pairs(features, labels, feature_names)
        figures.append(fig1)
        if save_path:
            fig1.savefig(f"{save_path}_feature_pairs.png", dpi=300, bbox_inches='tight')
    
    # PCA projection
    if features.shape[1] >= 2:
        print("Creating PCA projection...")
        fig2 = visualizer.plot_pca_projection(features, labels, feature_names)
        figures.append(fig2)
        if save_path:
            fig2.savefig(f"{save_path}_pca.png", dpi=300, bbox_inches='tight')
    
    # t-SNE projection (for larger datasets)
    if features.shape[0] >= 30 and features.shape[1] >= 2:
        try:
            print("Creating t-SNE projection...")
            fig3 = visualizer.plot_tsne_projection(features, labels)
            figures.append(fig3)
            if save_path:
                fig3.savefig(f"{save_path}_tsne.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            warnings.warn(f"t-SNE projection failed: {e}")
    
    # Centroids plot
    print("Creating centroids plot...")
    fig4 = visualizer.plot_cluster_centroids(centroids, feature_names)
    figures.append(fig4)
    if save_path:
        fig4.savefig(f"{save_path}_centroids.png", dpi=300, bbox_inches='tight')
    
    # Silhouette analysis
    if silhouette_samples is not None:
        print("Creating silhouette analysis...")
        fig5 = visualizer.plot_silhouette_analysis(features, labels, silhouette_samples)
        figures.append(fig5)
        if save_path:
            fig5.savefig(f"{save_path}_silhouette.png", dpi=300, bbox_inches='tight')
    
    return figures