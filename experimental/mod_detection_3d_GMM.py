
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Tuple

class ModificationAnalyzer:
    def __init__(self, n_components=3):
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type='full',
                                  random_state=42)
        
    def prepare_features(self, signal: np.ndarray, dwell_time: np.ndarray, 
                        std_dev: np.ndarray) -> np.ndarray:
        """Prepare and normalize 3D feature space."""
        features = np.vstack([
            signal,
            np.log(dwell_time),
            std_dev
        ]).T
        return features
    
    def fit_combined_data(self, sample_dict: Dict[str, pd.DataFrame], 
                         samples_per_condition: int = 1000) -> None:
        """
        Fit GMM on combined data from all samples with equal sampling.
        
        Parameters:
        -----------
        sample_dict : Dict[str, pd.DataFrame]
            Dictionary with sample names as keys and DataFrames as values.
        samples_per_condition : int
            Number of samples to take from each condition for fitting
        """
        # Collect features from each sample with equal sampling
        sampled_features = []
        
        for sample_name, data in sample_dict.items():
            features = self.prepare_features(
                data['signal'].values,
                data['dwell_time'].values,
                data['std_dev'].values
            )
            
            # Randomly sample from this condition
            n_available = len(features)
            n_samples = min(samples_per_condition, n_available)
            indices = np.random.choice(n_available, n_samples, replace=False)
            sampled_features.append(features[indices])
        
        # Combine all sampled features
        X_combined = np.vstack(sampled_features)
        
        # Fit scaler and GMM on combined data
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        self.gmm.fit(X_combined_scaled)
    
    def analyze_samples(self, sample_dict: Dict[str, pd.DataFrame], 
                       control_map: Dict[str, str]) -> Dict[str, Dict]:
        """
        Analyze modification rates for samples relative to their specific controls.
        
        Parameters:
        -----------
        sample_dict : Dict[str, pd.DataFrame]
            Dictionary with sample names as keys and DataFrames as values
        control_map : Dict[str, str]
            Dictionary mapping sample names to their respective control names
            
        Returns:
        --------
        Dict with sample names as keys and dictionaries containing:
            - mod_rate: overall modification rate
            - cluster_proportions: proportion of reads in each cluster
            - cluster_changes: changes relative to control for each cluster
        """
        results = {}
        
        # Analyze each sample relative to its specific control
        for sample_name, control_name in control_map.items():
            if sample_name not in sample_dict or control_name not in sample_dict:
                continue
                
            # Get control proportions
            control_data = sample_dict[control_name]
            X_control = self.prepare_features(
                control_data['signal'].values,
                control_data['dwell_time'].values,
                control_data['std_dev'].values
            )
            X_control_scaled = self.scaler.transform(X_control)
            control_proba = self.gmm.predict_proba(X_control_scaled)
            control_proportions = control_proba.mean(axis=0)
            
            # Analyze treatment sample
            sample_data = sample_dict[sample_name]
            X_sample = self.prepare_features(
                sample_data['signal'].values,
                sample_data['dwell_time'].values,
                sample_data['std_dev'].values
            )
            X_sample_scaled = self.scaler.transform(X_sample)
            sample_proba = self.gmm.predict_proba(X_sample_scaled)
            sample_proportions = sample_proba.mean(axis=0)
            
            # Calculate differences from control
            diffs = sample_proportions - control_proportions
            
            # Calculate modification rate as sum of positive changes
            mod_rate = np.sum(np.maximum(diffs, 0))

            # Run some more (orthogonal) test statistics for a more complete picture
            def nan_ttest(s1, s2):
                return -np.log10(ttest_ind(s1[~np.isnan(s1)], s2[~np.isnan(s2)]).pvalue)
            
            def nan_kstest(s1, s2):
                return -np.log10(kstest_ind(s1[~np.isnan(s1)], s2[~np.isnan(s2)]).pvalue)
            
            


            results[sample_name] = {
                'mod_rate': mod_rate,
                'cluster_proportions': sample_proportions,
                'cluster_changes': diffs,
                'cluster_probabilities': sample_proba,
                'num_observations': len(sample_data),
                'control_name': control_name,
                'control_proportions': control_proportions
            }
            
        return results

def analyze_modifications(samples: Dict[str, pd.DataFrame], 
                        control_map: Dict[str, str],
                        n_components: int = 3,
                        samples_per_condition: int = 1000) -> Dict[str, Dict]:
    """
    Analyze modification rates across multiple samples.
    
    Parameters:
    -----------
    samples : Dict[str, pd.DataFrame]
        Dictionary with sample names as keys and DataFrames as values.
        Each DataFrame should have columns ['signal', 'dwell_time', 'std_dev']
    control_map : Dict[str, str]
        Dictionary mapping sample names to their respective control names
    n_components : int
        Number of Gaussian components to fit
    samples_per_condition : int
        Number of samples to take from each condition for fitting
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with analysis results for each sample
    """
    analyzer = ModificationAnalyzer(n_components=n_components)
    
    # Fit model on sampled data from all conditions
    analyzer.fit_combined_data(samples, samples_per_condition)
    
    # Analyze each sample relative to its specific control
    results = analyzer.analyze_samples(samples, control_map)
    
    return results

# Example usage:
"""
samples = {
    'EtOH_5mM': control_5mM_df,
    'glyoxal_5mM': glyoxal_5mM_df,
    'EtOH_10mM': control_10mM_df,
    'glyoxal_10mM': glyoxal_10mM_df
}

control_map = {
    'glyoxal_5mM': 'EtOH_5mM',
    'glyoxal_10mM': 'EtOH_10mM'
}

results = analyze_modifications(
    samples, 
    control_map=control_map,
    samples_per_condition=1000  # Number of reads to sample from each condition
)

# Access results for a specific sample:
mod_rate = results['glyoxal_5mM']['mod_rate']
cluster_props = results['glyoxal_5mM']['cluster_proportions']
cluster_changes = results['glyoxal_5mM']['cluster_changes']
"""







import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import multivariate_normal

def create_gaussian_surface_points(mean, cov, n_points=50):
    """
    Create points for visualizing a 3D Gaussian surface at 2 standard deviations.
    
    Parameters:
    -----------
    mean : array-like, shape (3,)
        Mean of the Gaussian
    cov : array-like, shape (3, 3)
        Covariance matrix
    n_points : int
        Number of points per dimension for the surface
        
    Returns:
    --------
    tuple of (xs, ys, zs, values)
        Grid coordinates and probability density values
    """
    # Create grid
    std_dev = np.sqrt(np.diag(cov))
    x = np.linspace(mean[0] - 2*std_dev[0], mean[0] + 2*std_dev[0], n_points)
    y = np.linspace(mean[1] - 2*std_dev[1], mean[1] + 2*std_dev[1], n_points)
    z = np.linspace(mean[2] - 2*std_dev[2], mean[2] + 2*std_dev[2], n_points)
    
    # Create meshgrid
    xs, ys, zs = np.meshgrid(x, y, z)
    
    # Calculate probability density
    pos = np.empty(xs.shape + (3,))
    pos[:, :, :, 0] = xs
    pos[:, :, :, 1] = ys
    pos[:, :, :, 2] = zs
    
    rv = multivariate_normal(mean, cov)
    values = rv.pdf(pos)
    
    return xs, ys, zs, values

def plot_3d_gaussians(analyzer, samples_dict, sample_name):
    """
    Create interactive 3D visualization of Gaussian components and data points.
    
    Parameters:
    -----------
    analyzer : ModificationAnalyzer
        Fitted analyzer object
    samples_dict : dict
        Dictionary of sample DataFrames
    sample_name : str
        Name of sample to visualize
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Get sample data
    sample_data = samples_dict[sample_name]
    X = analyzer.prepare_features(
        sample_data['signal'].values,
        sample_data['dwell_time'].values,
        sample_data['std_dev'].values
    )
    X_scaled = analyzer.scaler.transform(X)
    
    # Create figure
    fig = go.Figure()
    
    # Plot data points
    cluster_assignments = analyzer.gmm.predict(X_scaled)
    for i in range(analyzer.gmm.n_components):
        mask = cluster_assignments == i
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0],
            y=X[mask, 1],
            z=X[mask, 2],
            mode='markers',
            marker=dict(size=2),
            name=f'Cluster {i} points'
        ))
    
    # Plot Gaussian surfaces
    for i in range(analyzer.gmm.n_components):
        # Transform means and covariances back to original space
        mean_scaled = analyzer.gmm.means_[i]
        cov_scaled = analyzer.gmm.covariances_[i]
        
        mean = analyzer.scaler.inverse_transform([mean_scaled])[0]
        # Transform covariance matrix back to original scale
        std_scaled = np.sqrt(analyzer.scaler.var_)
        cov = np.outer(std_scaled, std_scaled) * cov_scaled
        
        xs, ys, zs, values = create_gaussian_surface_points(mean, cov)
        
        # Create isosurfaces at different probability levels
        for level in [0.1, 0.5, 0.9]:
            fig.add_trace(go.Isosurface(
                x=xs.flatten(),
                y=ys.flatten(),
                z=zs.flatten(),
                value=values.flatten(),
                isomin=values.max() * level,
                isomax=values.max() * level,
                opacity=0.3,
                name=f'Cluster {i} ({level:.1f})',
                showscale=False
            ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Signal',
            yaxis_title='Log Dwell Time',
            zaxis_title='Std Dev'
        ),
        title=f'3D Gaussian Mixture Components - {sample_name}',
        showlegend=True
    )
    
    return fig



def create_2d_gaussian_contours(mean, cov, n_points=100):
    """
    Create points for visualizing a 2D Gaussian contour.
    
    Parameters:
    -----------
    mean : array-like, shape (2,)
        Mean of the 2D Gaussian
    cov : array-like, shape (2, 2)
        Covariance matrix subset
    n_points : int
        Number of points per dimension
        
    Returns:
    --------
    tuple of (x, y, z)
        Grid coordinates and probability density values
    """
    std_dev = np.sqrt(np.diag(cov))
    x = np.linspace(mean[0] - 3*std_dev[0], mean[0] + 3*std_dev[0], n_points)
    y = np.linspace(mean[1] - 3*std_dev[1], mean[1] + 3*std_dev[1], n_points)
    
    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((xx, yy))
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)
    
    return xx, yy, z

def plot_2d_projections(analyzer, samples_dict, sample_name):
    """
    Create 2D projections of the Gaussian components for each pair of dimensions.
    
    Parameters:
    -----------
    analyzer : ModificationAnalyzer
        Fitted analyzer object
    samples_dict : dict
        Dictionary of sample DataFrames
    sample_name : str
        Name of sample to visualize
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Get sample data
    sample_data = samples_dict[sample_name]
    X = analyzer.prepare_features(
        sample_data['signal'].values,
        sample_data['dwell_time'].values,
        sample_data['std_dev'].values
    )
    X_scaled = analyzer.scaler.transform(X)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'Signal vs Log Dwell Time',
            'Signal vs Std Dev',
            'Log Dwell Time vs Std Dev'
        ]
    )
    
    # Define dimension pairs for plotting
    dim_pairs = [(0,1), (0,2), (1,2)]
    dim_names = ['Signal', 'Log Dwell Time', 'Std Dev']
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # for different clusters
    
    # Plot each projection
    cluster_assignments = analyzer.gmm.predict(X_scaled)
    
    for col, (dim1, dim2) in enumerate(dim_pairs, 1):
        # Plot points
        for i in range(analyzer.gmm.n_components):
            mask = cluster_assignments == i
            fig.add_trace(
                go.Scatter(
                    x=X[mask, dim1],
                    y=X[mask, dim2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors[i % len(colors)],
                        opacity=0.5
                    ),
                    name=f'Cluster {i}',
                    showlegend=(col == 1)  # only show legend once
                ),
                row=1, col=col
            )
        
        # Plot Gaussian contours
        for i in range(analyzer.gmm.n_components):
            # Transform mean and covariance back to original space
            mean_scaled = analyzer.gmm.means_[i]
            cov_scaled = analyzer.gmm.covariances_[i]
            
            mean = analyzer.scaler.inverse_transform([mean_scaled])[0]
            std_scaled = np.sqrt(analyzer.scaler.var_)
            cov = np.outer(std_scaled, std_scaled) * cov_scaled
            
            # Extract 2D components
            mean_2d = mean[[dim1, dim2]]
            cov_2d = cov[np.ix_([dim1, dim2], [dim1, dim2])]
            
            # Create contour points
            xx, yy, z = create_2d_gaussian_contours(mean_2d, cov_2d)
            
            # Add contours
            fig.add_trace(
                go.Contour(
                    x=xx[0,:],
                    y=yy[:,0],
                    z=z,
                    contours=dict(
                        start=0,
                        end=z.max(),
                        size=z.max()/4
                    ),
                    colorscale=[[0, f'rgba(0,0,0,0)'],
                               [1, colors[i % len(colors)]]],
                    showscale=False,
                    name=f'Cluster {i} contour',
                    showlegend=False
                ),
                row=1, col=col
            )
    
    # Update layout
    fig.update_layout(
        title=f'2D Projections of Gaussian Mixtures - {sample_name}',
        height=400,
        width=1200,
        showlegend=True
    )
    
    # Update axes labels
    for col, (dim1, dim2) in enumerate(dim_pairs, 1):
        fig.update_xaxes(title_text=dim_names[dim1], row=1, col=col)
        fig.update_yaxes(title_text=dim_names[dim2], row=1, col=col)
    
    return fig

def plot_cluster_proportions(results):
    """
    Create bar plot comparing cluster proportions across samples.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_modifications
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    samples = list(results.keys())
    n_clusters = len(results[samples[0]]['cluster_proportions'])
    
    for i in range(n_clusters):
        proportions = [results[sample]['cluster_proportions'][i] for sample in samples]
        fig.add_trace(go.Bar(
            name=f'Cluster {i}',
            x=samples,
            y=proportions
        ))
    
    fig.update_layout(
        barmode='group',
        title='Cluster Proportions Across Samples',
        xaxis_title='Sample',
        yaxis_title='Proportion',
        yaxis_range=[0, 1]
    )
    
    return fig

# Example usage:
"""
# After running analysis:
analyzer = ModificationAnalyzer(n_components=3)
analyzer.fit_combined_data(samples)
results = analyzer.analyze_samples(samples, control_name='control')

# Create 3D visualization for a specific sample
fig_3d = plot_3d_gaussians(analyzer, samples, 'mod_50pct')
fig_3d.show()

# Create cluster proportion comparison
fig_proportions = plot_cluster_proportions(results)
fig_proportions.show()
"""