import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Tuple, Any, Union
from scipy.stats import beta
from scipy.optimize import minimize


def ttest(signals1, signals2) -> Dict[str, float]:
    from scipy.stats import ttest_ind

    result = ttest_ind(signals1, signals2, nan_policy="omit")
    return {
        "p_value": result.pvalue,
        "statistic": result.statistic,
    }


def kstest(signals1, signals2) -> Dict[str, float]:
    from scipy.stats import kstest

    result = kstest(signals1, signals2, nan_policy="omit")
    return {
        "p_value": result.pvalue,
        "statistic": result.statistic,
    }


class ModificationAnalyzer:
    def __init__(self, n_components=3):
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=42
        )

    def prepare_features(
        self, signal: np.ndarray, dwell_time: np.ndarray, std_dev: np.ndarray
    ) -> np.ndarray:
        """Prepare and normalize 3D feature space."""
        features = np.vstack([signal, np.log(dwell_time), std_dev]).T
        return features

    def fit_control_data(
        self, control_data: pd.DataFrame, n_samples: int = None
    ) -> None:
        """
        Fit GMM only on control data to model unmodified state.

        Parameters:
        -----------
        control_data : pd.DataFrame
            DataFrame containing control sample data
        n_samples : int, optional
            Number of samples to use for fitting. If None, use all data.
        """
        # Prepare features from control data
        features = self.prepare_features(
            control_data["signal"].values,
            control_data["dwell_time"].values,
            control_data["std_dev"].values,
        )

        # Sample if requested
        if n_samples is not None and n_samples < len(features):
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]

        # Fit scaler and GMM on control data only
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        self.gmm.fit(features_scaled)

        # Store control likelihood statistics for later comparison
        self.control_log_likelihoods = self.gmm.score_samples(features_scaled)
        self.sorted_control_log_ll = np.sort(self.control_log_likelihoods)

    def _fit_beta_mixture(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Fit a two-component beta mixture model with one component fixed to uniform.

        Parameters:
        -----------
        x : np.ndarray
            Array of p-values to fit

        Returns:
        --------
        Dict containing fitted parameters and mixture weights
        """
        # Transform p-values to avoid 0/1
        eps = 1e-5
        x = x.clip(eps, 1 - eps)

        def beta_mixture_nll(params):
            # Unpack parameters: [a2, b2, w]
            # First component is fixed to uniform Beta(1,1)
            a2, b2, w = params
            # Mixture likelihood
            l1 = 1.0  # uniform density is 1 everywhere on [0,1]
            l2 = beta.pdf(x, a2, b2)
            return -np.sum(np.log(w * l1 + (1 - w) * l2))

        # Initial guess - modified component near 0
        initial = [1, 10, 0.5]  # (a2, b2, weight of uniform component)
        bounds = [(0.1, 50), (0.1, 50), (0, 1)]

        # Fit mixture model
        result = minimize(beta_mixture_nll, initial, bounds=bounds)
        a2, b2, w = result.x

        return {
            "rate": 1 - w,  # w is weight of uniform (unmodified) component
            "modified_params": (a2, b2),
            "unmodified_params": (1, 1),  # fixed uniform
            "modified_mean": a2 / (a2 + b2),
            "unmodified_mean": 0.5,  # mean of uniform
            "convergence": result.success,
            "nll": result.fun,
        }

    def get_modification_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate per-read modification probabilities.

        Parameters:
        -----------
        features : np.ndarray
            Feature array from prepare_features()

        Returns:
        --------
        np.ndarray
            Array of modification probabilities for each read
        """
        # Scale features using control-fitted scaler
        features_scaled = self.scaler.transform(features)

        # Get posterior probabilities from GMM
        # The control GMM represents unmodified state, so modification probability
        # is 1 minus the probability of being in that state
        log_likelihoods = self.gmm.score_samples(features_scaled)

        p_values = np.searchsorted(self.sorted_control_log_ll, log_likelihoods) / len(
            self.control_log_likelihoods
        )

        return p_values

    def estimate_modification_rate(
        self, p_values: np.ndarray, method: str = "expectation", threshold: float = 0.95
    ) -> Union[float, Dict[str, Any]]:
        """
        Estimate overall modification rate from per-read probabilities.

        Parameters:
        -----------
        p_values : np.ndarray
            Array of per-read p_values
        method : str
            Method to calculate overall rate:
            - 'expectation': Use expected value (mean of probabilities)
            - 'threshold': Count reads above probability threshold
            - 'mixture': Fit mixture models (beta)
        threshold : float
            Probability threshold for 'threshold' method

        Returns:
        --------
        Union[float, Dict[str, Any]]
            Estimated overall modification rate or dictionary with mixture model results
        """
        if method == "expectation":
            # If all reads unmodified, mean would be 0.5
            # If all reads modified, mean would approach 0
            # Scale the deviation to estimate modification rate
            mean_pval = np.mean(p_values)
            return 2 * (0.5 - mean_pval)  # scales to [0,1]

        elif method == "threshold":
            return np.mean(p_values < threshold)

        elif method == "mixture":
            # Fit both mixture models
            return self._fit_beta_mixture(p_values)

        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_samples(
        self,
        sample_dict: Dict[str, pd.DataFrame],
        control_map: Dict[str, str],
        threshold: float = 0.05,
    ) -> Dict[str, Dict]:
        """
        Analyze modification probabilities for samples relative to their controls.

        Parameters:
        -----------
        sample_dict : Dict[str, pd.DataFrame]
            Dictionary with sample names as keys and DataFrames as values
        control_map : Dict[str, str]
            Dictionary mapping sample names to their respective control names
        threshold : float
            Probability threshold for threshold-based rate estimation

        Returns:
        --------
        Dict with sample names as keys and dictionaries containing:
            - mod_probabilities: per-read modification probabilities
            - mod_rates: dictionary of rates from different methods
            - num_observations: number of reads
            - control_name: name of control sample
        """
        results = {}

        for sample_name, control_name in control_map.items():
            if sample_name not in sample_dict or control_name not in sample_dict:
                continue

            # Fit model on control data
            control_data = sample_dict[control_name]
            self.fit_control_data(control_data)

            # Analyze treatment sample
            sample_data = sample_dict[sample_name]
            X_sample = self.prepare_features(
                sample_data["signal"].values,
                sample_data["dwell_time"].values,
                sample_data["std_dev"].values,
            )

            # perform simple tests on whole distributions
            t_test_results = ttest(
                control_data["signal"].values, sample_data["signal"].values
            )
            ks_test_results = kstest(
                control_data["signal"].values, sample_data["signal"].values
            )
            group_statistics = {
                "t_test": t_test_results,
                "ks_test": ks_test_results,
            }
            # Get per-read modification probabilities
            p_values = self.get_modification_probabilities(X_sample)
            mixture_results = self.estimate_modification_rate(p_values, "mixture")

            # Calculate modification rates using different methods
            mod_rates = {
                "expectation": self.estimate_modification_rate(p_values, "expectation"),
                "threshold": self.estimate_modification_rate(
                    p_values, "threshold", threshold
                ),
                "beta_mixture": mixture_results["rate"],
                "mixture_params": mixture_results,  # Store full mixture model parameters
            }

            results[sample_name] = {
                "per_read_p_values": p_values,
                "mod_rates": mod_rates,
                "group_statistics": group_statistics,
                "num_observations": len(sample_data),
                "control_name": control_name,
            }

        return results


def analyze_modifications(
    samples: Dict[str, pd.DataFrame],
    control_map: Dict[str, str],
    n_components: int = 3,
    method: str = "likelihood",
) -> Dict[str, Dict]:
    """
    Analyze modification probabilities across multiple samples.

    Parameters:
    -----------
    samples : Dict[str, pd.DataFrame]
        Dictionary with sample names as keys and DataFrames as values.
        Each DataFrame should have columns ['signal', 'dwell_time', 'std_dev']
    control_map : Dict[str, str]
        Dictionary mapping sample names to their respective control names
    n_components : int
        Number of Gaussian components to fit
    method : str
        Method to calculate modification probabilities

    Returns:
    --------
    Dict[str, Dict]
        Dictionary with analysis results for each sample
    """
    analyzer = ModificationAnalyzer(n_components=n_components)

    # Analyze each sample relative to its specific control
    results = analyzer.analyze_samples(samples, control_map, method=method)

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
    method='likelihood'  # or 'threshold'
)

# Access results for a specific sample:
mod_probs = results['glyoxal_5mM']['mod_probabilities']
mean_mod = results['glyoxal_5mM']['mean_mod_prob']
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
    x = np.linspace(mean[0] - 2 * std_dev[0], mean[0] + 2 * std_dev[0], n_points)
    y = np.linspace(mean[1] - 2 * std_dev[1], mean[1] + 2 * std_dev[1], n_points)
    z = np.linspace(mean[2] - 2 * std_dev[2], mean[2] + 2 * std_dev[2], n_points)

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
        sample_data["signal"].values,
        sample_data["dwell_time"].values,
        sample_data["std_dev"].values,
    )
    X_scaled = analyzer.scaler.transform(X)

    # Create figure
    fig = go.Figure()

    # Plot data points
    cluster_assignments = analyzer.gmm.predict(X_scaled)
    for i in range(analyzer.gmm.n_components):
        mask = cluster_assignments == i
        fig.add_trace(
            go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2],
                mode="markers",
                marker=dict(size=2),
                name=f"Cluster {i} points",
            )
        )

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
            fig.add_trace(
                go.Isosurface(
                    x=xs.flatten(),
                    y=ys.flatten(),
                    z=zs.flatten(),
                    value=values.flatten(),
                    isomin=values.max() * level,
                    isomax=values.max() * level,
                    opacity=0.3,
                    name=f"Cluster {i} ({level:.1f})",
                    showscale=False,
                )
            )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Signal", yaxis_title="Log Dwell Time", zaxis_title="Std Dev"
        ),
        title=f"3D Gaussian Mixture Components - {sample_name}",
        showlegend=True,
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
    x = np.linspace(mean[0] - 3 * std_dev[0], mean[0] + 3 * std_dev[0], n_points)
    y = np.linspace(mean[1] - 3 * std_dev[1], mean[1] + 3 * std_dev[1], n_points)

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
        sample_data["signal"].values,
        sample_data["dwell_time"].values,
        sample_data["std_dev"].values,
    )
    X_scaled = analyzer.scaler.transform(X)

    # Create subplot figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Signal vs Log Dwell Time",
            "Signal vs Std Dev",
            "Log Dwell Time vs Std Dev",
        ],
    )

    # Define dimension pairs for plotting
    dim_pairs = [(0, 1), (0, 2), (1, 2)]
    dim_names = ["Signal", "Log Dwell Time", "Std Dev"]
    colors = ["blue", "red", "green", "purple", "orange"]  # for different clusters

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
                    mode="markers",
                    marker=dict(size=3, color=colors[i % len(colors)], opacity=0.5),
                    name=f"Cluster {i}",
                    showlegend=(col == 1),  # only show legend once
                ),
                row=1,
                col=col,
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
                    x=xx[0, :],
                    y=yy[:, 0],
                    z=z,
                    contours=dict(start=0, end=z.max(), size=z.max() / 4),
                    colorscale=[[0, f"rgba(0,0,0,0)"], [1, colors[i % len(colors)]]],
                    showscale=False,
                    name=f"Cluster {i} contour",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title=f"2D Projections of Gaussian Mixtures - {sample_name}",
        height=400,
        width=1200,
        showlegend=True,
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
    n_clusters = len(results[samples[0]]["cluster_proportions"])

    for i in range(n_clusters):
        proportions = [results[sample]["cluster_proportions"][i] for sample in samples]
        fig.add_trace(go.Bar(name=f"Cluster {i}", x=samples, y=proportions))

    fig.update_layout(
        barmode="group",
        title="Cluster Proportions Across Samples",
        xaxis_title="Sample",
        yaxis_title="Proportion",
        yaxis_range=[0, 1],
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
