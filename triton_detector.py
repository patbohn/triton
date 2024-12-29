import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Tuple, Any, Union
from scipy.stats import beta
from scipy.optimize import minimize


import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")


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
        features[~np.isfinite(features)] = np.nan
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
        # drop na values
        all_samples = len(control_data)
        control_data = control_data.dropna()
        dropped_samples = all_samples - len(control_data)
        if dropped_samples > 0:
            logging.debug(
                f"[fit_control_data] Dropped {dropped_samples} NA samples, {len(control_data)} remaining."
            )

        # Prepare features from control data
        features = self.prepare_features(
            control_data["signal"].values,
            control_data["dwell_time"].values,
            control_data["std_dev"].values,
        )

        # Downsample if requested
        if n_samples is not None and n_samples < len(features):
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]

        # Fit scaler and GMM on control data only
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        self.gmm.fit(features_scaled)
        if not self.gmm.converged_:
            logging.warning(f"[fit_control_data] Warning! GMM fit did not converge.")

        # Store control likelihood statistics for later comparison
        self.control_log_likelihoods = self.gmm.score_samples(features_scaled)
        self.sorted_control_log_ll = np.sort(self.control_log_likelihoods)

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

        # Find non-NaN indices
        valid_mask = ~np.isnan(features_scaled).any(axis=1)
        if sum(valid_mask) < len(features_scaled[0]):
            logging.debug(
                f"[get_modification_probabilities] Dropped {sum(~valid_mask)} samples before prediction."
            )

        log_likelihoods = self.gmm.score_samples(features_scaled[valid_mask])

        p_values = np.full(features_scaled.shape[0], np.nan)

        p_values[valid_mask] = np.searchsorted(
            self.sorted_control_log_ll, log_likelihoods
        ) / len(self.control_log_likelihoods)

        return p_values

    def estimate_modification_rate(
        self, p_values: np.ndarray, method: str = "expectation", threshold: float = 0.05
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
            mean_pval = np.nanmean(p_values)
            return 2 * (0.5 - mean_pval)  # scales to [0,1]

        elif method == "threshold":
            return np.mean(p_values[~np.isnan(p_values)] < threshold)

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
            logging.debug(
                f"[analyze_samples] Sample {sample_name} compared against {control_name}."
            )
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

            # Calculate modification rates using different methods
            mod_rates = {
                "expectation": self.estimate_modification_rate(p_values, "expectation"),
                "threshold": self.estimate_modification_rate(
                    p_values, "threshold", threshold
                ),
            }

            results[sample_name] = {
                "per_read_p_values": p_values,
                "mod_rates": mod_rates,
                "group_statistics": group_statistics,
                "num_observations": sum(~np.isnan(sample_data["signal"])),
                "control_name": control_name,
            }

        return results


def analyze_modifications(
    samples: Dict[str, pd.DataFrame],
    control_map: Dict[str, str],
    n_components: int = 3,
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
    method='likelihood'  # or 'threshold'
)

# Access results for a specific sample:
mod_probs = results['glyoxal_5mM']['mod_probabilities']
mean_mod = results['glyoxal_5mM']['mean_mod_prob']
"""
