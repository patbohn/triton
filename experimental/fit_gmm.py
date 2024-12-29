"""
This reads in resquiggled data (remora)

"""

from pathlib import Path
import argparse
from typing import Optional
import toml

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
from scipy.special import logsumexp
import pickle
import pandas as pd

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

from . import signal_io


class MultiSampleGMM:
    """
    A GMM implementation that fits multiple samples simultaneously,
    with shared components for modifications but independent weights.
    """

    def __init__(
        self,
        fixed_means,
        fixed_covs,
        fixed_weight_ratios,
        n_new_components,
        n_samples,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        fixed_means : array-like
            Means of the fixed (unmodified) components
        fixed_covs : array-like
            Covariances of the fixed components
        fixed_weight_ratios : array-like
            Relative ratios of the fixed components
        n_new_components : int
            Number of new components to fit (shared across samples)
        n_samples : int
            Number of modified samples to fit simultaneously
        """
        self.n_fixed = len(fixed_means)
        self.n_new = n_new_components
        self.n_components = self.n_fixed + self.n_new
        self.n_samples = n_samples

        # Fixed parameters
        self.fixed_means = np.array(fixed_means).reshape(-1, 1)
        self.fixed_covs = np.array(fixed_covs).reshape(-1, 1, 1)
        self.fixed_ratios = np.array(fixed_weight_ratios)

        # Initialize parameters
        self.means_ = np.zeros((self.n_components, 1))
        self.covariances_ = np.zeros((self.n_components, 1, 1))
        self.weights_ = np.zeros((n_samples, self.n_components))

        # New component parameters (shared across samples)
        self.new_means_ = None
        self.new_covs_ = None

        self.converged_ = False
        self.n_iter_ = 0
        self.max_iter = kwargs.get("max_iter", 100)
        self.tol = kwargs.get("tol", 1e-3)
        self.random_state = kwargs.get("random_state", None)

    def _initialize(self, X_list):
        """Initialize parameters"""
        rng = np.random.RandomState(self.random_state)

        # Set fixed components
        self.means_[: self.n_fixed] = self.fixed_means
        self.covariances_[: self.n_fixed] = self.fixed_covs

        # Initialize new components using pooled data
        for i in range(self.n_samples):
            self.weights_[i, : self.n_fixed] = self.fixed_ratios / np.sum(
                self.fixed_ratios
            )
            if self.n_new > 0:
                # Initialize new components using pooled data
                X_all = np.concatenate(X_list)
                idx = rng.choice(len(X_all), self.n_new)
                self.means_[self.n_fixed :] = X_all[idx].reshape(-1, 1)
                self.covariances_[self.n_fixed :] = np.var(X_all) * np.ones(
                    (self.n_new, 1, 1)
                )
                self.weights_[i, self.n_fixed :] = 0.5 / self.n_new

    def _e_step(self, X_list):
        log_resp_list = []

        for i, X in enumerate(X_list):
            log_resp = np.zeros((len(X), self.n_components))

            for k in range(self.n_components):
                log_resp[:, k] = (
                    np.log(np.maximum(self.weights_[i, k], 1e-300))  # Prevent log(0)
                    + stats.norm.logpdf(
                        X.flatten(),
                        self.means_[k, 0],
                        np.sqrt(self.covariances_[k, 0, 0]),
                    )
                )

            # Normalize using logsumexp for numerical stability
            log_resp_normalized = log_resp - logsumexp(log_resp, axis=1)[:, np.newaxis]
            log_resp_list.append(log_resp_normalized)

        return log_resp_list

    def _m_step(self, X_list, log_resp_list):
        for i, (X, log_resp) in enumerate(zip(X_list, log_resp_list)):
            resp = np.exp(log_resp)
            total_resp = resp.sum(axis=0)

            # Update weights - maintain fixed ratios only for unmodified components
            total_fixed_weight = total_resp[: self.n_fixed].sum() / len(X)
            self.weights_[i, : self.n_fixed] = (
                self.fixed_ratios / np.sum(self.fixed_ratios) * total_fixed_weight
            )
            if self.n_new > 0:
                self.weights_[i, self.n_fixed :] = total_resp[self.n_fixed :] / len(X)

        # Update means and covariances of new components only (shared across samples)
        for k in range(self.n_fixed, self.n_components):
            # Pool weighted observations from all samples
            weighted_sum = 0
            total_resp = 0

            for X, log_resp in zip(X_list, log_resp_list):
                resp = np.exp(log_resp[:, k])
                weighted_sum += (resp * X.flatten()).sum()
                total_resp += resp.sum()

            self.means_[k] = weighted_sum / total_resp

            # Update covariance
            weighted_sum_sq = 0
            for X, log_resp in zip(X_list, log_resp_list):
                resp = np.exp(log_resp[:, k])
                diff = X - self.means_[k]
                weighted_sum_sq += (resp * diff.flatten() ** 2).sum()

            self.covariances_[k] = weighted_sum_sq / total_resp

    def fit(self, X_list):
        """
        Fit the model to multiple samples

        Parameters:
        -----------
        X_list : list of array-like
            List of samples to fit, each sample is an array of observations
        """
        X_list = [np.array(X).reshape(-1, 1) for X in X_list]

        # Initialize
        self._initialize(X_list)

        # EM algorithm
        prev_log_likelihood = None

        for n_iter in range(self.max_iter):
            # E-step
            log_resp_list = self._e_step(X_list)

            # M-step
            self._m_step(X_list, log_resp_list)

            # Check convergence
            log_likelihood = 0
            for i, X in enumerate(X_list):
                for k in range(self.n_components):
                    log_likelihood += np.sum(
                        self.weights_[i, k]
                        * stats.norm.logpdf(
                            X, self.means_[k, 0], np.sqrt(self.covariances_[k, 0, 0])
                        )
                    )

            if prev_log_likelihood is not None:
                change = abs(log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

            prev_log_likelihood = log_likelihood
            self.n_iter_ = n_iter + 1

        self.prev_log_likelihood = prev_log_likelihood

        return self


class NanoporeGMM:
    """
    A class for analyzing Nanopore sequencing signals using Gaussian Mixture Models
    to estimate modification rates across multiple samples.
    """

    def __init__(self, max_unmodified_components=3, max_modified_components=2):
        self.max_unmodified_components = max_unmodified_components
        self.max_modified_components = max_modified_components
        self.unmodified_gmm = None
        self.modified_gmm = None

    def fit_unmodified(self, signals, n_init=10):
        """Fit GMM to unmodified sample signals"""
        X = np.array(signals).reshape(-1, 1)

        best_bic = np.inf
        for n in range(1, self.max_unmodified_components + 1):
            gmm = GaussianMixture(
                n_components=n, n_init=n_init, random_state=42, covariance_type="full"
            )
            gmm.fit(X.reshape(-1, 1))
            bic = gmm.bic(X.reshape(-1, 1))

            if bic < best_bic:
                best_bic = bic
                self.unmodified_gmm = gmm

        return self

    def fit_modified_samples(self, signal_list, sample_ids=None, max_iter: int = 100):
        """
        Fit multiple modified samples simultaneously

        Parameters:
        -----------
        signal_list : list of array-like
            List of signal arrays from modified samples
        sample_ids : list of str, optional
            Identifiers for each sample
        """
        if self.unmodified_gmm is None:
            raise ValueError("Must fit unmodified sample first")

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(signal_list))]

        # Try different numbers of new components (including 0)
        best_bic = np.inf
        best_gmm = None

        # Calculate baseline fit with no new components
        base_gmm = MultiSampleGMM(
            fixed_means=self.unmodified_gmm.means_.flatten(),
            fixed_covs=self.unmodified_gmm.covariances_,
            fixed_weight_ratios=self.unmodified_gmm.weights_,
            n_new_components=0,
            n_samples=len(signal_list),
            random_state=42,
        )
        base_gmm.fit(signal_list)
        base_ll = base_gmm.prev_log_likelihood

        for n_new in range(0, self.max_modified_components + 1):
            gmm = MultiSampleGMM(
                fixed_means=self.unmodified_gmm.means_.flatten(),
                fixed_covs=self.unmodified_gmm.covariances_,
                fixed_weight_ratios=self.unmodified_gmm.weights_,
                n_new_components=n_new,
                n_samples=len(signal_list),
                random_state=42,
                max_iter=max_iter,
            )

            gmm.fit(signal_list)

            # TODO: Calculate BIC for each sample given components
            X_all = np.concatenate([np.array(X).reshape(-1, 1) for X in signal_list])
            n_params = (
                n_new * 2  # means and vars for new components
                + len(signal_list) * (self.unmodified_gmm.n_components + n_new - 1)
            )  # weights per sample

            # Improvement over base model
            ll_improvement = gmm.prev_log_likelihood - base_ll
            if ll_improvement < 0:
                ll_improvement = 0

            bic = -2 * ll_improvement + n_params * np.log(len(X_all))

            logging.debug(f"n_new: {n_new}, BIC: {bic:.2f}")
            logging.debug(
                f"New component means: {gmm.means_[self.unmodified_gmm.n_components:]}"
            )
            logging.debug(
                f"New component weights: {gmm.weights_[:, self.unmodified_gmm.n_components:]}"
            )

            if bic < best_bic:
                logging.debug(
                    f"Model updated because current bic is below best: {bic=} {best_bic=}"
                )
                best_bic = bic
                best_gmm = gmm

        self.modified_gmm = best_gmm
        self.sample_ids = sample_ids
        return self

    def has_modifications(self):
        """
        Returns True if the model found evidence for modifications
        (i.e., if additional components improved the fit)
        """
        if self.modified_gmm is None:
            raise ValueError("Must fit modified samples first")
        return self.modified_gmm.n_components > self.unmodified_gmm.n_components

    def get_modification_rates(
        self,
        signal_list: list,
        with_ci=False,
        max_iter: int = 100,
        n_bootstrap=100,
        ci_level=0.95,
    ):
        """
        Get modification rates for all samples, optionally with confidence intervals.
        Returns 0 for all samples if no modification components were needed.

        Parameters:
        -----------
        with_ci : bool, default=False
            If True, compute confidence intervals using bootstrap resampling
        n_bootstrap : int, default=100
            Number of bootstrap resamples to use
        ci_level : float, default=0.95
            Confidence level for the intervals

        Returns:
        --------
        If with_ci=False:
            dict: {sample_id: modification_rate}
        If with_ci=True:
            dict: {sample_id: (modification_rate, lower_ci, upper_ci)}
        """
        if self.modified_gmm is None:
            raise ValueError("Must fit modified samples first")

        if not with_ci:
            rates = {}
            for i, sample_id in enumerate(self.sample_ids):
                if self.modified_gmm.n_components > self.unmodified_gmm.n_components:
                    rates[sample_id] = np.sum(
                        self.modified_gmm.weights_[
                            i, self.unmodified_gmm.n_components :
                        ]
                    )
                else:
                    rates[sample_id] = 0.0
            return rates

        if not self.has_modifications():
            return {sample_id: (0.0, 0.0, 0.0) for sample_id in self.sample_ids}

        # With confidence intervals
        rates = {}

        for i, sample_id in enumerate(self.sample_ids):
            # Bootstrap resampling
            bootstrap_rates = []
            X = signal_list[i]
            n_samples = len(X)

            for _ in range(n_bootstrap):
                idx = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[idx]

                gmm = MultiSampleGMM(
                    fixed_means=self.unmodified_gmm.means_.flatten(),
                    fixed_covs=self.unmodified_gmm.covariances_,
                    fixed_weight_ratios=self.unmodified_gmm.weights_,
                    n_new_components=self.modified_gmm.n_components
                    - self.unmodified_gmm.n_components,
                    n_samples=1,
                    max_iter=max_iter,
                )

                try:
                    # Let the bootstrap find its own means and covs
                    gmm.fit([X_boot])
                    mod_rate = np.sum(
                        gmm.weights_[0, self.unmodified_gmm.n_components :]
                    )
                    bootstrap_rates.append(mod_rate)
                except Exception:
                    continue  # Skip failed bootstraps

            # Calculate confidence intervals
            alpha = 1 - ci_level

            # Only compute CIs if we have enough valid bootstrap samples
            if len(bootstrap_rates) > n_bootstrap / 2:
                lower = np.percentile(bootstrap_rates, alpha / 2 * 100)
                upper = np.percentile(bootstrap_rates, (1 - alpha / 2) * 100)
            else:
                lower = upper = np.nan

            # Store point estimate and CIs
            rates[sample_id] = (
                np.sum(
                    self.modified_gmm.weights_[i, self.unmodified_gmm.n_components :]
                ),
                lower,
                upper,
            )

        return rates

    def save_model(self, filename):
        """Save the fitted model parameters to a file"""
        if self.unmodified_gmm is None:
            raise ValueError("Must fit unmodified sample first")

        model_data = {
            "unmodified": {
                "means": self.unmodified_gmm.means_.tolist(),
                "covariances": self.unmodified_gmm.covariances_.tolist(),
                "weights": self.unmodified_gmm.weights_.tolist(),
                "n_components": self.unmodified_gmm.n_components,
            },
            "modified": None
            if self.modified_gmm is None
            else {
                "means": self.modified_gmm.means_.tolist(),
                "covariances": self.modified_gmm.covariances_.tolist(),
                "weights": self.modified_gmm.weights_.tolist(),
                "n_components": self.modified_gmm.n_components,
                "sample_ids": self.sample_ids,
            },
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        """Load a saved model"""
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        model = cls()

        # Reconstruct unmodified GMM
        model.unmodified_gmm = GaussianMixture(
            n_components=model_data["unmodified"]["n_components"]
        )
        model.unmodified_gmm.means_ = np.array(model_data["unmodified"]["means"])
        model.unmodified_gmm.covariances_ = np.array(
            model_data["unmodified"]["covariances"]
        )
        model.unmodified_gmm.weights_ = np.array(model_data["unmodified"]["weights"])

        # Reconstruct modified GMM if it exists
        if model_data["modified"] is not None:
            model.sample_ids = model_data["modified"]["sample_ids"]
            model.modified_gmm = MultiSampleGMM(
                fixed_means=model.unmodified_gmm.means_.flatten(),
                fixed_covs=model.unmodified_gmm.covariances_,
                fixed_weight_ratios=model.unmodified_gmm.weights_,
                n_new_components=model_data["modified"]["n_components"]
                - model_data["unmodified"]["n_components"],
                n_samples=len(model_data["modified"]["weights"]),
            )
            model.modified_gmm.means_ = np.array(model_data["modified"]["means"])
            model.modified_gmm.covariances_ = np.array(
                model_data["modified"]["covariances"]
            )
            model.modified_gmm.weights_ = np.array(model_data["modified"]["weights"])

        return model


class FitGMMResult:
    def __init__(self, model, rates):
        self.model = model
        self.rates = rates


def fit_gmm(
    control_signals: np.ndarray,
    mod_signals: list[np.ndarray],
    control_name: str,
    mod_names: list[str] = None,
    max_unmodified_components: int = 3,
    max_modified_components: int = 2,
    max_iter: int = 100,
    n_bootstrap: int = 100,
) -> FitGMMResult:
    """
    Fit GMM to signals from a single position.

    Args:
        control_signals: Primary control signals at this position
        mod_signals: List of modified sample signals
        sample_names: Names for modified samples (optional)
        max_unmodified_components: Max components for control
        max_modified_components: Max additional components for modified samples

    Returns:
        FitGMMResult containing fitted model and modification rates
    """
    model = NanoporeGMM(
        max_unmodified_components=max_unmodified_components,
        max_modified_components=max_modified_components,
    )

    model.fit_unmodified(control_signals)
    model.fit_modified_samples(mod_signals, sample_ids=mod_names)

    rates_with_ci = model.get_modification_rates(
        signal_list=mod_signals,
        with_ci=True,
        n_bootstrap=100,
        ci_level=0.95,
        max_iter=max_iter,
    )

    return FitGMMResult(model, rates_with_ci)


def process_positions(
    metrics: signal_io.SampleMetrics,
    config: dict,
):
    positions_data = []
    metric = config.get("metric")
    if not metric:
        raise ValueError("Metric must be defined in the config file")

    for ref_pos, pos_data in metrics.iter_positions():
        control_signals = pos_data["primary_control"][0][1][metric]
        mod_signals = [m[metric] for _, m in pos_data["modified_samples"]]
        mod_names = [s.name for s, _ in pos_data["modified_samples"]]

        result = fit_gmm(
            control_signals=control_signals,
            mod_signals=mod_signals,
            sample_names=mod_names,
        )

        positions_data.append({"position": ref_pos, "rates": result.rates})

    return positions_data


def main(
    signal_file: Path,
    config_path: Path,
    output_path: Path,
):
    """Process signals and fit GMM model per reference position."""
    try:
        config = toml.load(config_path)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return 1

    try:
        metrics = signal_io.SampleMetrics.load(signal_file)
    except Exception as e:
        print(f"Could not load in data: {e}")
        return 1

    try:
        positions_data = process_positions(metrics, config)
    except Exception as e:
        print(f"Error processing signals: {e}")
        return 1

    # Save results
    df = pd.DataFrame(positions_data)
    df.to_csv(output_path, index=None)

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform GMM matching to estimate true modification rates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default="config.toml",
        help="Path to TOML configuration file",
    )

    parser.add_argument(
        "--signal-file",
        type=Path,
        default="signals.hdf5",
        help="Path to load resquiggled signal from",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default="output.csv",
        help="Path to store reactivity estimates in",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    return_code = main(
        signal_file=args.signal_file,
        config_path=args.config,
        output_path=args.output_path,
    )

    exit(return_code)
