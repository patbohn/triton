import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.cluster import KMeans


class NestedMixtureModel:
    def __init__(self, n_unmodified_components=2, n_modified_components=2):
        """
        Initialize nested mixture model with separate mixtures for unmodified and modified distributions

        Parameters:
        -----------
        n_unmodified_components : int
            Number of components in the unmodified distribution
        n_modified_components : int
            Number of components in the modified distribution
        """
        self.n_unmodified_components = n_unmodified_components
        self.n_modified_components = n_modified_components

        # Parameters to be estimated
        self.unmodified_weights = None
        self.unmodified_means = None
        self.unmodified_stds = None
        self.modified_weights = None
        self.modified_means = None
        self.modified_stds = None
        self.modification_rate = None

    def _mixture_pdf(self, x, weights, means, stds):
        """
        Compute PDF of mixture distribution

        Parameters:
        -----------
        x : float or array-like
            Input data points
        weights : array-like
            Component weights
        means : array-like
            Component means
        stds : array-like
            Component standard deviations

        Returns:
        --------
        float or array-like
            Probability density at input points
        """
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Compute mixture PDF
        pdf = np.zeros_like(x, dtype=float)
        for w, mu, std in zip(weights, means, stds):
            pdf += w * norm.pdf(x, mu, std)
        return pdf

    def _log_likelihood(self, params, data):
        """
        Compute negative log-likelihood for optimization

        Parameters:
        -----------
        params : array-like
            Parameter vector to optimize
        data : array-like
            Observed data

        Returns:
        --------
        float
            Negative log-likelihood
        """
        n_u = self.n_unmodified_components
        n_m = self.n_modified_components

        # Unpack parameters
        # Unmodified distribution
        unmodified_weights = params[:n_u]
        unmodified_weights /= np.sum(unmodified_weights)

        unmodified_means = params[n_u : n_u * 2]
        unmodified_stds = np.abs(params[n_u * 2 : n_u * 3])

        # Modified distribution
        modified_weights = params[n_u * 3 : n_u * 3 + n_m]
        modified_weights /= np.sum(modified_weights)

        modified_means = params[n_u * 3 + n_m : n_u * 3 + n_m + n_m]
        modified_stds = np.abs(params[n_u * 3 + n_m + n_m : n_u * 3 + n_m + n_m * 2])

        # Modification rate (constrained between 0 and 1)
        modification_rate = np.clip(params[-1], 0, 1)

        # Compute log-likelihood
        log_likelihood = 0
        for x in data:
            # Unmodified component PDF
            unmodified_pdf = self._mixture_pdf(
                x, unmodified_weights, unmodified_means, unmodified_stds
            )

            # Modified component PDF
            modified_pdf = self._mixture_pdf(
                x, modified_weights, modified_means, modified_stds
            )

            # Likelihood: mixture of unmodified and modified
            # Prioritize unmodified distribution to prevent overestimation
            point_likelihood = (
                1 - modification_rate
            ) * unmodified_pdf + modification_rate * modified_pdf

            log_likelihood += np.log(max(point_likelihood, 1e-300))

        return -log_likelihood

    def fit(self, data, initial_guess=None):
        """
        Fit nested mixture model to data with improved optimization
        """
        # Flatten data if needed
        data = np.array(data).flatten()

        # Define parameter bounds and initial guess
        n_u = self.n_unmodified_components
        n_m = self.n_modified_components

        # Bounds for parameters
        bounds = (
            # Unmodified weights (sum to 1)
            [(0, 1)] * n_u
            +
            # Unmodified means (data range)
            [(np.min(data), np.max(data))] * n_u
            +
            # Unmodified stds (positive)
            [(0.01, np.std(data) * 2)] * n_u
            +
            # Modified weights (sum to 1)
            [(0, 1)] * n_m
            +
            # Modified means (data range)
            [(np.min(data), np.max(data))] * n_m
            +
            # Modified stds (positive)
            [(0.01, np.std(data) * 2)] * n_m
            +
            # Modification rate
            [(0, 0.5)]
        )

        # Initial guess based on data characteristics
        if initial_guess is None:
            # Use percentiles to spread initial means
            unmodified_means = np.percentile(data, np.linspace(25, 75, n_u))
            modified_means = np.percentile(data, np.linspace(25, 75, n_m))

            initial_guess = (
                # Unmodified weights
                [1 / n_u] * n_u
                +
                # Unmodified means
                list(unmodified_means)
                +
                # Unmodified stds
                [np.std(data) / n_u] * n_u
                +
                # Modified weights
                [1 / n_m] * n_m
                +
                # Modified means
                list(modified_means)
                +
                # Modified stds
                [np.std(data) * 1.5 / n_m] * n_m
                +
                # Modification rate
                [0.1]
            )

        # Optimize parameters
        result = minimize(
            self._log_likelihood,
            initial_guess,
            args=(data,),
            method="L-BFGS-B",  # More efficient for bounded optimization
            bounds=bounds,
            options={
                "maxiter": 200,  # Limit iterations
                "disp": False,  # Suppress output
            },
        )

        # Store and return optimized parameters
        # (same parameter extraction as before)
        self.unmodified_weights = result.x[:n_u] / np.sum(result.x[:n_u])
        self.unmodified_means = result.x[n_u : n_u * 2]
        self.unmodified_stds = np.abs(result.x[n_u * 2 : n_u * 3])

        self.modified_weights = result.x[n_u * 3 : n_u * 3 + n_m] / np.sum(
            result.x[n_u * 3 : n_u * 3 + n_m]
        )
        self.modified_means = result.x[n_u * 3 + n_m : n_u * 3 + n_m + n_m]
        self.modified_stds = np.abs(
            result.x[n_u * 3 + n_m + n_m : n_u * 3 + n_m + n_m * 2]
        )

        self.modification_rate = np.clip(result.x[-1], 0, 1)

        return {
            "unmodified_weights": self.unmodified_weights,
            "unmodified_means": self.unmodified_means,
            "unmodified_stds": self.unmodified_stds,
            "modified_weights": self.modified_weights,
            "modified_means": self.modified_means,
            "modified_stds": self.modified_stds,
            "modification_rate": self.modification_rate,
            "optimization_success": result.success,
            "optimization_message": result.message,
        }


# Example usage
def example_usage():
    np.random.seed(42)

    # Simulate unmodified data (two Gaussian components)
    unmodified_data1 = np.random.normal(0, 1, 500)
    unmodified_data2 = np.random.normal(2, 0.5, 500)
    unmodified_data = np.concatenate([unmodified_data1, unmodified_data2])

    # Simulate modified data (two Gaussian components)
    modified_data1 = np.random.normal(1, 1.5, 100)
    modified_data2 = np.random.normal(3, 1.0, 100)
    modified_data = np.concatenate([modified_data1, modified_data2])

    # Combined dataset
    full_data = np.concatenate([unmodified_data, modified_data])

    # Fit nested mixture model
    model = NestedMixtureModel(n_unmodified_components=2, n_modified_components=2)
    results = model.fit(full_data)

    print("Estimation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()
