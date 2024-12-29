import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_unmodified_fit(gmm, X, bins=50, figsize=(10, 6)):
    """Plot unmodified sample fit"""
    plt.figure(figsize=figsize)

    if X is not None:
        plt.hist(X, bins=bins, density=True, alpha=0.5, color="gray", label="Data")

    x = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

    # Plot components
    for i in range(gmm.n_components):
        pdf = (
            stats.norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0]))
            * gmm.weights_[i]
        )
        plt.plot(x, pdf, "--b", label=f"Component {i+1}")

    # Total distribution
    logprob = gmm.score_samples(x)
    total_pdf = np.exp(logprob)
    plt.plot(x, total_pdf, "-k", label="Total", linewidth=2)

    plt.legend()
    plt.grid(True)
    plt.title("Unmodified Sample Fit")
    plt.xlabel("Signal")
    plt.ylabel("Density")

    return plt.gca()


def plot_modified_fit(model, X=None, sample_idx=0, bins=50, figsize=(10, 6)):
    """Plot modified sample fit with component highlighting"""
    plt.figure(figsize=figsize)

    if X is not None:
        plt.hist(X, bins=bins, density=True, alpha=0.5, color="gray", label="Data")

    x = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

    # Plot unmodified components
    for i in range(model.unmodified_gmm.n_components):
        pdf = (
            stats.norm.pdf(
                x,
                model.modified_gmm.means_[i, 0],
                np.sqrt(model.modified_gmm.covariances_[i, 0, 0]),
            )
            * model.modified_gmm.weights_[sample_idx, i]
        )
        plt.plot(x, pdf, "--b", label=f"Unmodified {i+1}")

    # Plot modification components
    for i in range(model.unmodified_gmm.n_components, model.modified_gmm.n_components):
        pdf = (
            stats.norm.pdf(
                x,
                model.modified_gmm.means_[i, 0],
                np.sqrt(model.modified_gmm.covariances_[i, 0, 0]),
            )
            * model.modified_gmm.weights_[sample_idx, i]
        )
        plt.plot(
            x, pdf, "--r", label=f"Modified {i-model.unmodified_gmm.n_components+1}"
        )

    # Total distribution
    total_pdf = np.zeros_like(x.flatten())
    for i in range(model.modified_gmm.n_components):
        total_pdf += (
            stats.norm.pdf(
                x,
                model.modified_gmm.means_[i, 0],
                np.sqrt(model.modified_gmm.covariances_[i, 0, 0]),
            ).flatten()
            * model.modified_gmm.weights_[sample_idx, i]
        )
    plt.plot(x, total_pdf, "-k", label="Total", linewidth=2)

    plt.legend()
    plt.grid(True)
    plt.title(f"Modified Sample Fit (Sample {model.sample_ids[sample_idx]})")
    plt.xlabel("Signal")
    plt.ylabel("Density")

    return plt.gca()


def plot_modification_rates(rates, ci=None, figsize=(10, 6)):
    """
    Plot modification rates with optional confidence intervals

    Parameters:
    -----------
    rates : dict
        Dictionary of modification rates by sample ID
    ci : dict, optional
        Dictionary of (lower, upper) confidence intervals by sample ID
    """
    plt.figure(figsize=figsize)

    x = np.arange(len(rates))
    plt.bar(x, list(rates.values()), alpha=0.6)

    if ci is not None:
        lower = [ci[sample_id][0] for sample_id in rates.keys()]
        upper = [ci[sample_id][1] for sample_id in rates.keys()]
        plt.errorbar(
            x,
            list(rates.values()),
            yerr=[
                np.array(list(rates.values())) - lower,
                upper - np.array(list(rates.values())),
            ],
            fmt="none",
            color="k",
            capsize=5,
        )

    plt.xticks(x, list(rates.keys()), rotation=45)
    plt.grid(True)
    plt.title("Modification Rates by Sample")
    plt.xlabel("Sample")
    plt.ylabel("Modification Rate")

    return plt.gca()


def plot_bootstrap_distribution(
    bootstrap_rates, sample_id, true_rate=None, bins=30, figsize=(10, 6)
):
    """Plot distribution of bootstrap estimates"""
    plt.figure(figsize=figsize)

    plt.hist(bootstrap_rates, bins=bins, density=True, alpha=0.6)
    if true_rate is not None:
        plt.axvline(true_rate, color="r", linestyle="--", label="Point Estimate")

    plt.title(f"Bootstrap Distribution ({sample_id})")
    plt.xlabel("Modification Rate")
    plt.ylabel("Density")
    plt.grid(True)
    if true_rate is not None:
        plt.legend()

    return plt.gca()
