import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import h5py
from tqdm import tqdm

from triton_detector import ModificationAnalyzer
from signal_io import Sample, load

import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze modifications in signal data using control samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        required=True,
        help="HDF5 files containing sample metrics",
    )
    parser.add_argument(
        "--control-map",
        type=Path,
        required=True,
        help="YAML file mapping samples to their controls",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        required=True,
        help="Output TSV file for statistics",
    )
    parser.add_argument(
        "--pvals-output",
        type=Path,
        required=True,
        help="Output HDF5 file for per-read p-values",
    )
    parser.add_argument(
        "--drop-short-dwell",
        action="store_true",
        help="Drop events that have a dwell time below 6 signals",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of Gaussian components to fit",
    )

    return parser.parse_args()


def load_samples(file_paths: List[Path]) -> Dict[str, Sample]:
    """Load all samples from multiple HDF5 files into a dictionary."""
    samples = {}
    for file_path in file_paths:
        file_samples = load(file_path)
        for sample in file_samples:
            samples[sample.name] = sample
    logging.info(f"[load_samples] Loaded {len(samples)} samples in total from files.")
    return samples


def clean_up_df(df: pd.DataFrame, drop_short_dwell: bool = False) -> pd.DataFrame:
    """Drop nan and optionally very short dwell times"""
    return df[df["dwell_time"] > 5] if drop_short_dwell else df


def get_position_data(
    sample: Sample, position: int, drop_short_dwell: bool
) -> pd.DataFrame:
    """Extract metrics for a specific position from a Sample."""
    df = pd.DataFrame(
        {
            "signal": sample.metrics["trimmean"][:, position],
            "dwell_time": sample.metrics["dwell"][:, position],
            "std_dev": sample.metrics["trimsd"][:, position],
        }
    )
    before_cleanup = len(df)
    df = clean_up_df(df, drop_short_dwell)
    if len(df) < before_cleanup:
        logging.info(
            f"[get_position_data] {sample} at {postition}: Filtered {before_cleanup - len(df)} samples."
        )
    return df


def analyze_modifications_at_position(
    samples: Dict[str, Sample],
    control_map: Dict[str, str],
    position: int,
    drop_short_dwell: bool,
    n_components: int = 3,
) -> Dict[str, Dict]:
    """Analyze modifications at a specific position."""
    # Convert samples to DataFrames for this position
    sample_dfs = {
        name: get_position_data(sample, position, drop_short_dwell)
        for name, sample in samples.items()
    }

    # Run analysis
    analyzer = ModificationAnalyzer(n_components=n_components)
    results = analyzer.analyze_samples(sample_dfs, control_map)

    return results


def analyze_all_positions(
    samples: Dict[str, Sample],
    control_map: Dict[str, str],
    drop_short_dwell: bool,
    n_components: int = 3,
) -> Dict[str, List[Dict]]:
    """Analyze modifications at all positions."""
    # Get number of positions from first sample
    first_sample = next(iter(samples.values()))
    n_positions = next(iter(first_sample.metrics.values())).shape[1]
    logging.info(f"[analyze_all_positions] Processing {n_positions} positions.")

    # Initialize results structure
    results = {sample_name: [] for sample_name in control_map.keys()}

    # Analyze each position
    for pos in tqdm(range(n_positions), total=n_positions):
        logging.debug(f"[analyze_all_positions] Index position {pos}.")
        pos_results = analyze_modifications_at_position(
            samples,
            control_map,
            pos,
            drop_short_dwell,
            n_components,
        )

        # Store results for each sample
        for sample_name, sample_results in pos_results.items():
            results[sample_name].append(sample_results)

    return results


def save_results(
    results: Dict[str, List[Dict]],
    stats_output: Path,
    pvals_output: Path,
    samples: Dict[str, Sample],  # Add samples to get reference positions
) -> None:
    """
    Save analysis results to two files:
    - TSV file with statistics per position
    - HDF5 file with per-read p-values
    """
    # Save statistics to TSV
    rows = []
    for sample_name, positions_results in results.items():
        control_name = positions_results[0]["control_name"]
        start_pos = samples[sample_name].ref["start"]  # Get reference start position

        for array_pos, pos_results in enumerate(positions_results):
            genome_pos = start_pos + array_pos  # Calculate actual genomic position
            row = {
                "sample": sample_name,
                "control": control_name,
                "array_position": array_pos,
                "genome_position": genome_pos,
                "num_reads": pos_results["num_observations"],
            }

            # Add modification rates
            for method, rate in pos_results["mod_rates"].items():
                row[f"mod_rate_{method}"] = rate

            # Add statistics from tests
            for test_name, test_results in pos_results["group_statistics"].items():
                for stat_name, stat_value in test_results.items():
                    row[f"{test_name}_{stat_name}"] = stat_value

            rows.append(row)

    # Convert to DataFrame and save stats
    df = pd.DataFrame(rows)
    cols = ["sample", "control", "array_position", "genome_position", "num_reads"]
    stat_cols = [col for col in df.columns if col not in cols]
    df = df[cols + sorted(stat_cols)]
    df.to_csv(stats_output, sep="\t", index=False)

    # Save per-read p-values to HDF5
    with h5py.File(pvals_output, "w") as f:
        for sample_name, positions_results in results.items():
            sample_group = f.create_group(sample_name)
            n_positions = len(positions_results)
            n_reads = len(positions_results[0]["per_read_p_values"])

            # Create dataset for p-values
            p_values_ds = sample_group.create_dataset(
                "per_read_p_values",
                shape=(n_positions, n_reads),
                dtype=np.float32,
                compression="gzip",
                compression_opts=9,
            )

            # Store p-values for each position
            for pos, pos_results in enumerate(positions_results):
                p_values_ds[pos] = pos_results["per_read_p_values"]

            # Store metadata
            sample_group.attrs["control_name"] = positions_results[0]["control_name"]


def main():
    args = parse_args()

    # Load control mapping
    with open(args.control_map) as f:
        control_map = yaml.safe_load(f)
    logging.info(f"[main] Found {len(control_map)} samples in control_map file.")

    # Load all samples
    logging.info(f"[main] Loading samples from input files: {args.input_files}.")
    samples = load_samples(args.input_files)

    # Run analysis for all positions
    logging.info(f"[main] Running mod detection analysis.")
    results = analyze_all_positions(
        samples,
        control_map=control_map,
        drop_short_dwell=args.drop_short_dwell,
        n_components=args.n_components,
    )

    # Save results to both files
    logging.info(f"[main] Storing results.")
    save_results(
        results,
        stats_output=args.stats_output,
        pvals_output=args.pvals_output,
        samples=samples,  # Pass samples to get reference positions
    )

    return 0


if __name__ == "__main__":
    exit(main())
