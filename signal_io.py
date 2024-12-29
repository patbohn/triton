"""Util library for sample metrics with control type support."""

import h5py
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List


class Sample:
    def __init__(
        self, name: str, metrics: Dict[str, np.ndarray], config: dict, ref: dict
    ):
        self.name = name
        self.metrics = metrics
        self.config = config
        self.ref = ref

    def __repr__(self):
        return f"""Sample Name: {self.name}
Metrics: {list(self.metrics.keys())} (num reads: {len(next(iter(self.metrics.values())))}, length: {len(next(iter(self.metrics.values()))[0])})
Reference: {self.ref}

Signal Refinement Config:
{self.config['signal_refinement']}
"""


def save_metrics(samples: List[Sample], output_path: Path) -> None:
    """
    Store sample metrics and their configs to an HDF5 file.

    Args:
        samples: List of Sample objects containing metrics and configs
        output_path: Path to save the HDF5 file
    """
    if not samples:
        raise ValueError("No samples provided")

    with h5py.File(output_path, "w") as f:
        # Store metadata
        f.attrs["sample_names"] = [sample.name for sample in samples]

        # Store samples
        for sample in samples:
            group = f.create_group(sample.name)

            # Store metrics
            metrics_group = group.create_group("metrics")
            for metric_name, metric_data in sample.metrics.items():
                metrics_group.create_dataset(
                    metric_name,
                    data=metric_data,
                    compression="gzip",
                    compression_opts=9,
                )

            # Store config as YAML string
            config_str = yaml.dump(sample.config)
            group.attrs["config"] = config_str
            ref_str = yaml.dump(sample.ref)
            group.attrs["ref"] = ref_str


def load(file_path: Path) -> List[Sample]:
    """Load sample metrics from an HDF5 file."""
    samples = []
    with h5py.File(file_path, "r") as f:
        sample_names = f.attrs["sample_names"]

        for name in sample_names:
            # Load metrics
            metrics = {}
            metrics_group = f[name]["metrics"]
            for metric_name in metrics_group.keys():
                metrics[metric_name] = metrics_group[metric_name][:]

            # Load config
            config_str = f[name].attrs["config"]
            config = yaml.safe_load(config_str)
            ref_str = f[name].attrs["ref"]
            ref = yaml.safe_load(ref_str)

            samples.append(
                Sample(
                    name=name,
                    metrics=metrics,
                    config=config,
                    ref=ref,
                )
            )

    return samples
