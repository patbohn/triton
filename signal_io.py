"""Util library for sample metrics with control type support."""

from enum import Enum
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

class ControlType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NONE = "none"

class Sample:
    def __init__(self, name: str, metrics: Dict[str, np.ndarray], control: str):
        self.name = name
        self.metrics = metrics
        self.control = ControlType(control)

    @property
    def is_primary_control(self) -> bool:
        return self.control == ControlType.PRIMARY
        
    @property
    def is_secondary_control(self) -> bool:
        return self.control == ControlType.SECONDARY

    def get_metrics_at(self, idx: int) -> Dict[str, np.ndarray]:
        return {name: metric[idx] for name, metric in self.metrics.items() if name != "Position"}

class SampleMetrics:
    def __init__(self, samples: List[Sample] = None):
        self.samples = samples or []
        self._validate_controls()
        self._organize_samples()
        
    def _validate_controls(self):
        primary_controls = sum(1 for s in self.samples if s.is_primary_control)
        if primary_controls != 1:
            raise ValueError(f"Expected exactly one primary control, found {primary_controls}")
            
        secondary_controls = sum(1 for s in self.samples if s.is_secondary_control)
        if secondary_controls > 1:
            raise ValueError(f"Expected at most one secondary control, found {secondary_controls}")

    def _organize_samples(self):
        self.primary_control = [(s, s.metrics) for s in self.samples if s.is_primary_control]
        self.modified_samples = [(s, s.metrics) for s in self.samples if not s.is_primary_control and not s.is_secondary_control]
        self.secondary_control = [(s, s.metrics) for s in self.samples if s.is_secondary_control]
    
    def iter_positions(self) -> Iterator[Tuple[int, Dict]]:
        """Iterate through positions, returning structured data for GMM fitting"""
        if not self.samples:
            return
            
        first_sample = self.samples[0]
        positions = first_sample.metrics["Position"]
        n_positions = len(positions)
        
        for idx in range(n_positions):
            ref_pos = positions[idx]
            yield ref_pos, {
                "primary_control": [(s, s.get_metrics_at(idx)) for s, _ in self.primary_control],
                "modified_samples": [(s, s.get_metrics_at(idx)) for s, _ in self.modified_samples],
                "secondary_control": [(s, s.get_metrics_at(idx)) for s, _ in self.secondary_control]
            }
    
    @classmethod
    def load(cls, file_path: Path) -> 'SampleMetrics':
        samples = []
        with h5py.File(file_path, 'r') as f:
            sample_names = f.attrs['sample_names']
            control_types = f.attrs['control_types']
            
            for idx, name in enumerate(sample_names):
                metrics = {}
                group = f[name]
                
                for metric_name in group.keys():
                    metrics[metric_name] = group[metric_name][:]
                
                samples.append(Sample(
                    name=name,
                    metrics=metrics,
                    control=control_types[idx]
                ))
        
        return cls(samples)