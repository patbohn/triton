"""
This is to extract per read per position signal information as resquiggled by remora. 

For each read and each position, stores dwell time, trimsd and trimmean signal in a hdf5 object. 

Usage: 
python script.py \
    --pod5-dir /path/to/pod5/files \
    --ref-file /path/to/reference.fasta \
    --sample-bam-files /path/to/sample1.bam /path/to/sample2.bam \
    --sample-names sample1 sample2 \
    --config /path/to/config.toml \
    --signal-outfile /path/to/output.hdf5 \
    [--start-spacing START_SPACING] \
    [--end-spacing END_SPACING] \
    [--metrics METRICS]
"""

from remora import io, refine_signal_map
import numpy as np
from pathlib import Path
import pod5
from Bio import SeqIO
import argparse
import toml
import yaml
from typing import Optional, List, Dict

import signal_io
from signal_io import Sample


def get_ref_region(
    ref_file: Path,
    config: dict = None,
    start_spacing: int | None = None,
    end_spacing: int | None = None,
) -> dict[np.array]:
    ref = SeqIO.read(ref_file, "fasta")
    return io.RefRegion(
        ctg=ref.id,
        strand="+",
        start=(start_spacing or config["ref_spacing"]["start"]),
        end=len(ref) - (end_spacing or config["ref_spacing"]["end"]),
    )


def extract_signal(
    pod5_dir: Path,
    sample_info: list[dict],
    ref_reg,
    config: dict,
    metrics: str = "dwell_trimmean_trimsd",
) -> List[Dict[str, np.ndarray]]:
    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=config["paths"]["level_table"],
        do_rough_rescale=config["signal_refinement"]["do_rough_rescale"],
        scale_iters=config["signal_refinement"]["scale_iters"],
        do_fix_guage=config["signal_refinement"]["do_fix_guage"],
    )

    pod5_dr = pod5.DatasetReader(pod5_dir)

    bam_handlers = [
        (pod5_dr, io.ReadIndexedBam(Path(sample["bam_file"]))) for sample in sample_info
    ]

    samples_metrics, _all_bam_reads = io.get_ref_reg_samples_metrics(
        ref_reg,
        bam_handlers,
        metric=metrics,
        sig_map_refiner=sig_map_refiner,
        reverse_signal=True,
    )

    print(f"Metrics computed: {', '.join(samples_metrics[0].keys())}")
    return samples_metrics


def main(
    pod5_dir: Path,
    ref_file: Path,
    sample_bam_files: list[Path],
    sample_names: list[str],
    config_file: Path,
    output_path: Path,
    start_spacing: Optional[int] = None,
    end_spacing: Optional[int] = None,
    metrics: str = "dwell_trimmean_trimsd",
) -> int:
    """
    Main function that can be called either from command line or as a module.

    Returns:
        int: Return code (0 for success, 1 for error)
    """
    try:
        config = toml.load(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    if len(sample_names) != len(sample_bam_files):
        raise ValueError("Length of sample names and bam files does not match!")

    # Create sample info list
    sample_info = [
        {"name": name, "bam_file": str(bam_file)}
        for name, bam_file in zip(sample_names, sample_bam_files)
    ]

    ref_reg = get_ref_region(ref_file, config, start_spacing, end_spacing)

    # Extract raw metrics
    samples_metrics = extract_signal(
        pod5_dir=pod5_dir,
        sample_info=sample_info,
        ref_reg=ref_reg,
        config=config,
        metrics=metrics,
    )

    # Create Sample objects
    samples = [
        Sample(
            name=name,
            metrics=metrics,
            config=config,
            ref={
                "ctg": ref_reg.ctg,
                "start": ref_reg.coord_range[0],
                "end": ref_reg.coord_range[-1],
            },
        )
        for name, metrics in zip(sample_names, samples_metrics)
    ]

    print(f"Successfully processed {len(samples)} samples. Saving to {output_path}")
    signal_io.save_metrics(samples, output_path)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-read per-position signal information from remora resquiggled data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pod5-dir", type=Path, required=True, help="Location of pod5 dir/file"
    )
    parser.add_argument(
        "--ref-file", type=Path, required=True, help="Location of reference file"
    )
    parser.add_argument(
        "--sample-bam-files",
        type=Path,
        nargs="+",
        required=True,
        help="Location of bam files",
    )
    parser.add_argument(
        "--sample-names", type=str, nargs="+", required=True, help="Sample names"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config.toml",
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--signal-outfile",
        type=Path,
        default="signals.hdf5",
        help="Path to store resquiggled signal in",
    )
    parser.add_argument(
        "--start-spacing",
        type=int,
        default=0,
        help="Override start spacing from config",
    )
    parser.add_argument(
        "--end-spacing", type=int, default=0, help="Override end spacing from config"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="dwell_trimmean_trimsd",
        help="Metrics to compute (comma-separated)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    return_code = main(
        pod5_dir=args.pod5_dir,
        ref_file=args.ref_file,
        sample_bam_files=args.sample_bam_files,
        sample_names=args.sample_names,
        config_file=args.config,
        output_path=args.signal_outfile,
        start_spacing=args.start_spacing,
        end_spacing=args.end_spacing,
        metrics=args.metrics,
    )

    exit(return_code)
