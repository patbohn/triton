"""
This is to extract per read per position signal information as resquiggled by remora. 

For each read and each position, stores dwell time, trimsd and trimmean signal

Usage: 
python script.py \
    --pod5-dir /path/to/pod5/files \
    --control-bam /path/to/control.bam \
    --mod-bams /path/to/mod1.bam /path/to/mod2.bam \
    --ref-file /path/to/reference.fasta
"""

from remora import io, refine_signal_map
import numpy as np
from pathlib import Path
import pod5
from Bio import SeqIO
import argparse
import toml
import yaml
from typing import Optional

from . import signal_io


def get_ref_region(
    ref_file: Path,
    start_spacing: int | None = None,
    end_spacing: int | None = None,
    config: dict = None,
) -> dict[np.array]:
    ref = SeqIO.read(ref_file, "fasta")
    return io.RefRegion(
        ctg=ref.id, strand="+", 
        start=len(ref) + (start_spacing or config['ref_spacing']['start']), 
        end=len(ref) - (end_spacing or config['ref_spacing']['end']),
    )



def extract_signal(
    pod5_dir: Path, 
    sample_info: list[dict],
    ref_file: Path,
    config: dict,
    start_spacing: int | None = None,
    end_spacing: int | None = None, 
    metrics: str = "dwell_trimmean_trimsd",
) -> dict[np.array]:
    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=config['paths']['level_table'],
        do_rough_rescale=config['signal_refinement']['do_rough_rescale'],
        scale_iters=config['signal_refinement']['scale_iters'],
        do_fix_guage=config['signal_refinement']['do_fix_guage'],
    )

    ref_reg = get_ref_region(ref_file, start_spacing, end_spacing)
    pod5_dr = pod5.DatasetReader(pod5_dir)
    
    bam_handlers = [
        (pod5_dr, io.ReadIndexedBam(Path(sample['bam_file'])))
        for sample in sample_info
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
    sample_info: Path,
    config_file: Path,
    output_path: Path,
    start_spacing: Optional[int] = None,
    end_spacing: Optional[int] = None,
    metrics: str = "dwell_trimmean_trimsd"
) -> int:
    """
    Main function that can be called either from command line or as a module.
    
    Returns:
        tuple: (samples_metrics, return_code)
    """
    try:
        with open(sample_info) as f:
            sample_info = yaml.safe_load(f)
        config = toml.load(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    try:
        samples_metrics = extract_signal(
            pod5_dir=sample_info['pod5_dir'],
            sample_info=sample_info['samples'],
            ref_file=Path(sample_info['ref_file']),
            config=config,
            start_spacing=start_spacing,
            end_spacing=end_spacing,
            metrics=metrics,
        )
        
        sample_names = [s['name'] for s in sample_info['samples']]
        control_flags = [
            s['control'] == 'primary' or s['control'] == 'secondary'
            for s in sample_info['samples']
        ]
        
        print(f"Successfully processed {len(samples_metrics)} samples")
        signal_io.save_metrics(
            samples_metrics,
            output_path,
            sample_names=sample_names,
            control_flags=control_flags
        )

        return 0
        
    except Exception as e:
        print(f"Error processing signals: {e}")
        return None, 1



def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-read per-position signal information from remora resquiggled data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--sample-yaml", type=Path, required=True,
                       help="YAML samples file")
    parser.add_argument("--config", type=Path, default="config.toml",
                       help="Path to TOML configuration file")
    parser.add_argument("--signal-outfile", type=Path, default="signals.hdf5",
                       help="Path to store resquiggled signal in")
    parser.add_argument("--start-spacing", type=int,
                       help="Override start spacing from config")
    parser.add_argument("--end-spacing", type=int,
                       help="Override end spacing from config")
    parser.add_argument("--metrics", type=str, default="dwell_trimmean_trimsd",
                       help="Metrics to compute (comma-separated)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    _, return_code = main(
        sample_info=args.sample_yaml,
        config_file=args.config,
        output_path=args.signal_outfile,
        start_spacing=args.start_spacing,
        end_spacing=args.end_spacing,
        metrics=args.metrics
    )
    
    exit(return_code)