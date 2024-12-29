# Modification (Outlier) detection with Gaussian Mixture Models

Triton a tool to detect outliers of a multidimensional Gaussian Mixture Model (named because of the 3 dimensions signal mean, standard deviation and dwell time). 

It is targeted towards Nanopore sequencing. Specifically, it contains a wrapper around remora for resquiggling in addition to a custom storage format to store resquiggled signals that makes it relatively efficient. Then, given a control and some potentially modified sample, for each position a GMM is fit based on the control observations. Based on how likely the potentially modified observation is within this model, we can give it a probability of being part of the same distribution (in other words, of it being unmodified, i.e. a p value). 

This tool is primarily built for de novo modification detection using sample-compare statistics. This is needed in cases where it is infeasible to generate ground modification truth datasets. In particular, we employ it to detect chemically introduced modifications that are present at high rates. 

This tool is somewhat similar to other GMM tools such as Nanocompore or xPore, with the difference that we don't attempt to create components for potentially modified observations. We don't do this because this would potentially bias the detection towards/against different types of modifications. In addition, we have observed that even fully unmodified samples can have multiple components, as such assigning either of these to be 'modified' would result in a larger number of false-positive modification calls. 

This tool does not have base-resolution, as modifications on a single base is likely to affect the signal of surrounding bases as well, likely resulting in these also being detected as modified. However, it is a good starting point when attempting to detect modifiations de novo. 

## How to run it

First run remora resquiggle. This requires a pod5 file and alignment files (soft-clipped, with mv,ns,ts,sp tags, see [bambutler](https://github.com/patbohn/bambutler) if yours aren't). This will generate the resquiggle.hdf5 files required for triton. 
Then run trition (see `python triton_cli.py -h`). Specify `sample: control` pairs in a yaml file and add that as input parameter. 

### Run Remora Resquiggle

```bash
usage: remora_resquiggle.py [-h] --pod5-dir POD5_DIR --ref-file REF_FILE --sample-bam-files
                            SAMPLE_BAM_FILES [SAMPLE_BAM_FILES ...] --sample-names SAMPLE_NAMES
                            [SAMPLE_NAMES ...] --config CONFIG [--signal-outfile SIGNAL_OUTFILE]
                            [--start-spacing START_SPACING] [--end-spacing END_SPACING]
                            [--metrics METRICS]

Extract per-read per-position signal information from remora resquiggled data.

options:
  -h, --help            show this help message and exit
  --pod5-dir POD5_DIR   Location of pod5 dir/file (default: None)
  --ref-file REF_FILE   Location of reference file (default: None)
  --sample-bam-files SAMPLE_BAM_FILES [SAMPLE_BAM_FILES ...]
                        Location of bam files (default: None)
  --sample-names SAMPLE_NAMES [SAMPLE_NAMES ...]
                        Sample names (default: None)
  --config CONFIG       Path to TOML configuration file (default: config.toml)
  --signal-outfile SIGNAL_OUTFILE
                        Path to store resquiggled signal in (default: signals.hdf5)
  --start-spacing START_SPACING
                        Override start spacing from config (default: 0)
  --end-spacing END_SPACING
                        Override end spacing from config (default: 0)
  --metrics METRICS     Metrics to compute (comma-separated) (default: dwell_trimmean_trimsd)
```


### Run Triton

```bash
usage: triton_cli.py [-h] --input-files INPUT_FILES [INPUT_FILES ...] --control-map CONTROL_MAP
                     --stats-output STATS_OUTPUT --pvals-output PVALS_OUTPUT [--drop-short-dwell]
                     [--n-components N_COMPONENTS]

Analyze modifications in signal data using control samples.

options:
  -h, --help            show this help message and exit
  --input-files INPUT_FILES [INPUT_FILES ...]
                        HDF5 files containing sample metrics (default: None)
  --control-map CONTROL_MAP
                        YAML file mapping samples to their controls (default: None)
  --stats-output STATS_OUTPUT
                        Output TSV file for statistics (default: None)
  --pvals-output PVALS_OUTPUT
                        Output HDF5 file for per-read p-values (default: None)
  --drop-short-dwell    Drop events that have a dwell time below 6 signals (default: False)
  --n-components N_COMPONENTS
                        Number of Gaussian components to fit (default: 3)
```