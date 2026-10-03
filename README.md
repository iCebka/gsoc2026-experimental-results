# GSoC 2026 Experimental Dataset

This repository contains the final experimental runs from the Google Summer of
Code 2026 project **Characterizing I/O Performance for ML Data Loaders at Scale
Using Darshan**.

The experiments were run on NERSC Perlmutter using a common dataset of
10 million events represented as:

- ROOT/RNTuple
- NPZ
- HDF5

The repository contains the application logs, Darshan logs, and CSV files
generated during post-processing for each run. The workload scripts used for
the three campaigns are included as well.

The complete GSoC project description and links to the other project artifacts
are in the [project report](https://gist.github.com/iCebka/4ea6b5927a773c7bad6e66bd46276af2).

The Darshan logs were processed with the
[DarshanFlow analysis pipeline](https://github.com/iCebka/gsoc2026-darshan-internal/tree/main/code/darshanflow).


## Dataset scope

The released dataset contains **94 experimental runs**:

| Format | Experiment group | Runs |
| --- | --- | ---: |
| ROOT/RNTuple | Eager baseline | 6 |
| ROOT/RNTuple | Stream baseline | 6 |
| ROOT/RNTuple | `batches_in_memory` sweep | 32 |
| ROOT/RNTuple | Batch × BiM, product = 4096 | 4 |
| ROOT/RNTuple | Batch × BiM, product = 65536 | 4 |
| ROOT/RNTuple | Extended 50-epoch BiM experiments | 4 |
| NPZ | Eager worker sweep | 9 |
| NPZ | Stream worker sweep | 9 |
| HDF5 | Eager worker sweep | 10 |
| HDF5 | Stream worker sweep | 10 |
| **Total** | | **94** |


## Experimental matrix

### ROOT/RNTuple

ROOT experiments use `ROOT.Experimental.ML.RDataLoader`.

#### R1. Eager baseline

| Parameter | Values |
| --- | --- |
| Loading mode | Eager |
| Batch size | 64 |
| Shuffle | `{false, true}` |
| Epochs | `{1, 10, 50}` |
| Runs | 6 |

#### R2. Stream baseline

| Parameter | Values |
| --- | --- |
| Loading mode | Non-eager / stream |
| Batch size | 64 |
| Batches in memory | 2 |
| Shuffle | `{false, true}` |
| Epochs | `{1, 10, 50}` |
| Runs | 6 |

#### R3. Batches-in-memory sweep

| Parameter | Values |
| --- | --- |
| Loading mode | Non-eager / buffered |
| Batch size | 64 |
| Batches in memory | `{4, 8, 16, 64, 256, 1024, 4096, 16384}` |
| Shuffle | `{false, true}` |
| Epochs | `{1, 10}` |
| Runs | 32 |

This sweep studies the effect of the amount of data retained by the loader
between accesses while keeping the batch size fixed.

#### R4. Fixed Batch × BiM product: 4096

| Batch size | Batches in memory |
| ---: | ---: |
| 32 | 128 |
| 64 | 64 |
| 128 | 32 |
| 256 | 16 |

Common configuration:

| Parameter | Value |
| --- | --- |
| Loading mode | Non-eager / buffered |
| Shuffle | false |
| Epochs | 1 |
| `batch_size × batches_in_memory` | 4096 |
| Runs | 4 |

#### R5. Fixed Batch × BiM product: 65536

| Batch size | Batches in memory |
| ---: | ---: |
| 32 | 2048 |
| 64 | 1024 |
| 128 | 512 |
| 256 | 256 |

Common configuration:

| Parameter | Value |
| --- | --- |
| Loading mode | Non-eager / buffered |
| Shuffle | false |
| Epochs | 1 |
| `batch_size × batches_in_memory` | 65536 |
| Runs | 4 |

#### R6. Extended 50-epoch experiments

| Parameter | Values |
| --- | --- |
| Loading mode | Non-eager / buffered |
| Batch size | 64 |
| Batches in memory | `{8, 16}` |
| Shuffle | `{false, true}` |
| Epochs | 50 |
| Runs | 4 |


### NPZ

NPZ experiments use PyTorch DataLoader workloads and concentrate on worker
scaling.

#### N1. Eager worker sweep

| Parameter | Values |
| --- | --- |
| Strategy | Eager |
| Batch size | 64 |
| Workers | `{1, 2, 4, 8, 16, 32, 64, 128, 256}` |
| Shuffle | false |
| Epochs | 1 |
| Runs | 9 |

#### N2. Stream worker sweep

| Parameter | Values |
| --- | --- |
| Strategy | Stream |
| Batch size | 64 |
| Workers | `{1, 2, 4, 8, 16, 32, 64, 128, 256}` |
| Block rows | 1024 |
| Shuffle | false |
| Epochs | 1 |
| Runs | 9 |


### HDF5

HDF5 experiments also use PyTorch DataLoader workloads and extend the worker
sweep up to 512 workers.

#### H1. Eager worker sweep

| Parameter | Values |
| --- | --- |
| Strategy | Eager |
| Batch size | 64 |
| Workers | `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}` |
| Shuffle | false |
| Epochs | 1 |
| Thread caps | None |
| Data copies | `{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}` |
| Runs | 10 |

#### H2. Stream worker sweep

| Parameter | Values |
| --- | --- |
| Strategy | Stream |
| Batch size | 64 |
| Workers | `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}` |
| Block rows | auto |
| Shuffle | false |
| Epochs | 1 |
| Thread caps | None |
| Data copies | `{21, 22, 23, 24, 25, 26, 27, 28, 29, 30}` |
| Runs | 10 |


## Repository layout

At the top level, the dataset is separated by storage format together with
the workload scripts used for the campaigns.

```text
.
├── README.md
├── scripts/
├── npz/
├── h5/
└── root/
```

Each format directory contains one directory per experimental run.

Run directory names encode the main configuration and end with the execution
timestamp. For example:

```text
root_burst_bim_256_shuffle_false_batch_64_epochs_1_20260927_045840
```

corresponds to a ROOT/RNTuple run using the buffered/burst loading path with:

```text
batches_in_memory = 256
shuffle = false
batch_size = 64
epochs = 1
```

followed by its execution timestamp.


## Run directory structure

A typical run has the following structure:

```text
.
├── analysis/
│   ├── access_pattern/
│   │   ├── file_metrics.csv
│   │   ├── proc_metrics.csv
│   │   └── run_metrics.csv
│   ├── effective_read/
│   │   ├── file_metrics.csv
│   │   ├── proc_metrics.csv
│   │   └── run_metrics.csv
│   ├── io_intensity/
│   │   ├── file_metrics.csv
│   │   ├── proc_metrics.csv
│   │   └── run_metrics.csv
│   ├── metadata_pressure/
│   │   ├── file_metrics.csv
│   │   ├── proc_metrics.csv
│   │   └── run_metrics.csv
│   ├── simplified_all.json
│   └── simplified.json
│
├── darshan_logs/
│   └── <experiment-configuration>/
│       ├── *.darshan
│       └── *.darshan.json
│
└── logs/
    └── log.<experiment-configuration>.txt
```

### `darshan_logs/`

Contains the Darshan profiling output collected during the experiment.

A run may contain several `.darshan` files because different instrumented
processes involved in the workload can produce independent Darshan logs.

Where available, `*.darshan.json` is the JSON representation produced during
post-processing with PyDarshan.

The original `.darshan` files are retained as the lowest-level profiling
artifacts in this release.


### `logs/`

Contains the application-level log produced by the workload execution.

The file name encodes the corresponding experimental configuration.


### `analysis/`

Contains the structured data generated by the DarshanFlow post-processing
pipeline.

No plots are included in this dataset release. The CSV files constitute the
analysis-ready representation of the collected measurements.

The currently exported metric families are:

| Directory | Purpose |
| --- | --- |
| `io_intensity/` | I/O operation and transferred-data measurements |
| `access_pattern/` | Read-access and seek-pattern measurements |
| `metadata_pressure/` | Metadata-related activity |
| `effective_read/` | Read-performance/effective-read measurements |

Each metric family is exported at three aggregation levels:

| File | Aggregation level |
| --- | --- |
| `file_metrics.csv` | Metrics for individual files observed in the run |
| `proc_metrics.csv` | Metrics grouped at process level |
| `run_metrics.csv` | Run-level aggregate measurements |

The CSV files are the primary entry point for downstream analysis and
visualization.


### `simplified.json`

Normalized and filtered Darshan records corresponding to the workload and
dataset records selected by the post-processing pipeline.

This representation is used as input by the metric extraction modules.


### `simplified_all.json`

A broader normalized representation retaining Python-process records before
the workload/dataset-specific filtering applied to `simplified.json`.

It is retained because some analyses, particularly metadata-oriented ones,
need information outside the main dataset records.


## Workload scripts

The `scripts/` directory contains the workload programs used to produce the
experimental runs included in this release.

These scripts define the data-loading behavior exercised by the campaigns.
Experiment orchestration, Darshan instrumentation, post-processing, and metric
generation are implemented separately in DarshanFlow.

For the complete execution and analysis implementation, see the main project
repository and the DarshanFlow source linked above.


## Analysis data

The dataset intentionally provides the processed CSV outputs in addition to
the original Darshan logs.

This allows the results to be used at two different levels:

1. the `.darshan` files can be inspected or reprocessed from the original
   profiling data; and
2. the CSV files can be used directly for statistical analysis,
   cross-experiment comparison, and visualization.

The release does not prescribe a particular plotting or statistical workflow.


## Context

This dataset corresponds to the final experimental campaign completed during
Google Summer of Code 2026.

Earlier exploratory runs used additional formats and configurations, including
binary and CSV workloads. Those exploratory campaigns are not part of this
dataset snapshot.

For the complete project context, implementation history, documentation, and
GSoC report, see the main project repository.
