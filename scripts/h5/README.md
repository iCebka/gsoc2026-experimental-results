# HDF5 experiment scripts

## Description

This directory contains the HDF5 workload, Slurm launcher, and Darshan configuration used for the HDF5 experimental campaign.

### `h5_tr_106.py`

PyTorch training workload for a tabular binary classification dataset stored in HDF5.

The input file must contain two datasets at its root:

```text
X    # shape: (num_samples, num_features)
y    # shape: (num_samples,)
```

Four data-loading strategies are implemented:

- `eager`: reads the complete `X` and `y` arrays into memory before training.
- `lazy`: keeps the HDF5 file on disk and reads one requested row at a time. Each worker opens its own file handle on first access.
- `stream`: reads contiguous blocks in file order. With multiple workers, the row range is divided into non-overlapping spans. The DataLoader prefetch factor is fixed at 1.
- `burst`: uses the same block-based dataset as `stream`, but allows a larger DataLoader `prefetch_factor`.

The script also contains three model configurations:

- `M1`: single linear layer.
- `M2`: two-hidden-layer MLP.
- `M3`: larger MLP used to increase computation per batch.

Example:

```bash
python3 h5_tr_106.py \
    --data-path /path/to/dataset.h5 \
    --strategy eager \
    --model M2 \
    --batch-size 64 \
    --num-workers 32 \
    --epochs 1 \
    --persistent-workers
```

For the block-based strategies:

```bash
python3 h5_tr_106.py \
    --data-path /path/to/dataset.h5 \
    --strategy stream \
    --model M2 \
    --batch-size 64 \
    --num-workers 32 \
    --block-rows auto \
    --epochs 1 \
    --persistent-workers
```

When `--block-rows auto` is used, the stream dataset takes the row dimension from the HDF5 chunk shape. If the file is not chunked, the script falls back to 1024 rows per block.

See:

```bash
python3 h5_tr_106.py --help
```

for the complete argument list.

### `106jobpost.sh`

Slurm launcher used to run the HDF5 workload on a Perlmutter CPU node and collect Darshan logs.

The submitted job requests:

```text
nodes            1
cpus-per-task    256
constraint       cpu
qos              regular
time             05:00:00
```

The experiment parameters are defined near the beginning of the script:

```bash
STRATEGY=eager
SHUFFLE=false
MODEL=M2
BATCHSIZE=64
NUM_WORKERS=32
EPOCHS=1
BLOCK_ROWS=auto
PERSISTENT_WORKERS=true
```

For the worker sweep represented by this launcher, only `STRATEGY` and `NUM_WORKERS` were intended to change between runs.

Although `h5_tr_106.py` implements `eager`, `lazy`, `stream`, and `burst`, this launcher currently accepts only:

```text
eager
stream
```

The dataset is selected through `DATA_DIR` and `DATA_VARIANT`. Numbered variants refer to physical copies such as:

```text
mc_normalized_rntuple_10M_6.h5
```

Each execution creates a timestamped directory under:

```text
runs/<RUN_ID>/
├── logs/
└── darshan_logs/
```

The run identifier records the main experiment parameters. Stream runs also include `block_rows`.

Before submitting the script, set the installation-specific paths marked in the file:

```bash
ATLAS_LOCAL_ROOT_BASE=
source # virtual environment
DATA_DIR=
LD_PRELOAD=
```

Submit with:

```bash
sbatch 106jobpost.sh
```

### `darshan_env_26.conf`

Darshan runtime configuration used for the campaign.

It enables:

```text
DXT_POSIX
DXT_MPIIO
```

and raises the record limits used for POSIX and DXT POSIX data:

```text
MAX_RECORDS 5000   POSIX
MAX_RECORDS 325520 DXT_POSIX
```

The configuration also:

- excludes several system and software paths from file-name recording;
- includes accesses under the configured Perlmutter scratch path;
- reserves additional Darshan module memory;
- limits application instrumentation to the relevant Python workload;
- enables the DXT small-I/O trigger;
- dumps the active Darshan configuration into the generated log.

The current `NAME_INCLUDE` rule contains a campaign-specific scratch path:

```text
/pscratch/sd/s/satt/
```

Change it if the experiments are run from a different location.

## Darshan configuration path

`106jobpost.sh` must point to the configuration file stored in this directory.

If the file is named:

```text
darshan_env_26.conf
```

the launcher should contain:

```bash
export DARSHAN_CONFIG_PATH=$localdir/darshan_env_26.conf
```

rather than a reference to an older configuration filename.

## Typical run

After setting the environment, dataset directory, Darshan library, and configuration path:

```bash
sbatch 106jobpost.sh
```

The launcher validates the selected strategy and dataset path, creates the run directories, prints the experiment configuration, and starts `h5_tr_106.py` through `srun`.