# NPZ experiment scripts

## Description

This directory contains the NPZ training workload, the Slurm script used to launch the experiments, and the Darshan runtime configuration.

## `npz_tr_106.py`

PyTorch training workload for a tabular binary classification dataset stored in an NPZ archive.

The archive is expected to contain:

```text
X.npy    # shape: (num_samples, num_features)
y.npy    # shape: (num_samples,)
```

The script implements four loading strategies:

- `eager`: loads both arrays into memory with `np.load` before training.
- `lazy`: seeks directly to the requested row inside `X.npy` and `y.npy` for every sample access.
- `stream`: reads both arrays sequentially in contiguous blocks of rows. With multiple workers, each worker is assigned a separate row range.
- `burst`: uses the same sequential dataset as `stream`, with `prefetch_factor` passed to the PyTorch DataLoader.

The row-level access used by `lazy` and `stream` requires an uncompressed NPZ archive created with `numpy.savez`. These strategies inspect and seek within the `.npy` members directly. Archives produced with `numpy.savez_compressed` cannot be used for this type of row access.

The `eager` strategy reads the complete arrays and does not have this restriction.

Three model configurations are available:

```text
M1    single linear layer
M2    two-hidden-layer MLP
M3    larger MLP with additional computation per batch
```

Example:

```bash
python3 npz_tr_106.py \
    --data-path /path/to/dataset.npz \
    --strategy eager \
    --model M2 \
    --batch-size 64 \
    --num-workers 32 \
    --epochs 1 \
    --persistent-workers
```

A stream run can specify how many rows are read in each contiguous block:

```bash
python3 npz_tr_106.py \
    --data-path /path/to/dataset.npz \
    --strategy stream \
    --model M2 \
    --batch-size 64 \
    --num-workers 32 \
    --block-rows 1024 \
    --epochs 1 \
    --persistent-workers
```

If `--block-rows auto` is used, the script uses 1024 rows per block.

The complete argument list is available with:

```bash
python3 npz_tr_106.py --help
```

## `106jobpost.sh`

Slurm launcher used for the NPZ campaign on a Perlmutter CPU node.

The job requests:

```text
nodes            1
cpus-per-task    256
constraint       cpu
qos              regular
time             08:00:00
```

The experiment configuration is defined near the beginning of the script:

```bash
STRATEGY=stream
SHUFFLE=false
MODEL=M2
BATCHSIZE=64
NUM_WORKERS=32
EPOCHS=1
BLOCK_ROWS=1024
PERSISTENT_WORKERS=true
```

This launcher accepts `eager` and `stream`. The underlying Python workload also implements `lazy` and `burst`, but they are not included in the experiment matrix handled by this launcher.

The NPZ campaign represented by this script fixes `shuffle=false`. `npz_tr_106.py` constructs its DataLoaders without shuffling.

`BLOCK_ROWS` is passed only for `stream` runs. The campaign uses 1024 except when running a separate block-size experiment.

The dataset used by the script is:

```text
mc_normalized_rntuple_10M_106.npz
```

Each execution creates a timestamped run directory:

```text
runs/<RUN_ID>/
├── logs/
└── darshan_logs/
```

For stream runs, the run identifier also records `block_rows`.

The launcher uses:

```text
npz_tr_106.py
darshan_env_106.conf
```

from the Slurm submission directory.

Several paths are specific to the environment in which the campaign was run, including the Python virtual environment, dataset directory, Darshan library, and Slurm account. They should be changed if the script is reused in another environment.

Submit a run with:

```bash
sbatch 106jobpost.sh
```

## `darshan_env_106.conf`

Darshan runtime configuration used by the launcher.

The configuration enables the DXT POSIX and MPI-IO modules:

```text
MOD_ENABLE DXT_POSIX,DXT_MPIIO
```

It also sets the following record limits:

```text
MAX_RECORDS 5000   POSIX
MAX_RECORDS 325520 DXT_POSIX
```

File-name filtering excludes several system and software paths and includes accesses under:

```text
/pscratch/sd/s/satt/
```

Other settings used by this configuration include:

```text
MODMEM 100
DXT_SMALL_IO_TRIGGER .01
APP_INCLUDE python
DUMP_CONFIG
```

The application exclusion list removes commands such as `git`, `ls`, `srun`, `tee`, and `prmon` from the collected application records.

## Typical run

The three files are used together by the batch launcher:

```text
106jobpost.sh
    |
    +-- npz_tr_106.py
    |
    +-- darshan_env_106.conf
    |
    v
runs/<RUN_ID>/
    ├── logs/
    └── darshan_logs/
```

After selecting the experiment parameters in `106jobpost.sh`, submit the job with:

```bash
sbatch 106jobpost.sh
```