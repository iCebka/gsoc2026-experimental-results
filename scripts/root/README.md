# ROOT experiment scripts

## Description

This directory contains the ROOT training workload, the Slurm launcher used for the experiments, and its Darshan runtime configuration.

## `root_tr_116.py`

PyTorch training workload for a tabular binary classification dataset stored in a ROOT TTree or RNTuple.

The script uses `ROOT.Experimental.ML.RDataLoader` for the `eager`, `stream`, and `burst` strategies. These strategies do not use `torch.utils.data.DataLoader`. ROOT handles data reading and its internal buffering directly.

The `lazy` strategy is implemented separately with `ROOT.RNTupleReader` and a PyTorch DataLoader.

Four loading strategies are available:

- `eager`: constructs an `RDataLoader` with `load_eager=True`, loading the dataset into memory before training.
- `stream`: uses `load_eager=False` and reads batches from the ROOT file with `batches_in_memory=2`.
- `burst`: uses the same non-eager RDataLoader path, with `batches_in_memory` set through `--prefetch-factor`.
- `lazy`: performs row-indexed access through `RNTupleReader.GetView`. Each requested row is assembled by reading the selected feature fields and target field at that entry index.

The `lazy` implementation uses `RNTupleReader`, so this strategy requires an RNTuple even though the RDataFrame/RDataLoader paths can operate on either a TTree or an RNTuple.

Feature columns are selected from the numeric columns reported by `RDataFrame`. The target column is excluded from the feature list.

Three model configurations are available:

```text
M1    single linear layer
M2    two-hidden-layer MLP
M3    larger MLP with additional computation per batch
```

Example eager run:

```bash
python3 root_tr_116.py \
    --data-path /path/to/dataset.root \
    --tree-name tree \
    --target-col Label \
    --strategy eager \
    --shuffle \
    --model M2 \
    --batch-size 64 \
    --epochs 1
```

Example stream run:

```bash
python3 root_tr_116.py \
    --data-path /path/to/dataset.root \
    --tree-name tree \
    --target-col Label \
    --strategy stream \
    --no-shuffle \
    --model M2 \
    --batch-size 64 \
    --epochs 1
```

For burst, `--prefetch-factor` is passed to the RDataLoader as `batches_in_memory`:

```bash
python3 root_tr_116.py \
    --data-path /path/to/dataset.root \
    --tree-name tree \
    --target-col Label \
    --strategy burst \
    --no-shuffle \
    --model M2 \
    --batch-size 64 \
    --prefetch-factor 16384 \
    --epochs 1
```

For lazy, PyTorch worker settings are available because this is the only strategy that uses `torch.utils.data.DataLoader`:

```bash
python3 root_tr_116.py \
    --data-path /path/to/dataset.root \
    --tree-name tree \
    --target-col Label \
    --strategy lazy \
    --no-shuffle \
    --model M2 \
    --batch-size 64 \
    --num-workers 32 \
    --persistent-workers \
    --epochs 1
```

`--root-threads` can be used to call `ROOT.EnableImplicitMT()` for ROOT-level parallelism. It is separate from `--num-workers`, which applies only to `lazy`.

The complete argument list is available with:

```bash
python3 root_tr_116.py --help
```

## `116jobpost.sh`

Slurm launcher for the ROOT workload on a Perlmutter CPU node.

The job requests:

```text
nodes            1
cpus-per-task    256
constraint       cpu
qos              regular
time             03:00:00
```

The supplied launcher is configured for:

```bash
STRATEGY=burst
SHUFFLE=true
MODEL=M2
BATCHSIZE=64
EPOCHS=10
BATCHES_IN_MEMORY=16384
```

`BATCHES_IN_MEMORY` is passed to `root_tr_116.py` through `--prefetch-factor`.

The input used by this version of the launcher is:

```text
mc_normalized_rntuple_10M_116.root
```

Each run receives an identifier containing the strategy, `batches_in_memory`, shuffle setting, and timestamp:

```text
root_<strategy>_bim_<value>_shuffle_<value>_<timestamp>
```

Output is written under:

```text
runs/<RUN_ID>/
├── logs/
└── darshan_logs/
```

The launcher uses:

```text
root_tr_116.py
darshan_env_116.conf
```

from the Slurm submission directory.

Several values are left as environment-specific placeholders and must be set before submission:

```bash
#SBATCH --account=
ATLAS_LOCAL_ROOT_BASE=
source # virtual environment
DATA_DIR=
LD_PRELOAD=
```

Submit the job with:

```bash
sbatch 116jobpost.sh
```

### Shuffle setting

The supplied `116jobpost.sh` sets:

```bash
STRATEGY=burst
SHUFFLE=true
```

This is a valid configuration accepted by the workload.

For experiments intended to compare `stream` and `burst` while changing only the RDataLoader buffer depth, use the same shuffle setting for both runs. The comparison described in `root_tr_116.py` uses `shuffle=false`.

## `darshan_env_116.conf`

Darshan runtime configuration used by the ROOT launcher.

The configuration enables:

```text
DXT_POSIX
DXT_MPIIO
```

and sets:

```text
MAX_RECORDS 5000   POSIX
MAX_RECORDS 325520 DXT_POSIX
MODMEM 100
```

The active name filter includes Python and ROOT files:

```text
NAME_INCLUDE .py$,.root$ *
```

The application filter includes:

```text
python
python3
```

and excludes commands such as `git`, `ls`, `srun`, `tee`, and `prmon`.

DXT tracing uses:

```text
DXT_SMALL_IO_TRIGGER .01
```

and `DUMP_CONFIG` records the active Darshan configuration.

## Typical run

The batch script connects the three files:

```text
116jobpost.sh
    |
    +-- root_tr_116.py
    |
    +-- darshan_env_116.conf
    |
    v
runs/<RUN_ID>/
    ├── logs/
    └── darshan_logs/
```

After setting the environment-specific paths and the desired experiment parameters:

```bash
sbatch 116jobpost.sh
```