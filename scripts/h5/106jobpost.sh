#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=05:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account= # Insert account


# ===========================================================================
# Environment
# ===========================================================================

# Keep the same software setup used in the previous HDF5 campaign.
export ATLAS_LOCAL_ROOT_BASE= # INSERT ATLAS ROOT BASE
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

lsetup prmon
asetup Athena,main--dev3LCG,latest

source # INSERT venv DIRECTORY


# ===========================================================================
# Thread configuration
#
# ===========================================================================

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS
unset NUMEXPR_NUM_THREADS

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-<unset>}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-<unset>}"

echo "python3 resolved to: $(which python3)"
python3 -c "import torch, h5py; print('torch:', torch.__file__); print('h5py :', h5py.__file__)"

localdir=$SLURM_SUBMIT_DIR


# ===========================================================================
# Experiment configuration
#
# FOR THE MAIN WORKER SWEEP, CHANGE ONLY:
#
#     STRATEGY
#     NUM_WORKERS
#
# Everything else remains fixed.
# ===========================================================================

STRATEGY=eager          # eager | stream
SHUFFLE=false

MODEL=M2

BATCHSIZE=64
NUM_WORKERS=32
EPOCHS=1

# Only used by stream.
# auto = use the row dimension of X's HDF5 chunk.
BLOCK_ROWS=auto

PERSISTENT_WORKERS=true

LOG_EVERY=5000

CPUS_PER_TASK=256
WORKFLOW=hdf5_nocap


# ===========================================================================
# Dataset selection
#
# Change DATA_VARIANT if you want to use another physical copy of D10.
#
# DATA_VARIANT=3 -> mc_normalized_rntuple_10M_3.h5
#
# The unnumbered dataset can be selected separately if ever needed.
# ===========================================================================

DATA_VARIANT=6

DATA_DIR= # /data/converted_10M
DATA_FILE="mc_normalized_rntuple_10M_${DATA_VARIANT}.h5"
DATA_PATH="$DATA_DIR/$DATA_FILE"


# ===========================================================================
# Configuration validation
# ===========================================================================

if [[ "$STRATEGY" != "eager" && "$STRATEGY" != "stream" ]]; then
    echo "ERROR: STRATEGY must be eager or stream, got: $STRATEGY"
    exit 1
fi

if [[ "$SHUFFLE" == "true" ]]; then
    SHUFFLE_ARG="--shuffle"
elif [[ "$SHUFFLE" == "false" ]]; then
    SHUFFLE_ARG="--no-shuffle"
else
    echo "ERROR: SHUFFLE must be true or false, got: $SHUFFLE"
    exit 1
fi

if [[ "$NUM_WORKERS" -lt 1 ]]; then
    echo "ERROR: NUM_WORKERS must be >= 1"
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: dataset does not exist:"
    echo "       $DATA_PATH"
    exit 1
fi


# ===========================================================================
# Run identity
# ===========================================================================

if [[ "$STRATEGY" == "stream" ]]; then
    RUN_ID="${WORKFLOW}_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_blockrows_${BLOCK_ROWS}_epochs_${EPOCHS}_data_${DATA_VARIANT}_$(date +"%Y%m%d_%H%M%S")"
else
    RUN_ID="${WORKFLOW}_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_epochs_${EPOCHS}_data_${DATA_VARIANT}_$(date +"%Y%m%d_%H%M%S")"
fi


# ===========================================================================
# Script / output
# ===========================================================================

SCRIPT=$localdir/h5_tr_106.py

OUTPUT_DIR=$localdir/runs/$RUN_ID

LOG_FILE="${OUTPUT_DIR}/logs/log.strategy_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_epochs_${EPOCHS}.txt"

export DARSHAN_LOGDIR="${OUTPUT_DIR}/darshan_logs/strategy_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_epochs_${EPOCHS}"

export DARSHAN_LOGPATH="$DARSHAN_LOGDIR"


# ===========================================================================
# Darshan
#
# Historical mechanism kept unchanged.
# ===========================================================================

export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_CONFIG_PATH=$localdir/darshan_env_6.conf

export LD_PRELOAD= # INSERT Darshan Library directory: libdarshan.so
export DXT_ENABLE_IO_TRACE=1
export DARSHAN_DUMP_CONFIG=0


# ===========================================================================
# Output setup
# ===========================================================================

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$DARSHAN_LOGDIR"

rm -rf "$DARSHAN_LOGDIR"/*


# ===========================================================================
# Diagnostics
# ===========================================================================

echo "============================================================"
echo "Starting HDF5
echo "============================================================"

echo "Job ID:              $SLURM_JOB_ID"
echo "Host:                $(hostname)"
echo "Date:                $(date)"

echo
echo "Strategy:            $STRATEGY"
echo "Shuffle:             $SHUFFLE"
echo "Model:               $MODEL"
echo "Batch Size:          $BATCHSIZE"
echo "Workers:             $NUM_WORKERS"
echo "Epochs:              $EPOCHS"
echo "Persistent Workers:  $PERSISTENT_WORKERS"

if [[ "$STRATEGY" == "stream" ]]; then
    echo "Block Rows:          $BLOCK_ROWS"
else
    echo "Block Rows:          N/A"
fi

echo
echo "Dataset variant:     $DATA_VARIANT"
echo "Data Path:           $DATA_PATH"
echo "Script:              $SCRIPT"
echo "Output Directory:    $OUTPUT_DIR"
echo "Darshan Log Path:    $DARSHAN_LOGPATH"
echo "Darshan Config:      $DARSHAN_CONFIG_PATH"
echo "LD_PRELOAD:          $LD_PRELOAD"

echo "============================================================"


# ===========================================================================
# Application arguments
# ===========================================================================

ARGS=(
    --data-path "$DATA_PATH"
    --strategy "$STRATEGY"
    "$SHUFFLE_ARG"
    --model "$MODEL"
    --batch-size "$BATCHSIZE"
    --num-workers "$NUM_WORKERS"
    --epochs "$EPOCHS"
    --log-every "$LOG_EVERY"
)

if [[ "$PERSISTENT_WORKERS" == "true" ]]; then
    ARGS+=(--persistent-workers)
fi

if [[ "$STRATEGY" == "stream" ]]; then
    ARGS+=(--block-rows "$BLOCK_ROWS")
fi


# ===========================================================================
# Run
# ===========================================================================

echo
echo "Launching workload..."
echo

srun \
    --cpu-bind=cores \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    python3 -u "$SCRIPT" "${ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

STATUS=${PIPESTATUS[0]}

echo
echo "============================================================"
echo "Application exit status: $STATUS"
echo "Finished: $(date)"
echo "============================================================"

deactivate

exit "$STATUS"
