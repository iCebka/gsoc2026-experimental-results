#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=08:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m2845


# ===========================================================================
# Environment
# ===========================================================================

source # INSERT venv DIRECTORY

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-<unset>}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-<unset>}"

echo "python3 resolved to: $(which python3)"

localdir=$SLURM_SUBMIT_DIR


# ===========================================================================
# Experiment configuration
# ===========================================================================

STRATEGY=stream
SHUFFLE=false

MODEL=M2

BATCHSIZE=64
NUM_WORKERS=32
EPOCHS=1

# Fixed for all NPZ stream experiments unless explicitly running the
# separate block-size experiment.
BLOCK_ROWS=1024

PERSISTENT_WORKERS=true

LOG_EVERY=5000

CPUS_PER_TASK=256
WORKFLOW=npz


# ===========================================================================
# Configuration validation
# ===========================================================================

if [[ "$STRATEGY" != "eager" && "$STRATEGY" != "stream" ]]; then
    echo "ERROR: STRATEGY must be eager or stream, got: $STRATEGY"
    exit 1
fi

# Current NPZ campaign is intentionally shuffle=false.
# eager must therefore use the reviewed npz_tr.py where shuffle=False.
# stream is intrinsically sequential in the current implementation.
if [[ "$SHUFFLE" != "false" ]]; then
    echo "ERROR: NPZ campaign currently requires SHUFFLE=false."
    exit 1
fi

if [[ "$NUM_WORKERS" -lt 1 ]]; then
    echo "ERROR: NUM_WORKERS must be >= 1 because persistent_workers is enabled."
    exit 1
fi


# ===========================================================================
# Run identity
# ===========================================================================

if [[ "$STRATEGY" == "stream" ]]; then
    RUN_ID="${WORKFLOW}_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_blockrows_${BLOCK_ROWS}_epochs_${EPOCHS}_$(date +"%Y%m%d_%H%M%S")"
else
    RUN_ID="${WORKFLOW}_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_epochs_${EPOCHS}_$(date +"%Y%m%d_%H%M%S")"
fi


# ===========================================================================
# Input
# ===========================================================================

DATA_DIR= # /data/converted_10M
DATA_FILE= # mc_normalized_rntuple_10M_106.npz
DATA_PATH=$DATA_DIR/$DATA_FILE


# ===========================================================================
# Script / output
# ===========================================================================

SCRIPT=$localdir/npz_tr_106.py

OUTPUT_DIR=$localdir/runs/$RUN_ID
LOG_FILE="${OUTPUT_DIR}/logs/log.strategy_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_epochs_${EPOCHS}.txt"

export DARSHAN_LOGDIR="${OUTPUT_DIR}/darshan_logs/strategy_${STRATEGY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_workers_${NUM_WORKERS}_epochs_${EPOCHS}"
export DARSHAN_LOGPATH="$DARSHAN_LOGDIR"


# ===========================================================================
# Darshan
#
# Historical campaign configuration.
# Keep this identical to ROOT/HDF5 runs.
# ===========================================================================

export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_CONFIG_PATH=$localdir/darshan_env_106.conf

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
echo "Starting NPZ experiment"
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
