#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=03:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account= INSERT Account

export ATLAS_LOCAL_ROOT_BASE= # INSERT ATLAS ROOT BASE
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup prmon
#asetup Athena,main--dev3LCG,latest

source # INSERT venv DIRECTORY

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-<unset>}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-<unset>}"

echo "python3 resolved to: $(which python3)"

localdir=$SLURM_SUBMIT_DIR

STRATEGY=burst
SHUFFLE=true
if [[ "$SHUFFLE" == "true" ]]; then
    SHUFFLE_ARG="--shuffle"
elif [[ "$SHUFFLE" == "false" ]]; then
    SHUFFLE_ARG="--no-shuffle"
else
    echo "ERROR: SHUFFLE must be true or false, got: $SHUFFLE"
    exit 1
fi
MODEL=M2
BATCHSIZE=64
EPOCHS=10
LOG_EVERY=50000

CPUS_PER_TASK=256
WORKFLOW=root
BATCHES_IN_MEMORY=16384

RUN_ID="${WORKFLOW}_${STRATEGY}_bim_${BATCHES_IN_MEMORY}_shuffle_${SHUFFLE}_$(date +"%Y%m%d_%H%M%S")"

DATA_DIR= # Insert Data Directory /data/roots
DATA_FILE=mc_normalized_rntuple_10M_116.root
DATA_PATH=$DATA_DIR/$DATA_FILE

SCRIPT=$localdir/root_tr_116.py
OUTPUT_DIR=$localdir/runs/$RUN_ID

export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_CONFIG_PATH=$localdir/darshan_env_116.conf
export LD_PRELOAD= # Insert Darshan library directory libdarshan.so
export DXT_ENABLE_IO_TRACE=1
export DARSHAN_DUMP_CONFIG=0

LOG_FILE="${OUTPUT_DIR}/logs/log.strategy_${STRATEGY}_bim_${BATCHES_IN_MEMORY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}.txt"

export DARSHAN_LOGDIR="${OUTPUT_DIR}/darshan_logs/strategy_${STRATEGY}_bim_${BATCHES_IN_MEMORY}_shuffle_${SHUFFLE}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}"
export DARSHAN_LOGPATH="$DARSHAN_LOGDIR"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$DARSHAN_LOGDIR"
rm -rf "$DARSHAN_LOGDIR"/*

echo "Starting Test..."
echo "Strategy: $STRATEGY"
echo "Shuffle: $SHUFFLE"
echo "Batch Size: $BATCHSIZE"
echo "Batches in Memory: $BATCHES_IN_MEMORY"
echo "Epochs: $EPOCHS"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"

srun --cpu-bind=cores --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" python3 "$SCRIPT" \
    --data-path "$DATA_PATH" \
    --tree-name tree \
    --target-col Label \
    --strategy "$STRATEGY" \
    "$SHUFFLE_ARG" \
    --model "$MODEL" \
    --batch-size "$BATCHSIZE" \
    --epochs "$EPOCHS" \
    --log-every "$LOG_EVERY" \
    --prefetch-factor "$BATCHES_IN_MEMORY" \
    2>&1 | tee "$LOG_FILE"

deactivate
echo "Test Complete."
