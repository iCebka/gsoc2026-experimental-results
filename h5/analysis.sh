#!/usr/bin/env bash
# analyze_h5.sh -- final post-analysis for the HDF5 campaign.
#
# Usage:
#   bash analysis.sh
#   bash analysis.sh /another/campaign/runs
#
# Each run stores derived results under:
#   <run>/analysis/
#
# Raw Darshan logs and workload logs are never modified or deleted.
# Existing .darshan.json files are reused.
#
# All CSV outputs and all available graph types are generated.

set -uo pipefail

POST=/pscratch/sd/s/satt/sprints/cli-test/ccode/darshanflow/post

RUNS="${1:-/pscratch/sd/s/satt/sprints/final-lap/processed/h5-compact}"

# ---------------------------------------------------------------------------
# Final HDF5 analysis configuration.
#
# These values are intentionally fixed. They do NOT inherit values from the
# shell environment, preventing ROOT/NPZ settings from contaminating HDF5.
#
# The dataset expression matches every filename ending in .h5, including:
#
#   mc_normalized_rntuple_10M.h5
#   mc_normalized_rntuple_10M_1.h5
#   mc_normalized_rntuple_10M_10.h5
#   mc_normalized_rntuple_10M_101.h5
#
# The script expression matches:
#
#   h5_tr.py
#   h5_tr_2.py
#   h5_tr_23.py
#   h5_tr_203.py
#   h5_tr_210.py
# ---------------------------------------------------------------------------

INCLUDE='\.h5$|h5_tr(_[0-9]+)?\.py$'

GRAPHS=true
COMPACT=true
INDIVIDUAL=false
META_INDIVIDUAL=false


echo "======================================================================"
echo "HDF5 ANALYSIS CONFIGURATION"
echo "======================================================================"
echo "RUNS=$RUNS"
echo "POST=$POST"
echo "INCLUDE=$INCLUDE"
echo "GRAPHS=$GRAPHS"
echo "COMPACT=$COMPACT"
echo "INDIVIDUAL=$INDIVIDUAL"
echo "META_INDIVIDUAL=$META_INDIVIDUAL"
echo "======================================================================"
echo


# ---------------------------------------------------------------------------
# Basic setup validation.
# ---------------------------------------------------------------------------

if [[ ! -d "$RUNS" ]]; then
    echo "ERROR: runs directory does not exist:" >&2
    echo "  $RUNS" >&2
    exit 1
fi

required_scripts=(
    simplify_logs.py
    io_intensity.py
    access_pattern.py
    effective_read.py
    worker_balance.py
    metadata_pressure.py
)

for script in "${required_scripts[@]}"; do
    if [[ ! -f "$POST/$script" ]]; then
        echo "ERROR: missing post-analysis script:" >&2
        echo "  $POST/$script" >&2
        exit 1
    fi
done


# ---------------------------------------------------------------------------
# Validate simplified.json.
#
# This guard exists specifically to prevent a wrong filter from silently
# producing apparently valid analyses with zero HDF5 reads.
# ---------------------------------------------------------------------------

validate_simplified() {
    local simplified="$1"

    python3 - "$simplified" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as fh:
    doc = json.load(fh)

logs = doc.get("logs", [])

h5_records = 0
script_records = 0

for log in logs:
    for rec in log.get("records", []):
        name = str(rec.get("file", ""))

        if name.endswith(".h5"):
            h5_records += 1

        base = name.rsplit("/", 1)[-1]

        if (
            base == "h5_tr.py"
            or (
                base.startswith("h5_tr_")
                and base.endswith(".py")
                and base[len("h5_tr_"):-len(".py")].isdigit()
            )
        ):
            script_records += 1

print(
    f"  simplified validation: "
    f"{len(logs)} logs, "
    f"{h5_records} HDF5 records, "
    f"{script_records} training-script records"
)

if h5_records == 0:
    print(
        "ERROR: simplified.json contains no .h5 records.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}


analyze_run() {
    local run="$1"
    local logs="$run/darshan_logs"
    local out="$run/analysis"

    # -----------------------------------------------------------------------
    # Start from a clean derived-analysis directory.
    #
    # Only <run>/analysis is removed. Raw Darshan files, converted Darshan
    # JSON files and workload logs remain untouched.
    # -----------------------------------------------------------------------

    rm -rf "$out"
    mkdir -p "$out"

    # -----------------------------------------------------------------------
    # 1. Convert all Python Darshan logs to JSON.
    #
    # DataLoader workers are Python processes too, so every
    # satt_python3_*.darshan file belonging to the run is retained.
    #
    # Existing JSON conversions are reused.
    # -----------------------------------------------------------------------

    local f

    for f in "${pylogs[@]}"; do
        if [[ ! -f "$f.json" ]]; then
            echo "  converting: $(basename "$f")"

            python3 -m darshan to_json "$f" \
                > "$f.json.tmp" || return 1

            mv "$f.json.tmp" "$f.json"
        fi
    done

    # -----------------------------------------------------------------------
    # 2. Normalize Darshan records.
    #
    # simplified.json:
    #   only HDF5 datasets and HDF5 training scripts.
    #
    # simplified_all.json:
    #   all Python-process records, used for metadata-pressure analysis.
    # -----------------------------------------------------------------------

    python3 "$POST/simplify_logs.py" \
        "$logs" \
        -i "$INCLUDE" \
        -o "$out/simplified.json" || return 1

    python3 "$POST/simplify_logs.py" \
        "$logs" \
        -o "$out/simplified_all.json" || return 1

    # -----------------------------------------------------------------------
    # Explicitly validate that HDF5 records survived the dataset filter.
    # -----------------------------------------------------------------------

    validate_simplified "$out/simplified.json" || return 1

    # -----------------------------------------------------------------------
    # 3. I/O intensity.
    #
    # Produces:
    #   file_metrics.csv
    #   proc_metrics.csv
    #   run_metrics.csv
    #
    # plus all supported graphs.
    # -----------------------------------------------------------------------

    python3 "$POST/io_intensity.py" \
        "$out/simplified.json" \
        -o "$out/io_intensity" \
        --graphs "$GRAPHS" || return 1

    # -----------------------------------------------------------------------
    # 4. Access pattern.
    #
    # Consecutive, sequential-with-gaps, non-sequential and seek behavior.
    # -----------------------------------------------------------------------

    python3 "$POST/access_pattern.py" \
        "$out/simplified.json" \
        -o "$out/access_pattern" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$INDIVIDUAL" || return 1

    # -----------------------------------------------------------------------
    # 5. Effective read bandwidth / run throughput.
    # -----------------------------------------------------------------------

    python3 "$POST/effective_read.py" \
        "$out/simplified.json" \
        -o "$out/effective_read" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$INDIVIDUAL" || return 1

    # -----------------------------------------------------------------------
    # 6. Worker balance.
    #
    # Relevant for HDF5 because stream/eager DataLoader configurations use
    # forked Python workers.
    # -----------------------------------------------------------------------

    python3 "$POST/worker_balance.py" \
        "$out/simplified.json" \
        -o "$out/worker_balance" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$INDIVIDUAL" || return 1

    # -----------------------------------------------------------------------
    # 7. Metadata pressure.
    #
    # Intentionally uses the unfiltered representation because metadata
    # behavior includes runtime files, libraries, /proc entries, caches, etc.
    #
    # All compact and individual graphs are generated.
    # -----------------------------------------------------------------------

    python3 "$POST/metadata_pressure.py" \
        "$out/simplified.json" \
        -o "$out/metadata_pressure" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$META_INDIVIDUAL" || return 1
}


ok=()
skipped=()
failed=()


# ---------------------------------------------------------------------------
# Process every HDF5 experiment directory.
# ---------------------------------------------------------------------------

for run in "$RUNS"/*/; do
    run="${run%/}"
    name="$(basename "$run")"

    [[ -d "$run/darshan_logs" ]] || continue

    mapfile -t pylogs < <(
        find "$run/darshan_logs" \
            -type f \
            -name 'satt_python3_*.darshan' \
            | sort
    )

    if (( ${#pylogs[@]} == 0 )); then
        echo "--- no python3 Darshan log found, skipping: $name"
        skipped+=("$name")
        continue
    fi

    echo
    echo "======================================================================"
    echo "=== $name"
    echo "=== python logs: ${#pylogs[@]}"
    echo "======================================================================"

    if analyze_run "$run"; then
        ok+=("$name")
    else
        echo "!!! failed: $name" >&2
        failed+=("$name")
    fi
done


# ---------------------------------------------------------------------------
# Campaign summary.
# ---------------------------------------------------------------------------

echo
echo "======================================================================"
echo "HDF5 CAMPAIGN SUMMARY"
echo "======================================================================"
echo "successful: ${#ok[@]}"
echo "skipped:    ${#skipped[@]}"
echo "failed:     ${#failed[@]}"
echo

for n in "${skipped[@]}"; do
    echo "  skipped: $n"
done

for n in "${failed[@]}"; do
    echo "  failed: $n"
done

echo
echo "======================================================================"
