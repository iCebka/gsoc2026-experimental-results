#!/usr/bin/env bash
# analyze_npz.sh -- runs post-analysis for each NPZ campaign run.
#
#   bash analyze_npz.sh
#   bash analyze_npz.sh /another/campaign/runs
#
# Each run stores its results under <run>/analysis/.
# The script is safe to rerun: it only converts .darshan files that do not
# already have a corresponding .json file.
#
# All available CSV outputs and all graph types are enabled by default,
# including individual file/process/worker graphs.

set -uo pipefail

POST=/pscratch/sd/s/satt/sprints/cli-test/ccode/darshanflow/post

RUNS="${1:-/pscratch/sd/s/satt/sprints/final-lap/processed/npz-nograph}"

# Keep NPZ dataset files and the training-script records.
#
# Matches:
#   *.npz
#   npz_tr.py
#   npz_tr_2.py
#   npz_tr_106.py
INCLUDE="${INCLUDE:-\.npz$|npz_tr(_[0-9]+)?\.py$}"

# ---------------------------------------------------------------------------
# Graph generation
#
# Everything is enabled intentionally for the final processed corpus.
# CSV files are always generated independently of these flags.
# ---------------------------------------------------------------------------

GRAPHS="${GRAPHS:-false}"                    # io_intensity.py: all graphs
COMPACT="${COMPACT:-false}"                  # compact run/worker graphs
INDIVIDUAL="${INDIVIDUAL:-false}"            # individual file/process/worker graphs
META_INDIVIDUAL="${META_INDIVIDUAL:-false}"  # individual metadata graphs too


analyze_run() {
    local run="$1"
    local logs="$run/darshan_logs"
    local out="$run/analysis"

    # -----------------------------------------------------------------------
    # 1. Convert all python3 Darshan logs to JSON.
    #
    # DataLoader workers are also python3 processes, so all matching python3
    # logs must be retained. c++ and lua5.3 logs are not part of the training
    # workload analyzed here.
    # -----------------------------------------------------------------------

    local f

    for f in "${pylogs[@]}"; do
        if [[ ! -f "$f.json" ]]; then
            python3 -m darshan to_json "$f" > "$f.json.tmp" || return 1
            mv "$f.json.tmp" "$f.json"
        fi
    done

    mkdir -p "$out"

    # -----------------------------------------------------------------------
    # 2. Normalize Darshan logs.
    #
    # simplified.json:
    #     NPZ dataset + training-script records only.
    #
    # simplified_all.json:
    #     all records from the python3 workload. This is used for metadata
    #     analysis because metadata activity extends beyond the dataset itself.
    # -----------------------------------------------------------------------

    python3 "$POST/simplify_logs.py" "$logs" \
        -i "$INCLUDE" \
        -o "$out/simplified.json" || return 1

    python3 "$POST/simplify_logs.py" "$logs" \
        -o "$out/simplified_all.json" || return 1

    # -----------------------------------------------------------------------
    # 3. I/O intensity.
    #
    # Writes:
    #   file_metrics.csv
    #   proc_metrics.csv
    #   run_metrics.csv
    #
    # Also generates file-, process-, and run-level graphs.
    # -----------------------------------------------------------------------

    python3 "$POST/io_intensity.py" \
        "$out/simplified.json" \
        -o "$out/io_intensity" \
        --graphs "$GRAPHS" || return 1

    # -----------------------------------------------------------------------
    # 4. Access pattern.
    #
    # Includes:
    #   consecutive reads
    #   sequential reads with gaps
    #   non-sequential reads
    #   seek activity
    #
    # Both compact and individual graphs are generated.
    # -----------------------------------------------------------------------

    python3 "$POST/access_pattern.py" \
        "$out/simplified.json" \
        -o "$out/access_pattern" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$INDIVIDUAL" || return 1

    # -----------------------------------------------------------------------
    # 5. Effective read bandwidth / throughput.
    #
    # Both compact and individual graphs are generated.
    # -----------------------------------------------------------------------

    python3 "$POST/effective_read.py" \
        "$out/simplified.json" \
        -o "$out/effective_read" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$INDIVIDUAL" || return 1

    # -----------------------------------------------------------------------
    # 6. Worker balance.
    #
    # NPZ uses forked DataLoader workers, so worker-level balance is an
    # important part of the analysis. Metrics include bytes read, read count,
    # read time, effective bandwidth, and run-level worker variability.
    #
    # Both compact and individual worker graphs are generated.
    # -----------------------------------------------------------------------

    python3 "$POST/worker_balance.py" \
        "$out/simplified.json" \
        -o "$out/worker_balance" \
        --compact-graphs "$COMPACT" \
        --individual-graphs "$INDIVIDUAL" || return 1

    # -----------------------------------------------------------------------
    # 7. Metadata pressure.
    #
    # Uses simplified_all.json intentionally so that metadata activity from
    # libraries, runtime files, /proc entries, caches, etc. is preserved.
    #
    # Both compact and individual graphs are generated intentionally.
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
        echo "--- no python3 log found, skipping: $name"
        skipped+=("$name")
        continue
    fi

    echo
    echo "======================================================================"
    echo "=== $name"
    echo "======================================================================"

    if analyze_run "$run"; then
        ok+=("$name")
    else
        echo "!!! failed: $name" >&2
        failed+=("$name")
    fi
done


echo
echo "======================================================================"
echo "NPZ CAMPAIGN SUMMARY"
echo "======================================================================"
echo "successful: ${#ok[@]}   skipped: ${#skipped[@]}   failed: ${#failed[@]}"

for n in "${skipped[@]}"; do
    echo "  skipped: $n"
done

for n in "${failed[@]}"; do
    echo "  failed: $n"
done
