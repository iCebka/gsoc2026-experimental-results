#!/usr/bin/env bash
# analyze_root.sh -- runs post-analysis for each run directory.
#
#   bash analyze_root.sh                      # default ROOT campaign
#   bash analyze_root.sh /another/campaign/runs   # any other runs directory
#   INCLUDE='\.h5$' bash analyze_root.sh ...  # another dataset filter
#   INDIVIDUAL=false bash analyze_root.sh     # compact graphs only
#
# Each run stores its results under <run>/analysis/.
# The script is safe to rerun: it only converts .darshan files that do not
# already have a corresponding .json file.

set -uo pipefail

POST=/pscratch/sd/s/satt/sprints/cli-test/ccode/darshanflow/post
# RUNS="${1:-/pscratch/sd/s/satt/sprints/a_star_campaign/root/runs}"
RUNS="${1:-/pscratch/sd/s/satt/sprints/final-lap/processed/root-compact}"
# Filter: dataset + training script. Adjust this based on the output of:
#   python3 -m darshan name_records <one_python3_log> | grep -Ei '\.root|_tr'
INCLUDE="${INCLUDE:-\.root$|_tr(_[0-9]+)?\.py$}"

# Graphs (true | false). By default all graph options are disabled in the
# underlying scripts, so they are explicitly enabled here.
GRAPHS="${GRAPHS:-true}"                    # io_intensity.py: --graphs
COMPACT="${COMPACT:-true}"                  # other scripts: --compact-graphs
INDIVIDUAL="${INDIVIDUAL:-false}"            # access_pattern, effective_read: --individual-graphs

# metadata_pressure reads the unfiltered JSON. Its individual graphs generate
# two PNGs for every file touched by the process (libraries, .pcm files,
# /proc entries, etc.), so they remain disabled unless explicitly requested.
META_INDIVIDUAL="${META_INDIVIDUAL:-false}"

analyze_run() {
    local run="$1"
    local logs="$run/darshan_logs"
    local out="$run/analysis"

    # 1. Convert only python3 logs.
    #    c++ and lua5.3 logs do not belong to the training workload and would
    #    otherwise appear as ghost runs with no relevant records.
    local f
    for f in "${pylogs[@]}"; do
        if [[ ! -f "$f.json" ]]; then
            python3 -m darshan to_json "$f" > "$f.json.tmp" || return 1
            mv "$f.json.tmp" "$f.json"
        fi
    done

    mkdir -p "$out"

    # 2. Simplify.
    #    simplified.json:     dataset + training script only
    #    simplified_all.json: unfiltered records, used for metadata pressure
    python3 "$POST/simplify_logs.py" "$logs" -i "$INCLUDE" -o "$out/simplified.json"      || return 1
    python3 "$POST/simplify_logs.py" "$logs"               -o "$out/simplified_all.json" || return 1

    # 3. Metrics.
    #    worker_balance is not run here: ROOT eager/stream/burst does not use
    #    forked DataLoader workers (ids id<n>-<n>), so there is no worker
    #    balance to analyze.
    python3 "$POST/io_intensity.py" "$out/simplified.json" -o "$out/io_intensity" \
        --graphs "$GRAPHS" || return 1

    python3 "$POST/access_pattern.py" "$out/simplified.json" -o "$out/access_pattern" \
        --compact-graphs "$COMPACT" --individual-graphs "$INDIVIDUAL" || return 1

    python3 "$POST/effective_read.py" "$out/simplified.json" -o "$out/effective_read" \
        --compact-graphs "$COMPACT" --individual-graphs "$INDIVIDUAL" || return 1

    python3 "$POST/metadata_pressure.py" "$out/simplified_all.json" -o "$out/metadata_pressure" \
        --compact-graphs "$COMPACT" --individual-graphs "$META_INDIVIDUAL" || return 1
}

ok=()
skipped=()
failed=()

for run in "$RUNS"/*/; do
    run="${run%/}"
    name="$(basename "$run")"

    [[ -d "$run/darshan_logs" ]] || continue

    mapfile -t pylogs < <(
        find "$run/darshan_logs" -name 'satt_python3_*.darshan' | sort
    )

    if (( ${#pylogs[@]} == 0 )); then
        echo "--- no python3 log found, skipping: $name"
        skipped+=("$name")
        continue
    fi

    echo "=== $name"

    if analyze_run "$run"; then
        ok+=("$name")
    else
        echo "!!! failed: $name" >&2
        failed+=("$name")
    fi
done

echo
echo "successful: ${#ok[@]}   skipped: ${#skipped[@]}   failed: ${#failed[@]}"

for n in "${skipped[@]}"; do
    echo "  skipped: $n"
done

for n in "${failed[@]}"; do
    echo "  failed: $n"
done
