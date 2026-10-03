#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="$(pwd)"

echo "Working in: $BASE_DIR"
echo

echo "=== Removing analysis directories ==="

find "$BASE_DIR" \
    -mindepth 2 \
    -type d \
    -name "analysis" \
    -print \
    -exec rm -rf {} +

echo
echo "=== Removing JSON files under darshan_logs ==="

find "$BASE_DIR" \
    -type d \
    -name "darshan_logs" \
    -print0 |
while IFS= read -r -d '' darshan_dir; do
    echo "Scanning: $darshan_dir"

    find "$darshan_dir" \
        -type f \
        -name "*.json" \
        -print \
        -delete
done

echo
echo "Done."

