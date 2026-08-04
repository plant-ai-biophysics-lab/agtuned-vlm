#!/bin/bash
# Runs download_and_sample.py one active dataset at a time, each under an
# OS-level `timeout`, instead of one long process looping over everything.
#
# Why: on farm, individual HF Hub downloads have been observed to hang
# indefinitely with no exception raised and nothing printed — a known,
# widely-reported issue with HuggingFace's Xet transfer backend (see
# https://github.com/huggingface/datasets/issues/8129 and related), possibly
# compounded by farm's own outbound network path (this cluster has previously
# thrown "Network is unreachable" / "client has been closed" errors against
# huggingface.co — see HF_HUB_OFFLINE handling in experiments/verifier_cot/
# run_verifier_cot.py). Nothing in the underlying HF calls has a timeout, so
# a stall blocks the whole batch until someone notices and manually
# kills+restarts it. Wrapping each dataset individually converts that into
# "this one dataset times out after $TIMEOUT_SECONDS and the loop moves on,"
# unattended.
#
# Deliberately sequential, not parallel: if farm NATs compute-node egress
# through a shared IP, concurrent requests would only make HF-side rate
# limiting worse for everyone on the cluster, not just this job.
#
# Usage:
#   ./scripts/agml_hf/download_per_dataset.sh
#   TIMEOUT_SECONDS=600 ./scripts/agml_hf/download_per_dataset.sh   # override default
#
# Re-run freely — --skip_existing means already-completed datasets are
# skipped on every subsequent invocation, so a partial run + rerun (or a
# rerun after fixing/skipping a specific troublesome dataset in
# datasets.yaml) just picks up where it left off.

set -uo pipefail  # no -e: a per-dataset timeout/failure must not kill the loop

CONFIG="${CONFIG:-src/config/datasets.yaml}"
CATALOG="${CATALOG:-src/config/hf_catalog.json}"
N_PER_CLASS="${N_PER_CLASS:-50}"
OUTPUT="${OUTPUT:-dataset/selected_images.json}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

TIMEOUT_BIN="timeout"
if ! command -v timeout >/dev/null 2>&1; then
    if command -v gtimeout >/dev/null 2>&1; then
        TIMEOUT_BIN="gtimeout"  # macOS: brew install coreutils
    else
        echo "ERROR: no 'timeout' (or 'gtimeout') command found on this system." >&2
        exit 1
    fi
fi

DATASET_NAMES=()
while IFS= read -r ds; do
    DATASET_NAMES+=("$ds")
done < <(
    uv run python -c "
from src.common.dataset_config import active_datasets
for d in active_datasets('${CONFIG}'):
    print(d.agml_name)
"
)

echo "=== ${#DATASET_NAMES[@]} active datasets, ${TIMEOUT_SECONDS}s timeout each ==="

n_ok=0
n_timeout=0
n_fail=0
for ds in "${DATASET_NAMES[@]}"; do
    echo ""
    echo "--- ${ds} ---"
    "$TIMEOUT_BIN" "$TIMEOUT_SECONDS" uv run python scripts/agml_hf/download_and_sample.py \
        --config "$CONFIG" \
        --catalog "$CATALOG" \
        --n_per_class "$N_PER_CLASS" \
        --output "$OUTPUT" \
        --skip_existing \
        --no_charts \
        --only "^${ds}\$"
    status=$?
    if [ $status -eq 0 ]; then
        n_ok=$((n_ok + 1))
    elif [ $status -eq 124 ]; then
        echo "  TIMED OUT after ${TIMEOUT_SECONDS}s — moving on, retry later with --skip_existing"
        n_timeout=$((n_timeout + 1))
    else
        echo "  FAILED (exit ${status})"
        n_fail=$((n_fail + 1))
    fi
done

echo ""
echo "=== done: ${n_ok} ok, ${n_timeout} timed out, ${n_fail} failed ==="
if [ $((n_timeout + n_fail)) -gt 0 ]; then
    echo "Re-run this script (or download_and_sample.py --skip_existing directly)"
    echo "to retry only the datasets that didn't complete."
fi

# Charts weren't generated per-dataset (--no_charts above, to avoid redundant
# regeneration on every single-dataset invocation) — do it once at the end.
uv run python scripts/agml_hf/download_and_sample.py \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --plot_only
