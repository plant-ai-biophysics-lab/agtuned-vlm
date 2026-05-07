#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_scraper.sh
# Bash wrapper for scrape_idtools.py
#
# Usage:
#   bash run_scraper.sh [mode] [extra args]
#
# Modes:
#   discover    — find URLs only, no inference
#   run         — full pipeline (default)
#   resume      — alias for run (safe to call anytime, skips done URLs)
#   convert     — convert existing JSONL to Unsloth SFT format, no scraping
#   status      — show database progress summary
#   help        — print this message
#
# Examples:
#   bash run_scraper.sh discover
#   bash run_scraper.sh run
#   bash run_scraper.sh run --batch-size 4 --max-model-len 2048
#   bash run_scraper.sh resume                    # safe to run after interruption
#   bash run_scraper.sh convert
#   bash run_scraper.sh status
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration — edit these to match your environment ─────────────────────

MODEL="Qwen/Qwen3.6-35B-A3B"
DB="idtools_scraper.db"
OUTPUT="idtools_qa_dataset.jsonl"
LOG_FILE="scraper.log"
LOG_LEVEL="INFO"

# vLLM settings for 1x A100 80GB
# Qwen3.6-35B-A3B at bf16 ≈ 70GB weights.
# 0.90 * 80GB = 72GB for weights + KV cache.
# If you hit OOM: lower GPU_MEM to 0.88 and MAX_MODEL_LEN to 2048.
GPU_MEM=0.90
MAX_MODEL_LEN=4096

# QA generation settings
MAX_PAIRS=5          # QA pairs per fact sheet
BATCH_SIZE=8         # Inference batch size — lower if GPU runs tight
MAX_DISCOVER=1500    # Max fact sheet URLs to discover
SCRAPE_DELAY=1.5     # Seconds between HTTP requests (be polite to idtools.org)

# Python environment — update if you use conda or a different venv path
PYTHON="${PYTHON:-python3}"

# Optional: activate your conda/venv here
# source activate aggemma
# source /path/to/venv/bin/activate

# ─────────────────────────────────────────────────────────────────────────────

MODE="${1:-run}"
shift || true           # remaining args passed through to Python script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/scrape_idtools.py"

# ── Dependency check ──────────────────────────────────────────────────────────
check_deps() {
    local missing=()
    for pkg in vllm transformers bs4 requests psutil; do
        if ! $PYTHON -c "import $pkg" &>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "❌  Missing Python packages: ${missing[*]}"
        echo "    Install with:"
        echo "    pip install vllm transformers beautifulsoup4 requests psutil"
        exit 1
    fi
    echo "✓ All dependencies found"
}

# ── GPU check ─────────────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo "── GPU Status ────────────────────────────────────"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
                   --format=csv,noheader,nounits \
        | awk -F',' '{
            printf "  GPU:   %s\n", $1
            printf "  VRAM:  %s MB total | %s MB used | %s MB free\n", $2, $3, $4
            printf "  Usage: %s%%\n", $5
          }'
        echo "──────────────────────────────────────────────────"
        echo ""
    fi
}

# ── Database status ───────────────────────────────────────────────────────────
show_status() {
    if [[ ! -f "$DB" ]]; then
        echo "No database found at $DB — run 'bash run_scraper.sh discover' first."
        return
    fi
    echo ""
    echo "── Database: $DB ────────────────────────────────"
    sqlite3 "$DB" \
        "SELECT status, COUNT(*) as count FROM urls GROUP BY status ORDER BY count DESC;" \
        | column -t -s '|' | awk '{print "  " $0}'
    echo ""
    QA_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM qa_pairs;" 2>/dev/null || echo 0)
    echo "  Total QA pairs generated : $QA_COUNT"
    if [[ -f "$OUTPUT" ]]; then
        LINE_COUNT=$(wc -l < "$OUTPUT")
        echo "  JSONL output lines       : $LINE_COUNT  ($OUTPUT)"
    fi
    echo "──────────────────────────────────────────────────"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       IDtools Agricultural Visual QA Generator               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Mode    : $MODE"
echo "  Model   : $MODEL"
echo "  DB      : $DB"
echo "  Output  : $OUTPUT"
echo "  GPU mem : $GPU_MEM"
echo ""

case "$MODE" in

  # ── Help ───────────────────────────────────────────────────────────────────
  help|--help|-h)
    sed -n '2,30p' "${BASH_SOURCE[0]}"
    exit 0
    ;;

  # ── Status only ────────────────────────────────────────────────────────────
  status)
    show_status
    exit 0
    ;;

  # ── Convert existing output to SFT format ─────────────────────────────────
  convert)
    check_deps
    if [[ ! -f "$OUTPUT" ]]; then
        echo "❌  Output file not found: $OUTPUT"
        echo "    Run the scraper first before converting."
        exit 1
    fi
    SFT_OUTPUT="${OUTPUT%.jsonl}_sft.jsonl"
    echo "Converting $OUTPUT → $SFT_OUTPUT ..."
    $PYTHON - <<EOF
import sys
sys.path.insert(0, "$SCRIPT_DIR")
from scrape_idtools import convert_to_sft_format, setup_logging
setup_logging("INFO", "$LOG_FILE")
convert_to_sft_format("$OUTPUT", "$SFT_OUTPUT")
EOF
    echo "✓ SFT conversion complete: $SFT_OUTPUT"
    exit 0
    ;;

  # ── Discover URLs only ─────────────────────────────────────────────────────
  discover)
    check_deps
    check_gpu
    echo "Running URL discovery only (no inference) ..."
    $PYTHON "$SCRIPT" \
        --model        "$MODEL"        \
        --db           "$DB"           \
        --output       "$OUTPUT"       \
        --log-level    "$LOG_LEVEL"    \
        --log-file     "$LOG_FILE"     \
        --max-discover "$MAX_DISCOVER" \
        --scrape-delay "$SCRAPE_DELAY" \
        --discover-only \
        "$@"
    echo ""
    show_status
    ;;

  # ── Full run / resume ──────────────────────────────────────────────────────
  run|resume)
    check_deps
    check_gpu
    show_status

    echo "Starting scraper (safe to Ctrl+C and resume anytime) ..."
    echo "Logs → $LOG_FILE"
    echo ""

    $PYTHON "$SCRIPT" \
        --model          "$MODEL"        \
        --db             "$DB"           \
        --output         "$OUTPUT"       \
        --log-level      "$LOG_LEVEL"    \
        --log-file       "$LOG_FILE"     \
        --max-pairs      "$MAX_PAIRS"    \
        --batch-size     "$BATCH_SIZE"   \
        --max-model-len  "$MAX_MODEL_LEN" \
        --gpu-mem        "$GPU_MEM"      \
        --max-discover   "$MAX_DISCOVER" \
        --scrape-delay   "$SCRAPE_DELAY" \
        --convert-sft \
        "$@"

    echo ""
    show_status
    echo "✓ Done. Output: $OUTPUT"
    ;;

  *)
    echo "❌  Unknown mode: $MODE"
    echo "    Valid modes: discover | run | resume | convert | status | help"
    exit 1
    ;;
esac
