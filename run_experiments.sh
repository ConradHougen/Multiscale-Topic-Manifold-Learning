#!/usr/bin/env bash
#
# run_experiments.sh — Reproduce CAMSAP 2025 results both with and without
# the known legacy FAISS Hellinger conversion bug.
#
# Run from the MSTML repo root:
#   bash run_experiments.sh
#
# The script runs the full pipeline once with --reproduce_legacy_bug (exact
# reproduction of thesis/CAMSAP 2025 figures), then shares the expensive
# preprocessing and ensemble artefacts with the second run, which re-runs
# only the bug-affected stages 3-6 (distributions, manifold, scoring, viz)
# without the flag, producing corrected results.
#
# Data file:
#   data/arxiv/original/arxiv-metadata-oai-snapshot-atoms-lp.json
#   (the 3.79 GB snapshot used for the original AToMS-LP / CAMSAP 2025 runs)
#
# Output directories:
#   experiments/arxiv/camsap2025_with_bug/   (legacy / exact replication)
#   experiments/arxiv/camsap2025_fixed/      (corrected Hellinger distances)

set -euo pipefail
trap 'echo "ERROR: pipeline failed (line $LINENO). Aborting." >&2; exit 1' ERR

unset VIRTUAL_ENV   # prevent uv warning when a stale venv is set

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="$SCRIPT_DIR/data/arxiv/original/arxiv-metadata-oai-snapshot-atoms-lp.json"
OUT_BUG="$SCRIPT_DIR/experiments/arxiv/camsap2025_with_bug"
OUT_FIX="$SCRIPT_DIR/experiments/arxiv/camsap2025_fixed"

mkdir -p "$OUT_BUG"
mkdir -p "$OUT_FIX"

# Common CLI arguments (primary hyperparameters matching CAMSAP 2025)
COMMON=(
    --input_file  "$DATA"
    --categories  cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH
    --year_start  2012
    --year_end    2023
    --distance_metric  hellinger
    --embedding_method phate
    --linkage_method   ward
    --cut_height       0.68
    --log_level        INFO
)

# ---------------------------------------------------------------------------
# RUN 1 — with legacy bug (exact CAMSAP 2025 / thesis replication)
# ---------------------------------------------------------------------------

echo ""
echo "==========================================================="
echo " RUN 1/2  [with legacy bug]  ->  $OUT_BUG"
echo "==========================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

uv run python -m legacy.run_pipeline \
    "${COMMON[@]}" \
    --output_dir "$OUT_BUG" \
    --reproduce_legacy_bug \
    | tee "$OUT_BUG/pipeline_run.log"

echo ""
echo "RUN 1 complete: $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Share preprocessing + ensemble artefacts with the fixed run
#   Stages 1-2 produce identical output regardless of the bug flag.
#   The bug only affects stages 3-6 (distributions, manifold, scoring, viz).
# ---------------------------------------------------------------------------

echo ""
echo "Copying stages 1-2 artefacts to fixed-run directory ..."

SHARED=(
    main_df.pkl
    id2word.pkl
    name_to_id.pkl
    id_to_names.pkl
    topic_vectors.pkl
    ntopics_by_chunk.pkl
    inds_by_chunk.pkl
    doc_topic_distns.pkl
    expanded_doc_topic_distns.pkl
)

for f in "${SHARED[@]}"; do
    src="$OUT_BUG/$f"
    if [ -f "$src" ]; then
        cp "$src" "$OUT_FIX/$f"
        echo "  Copied: $f"
    else
        echo "  WARNING: Not found (skipping): $f" >&2
    fi
done

# ---------------------------------------------------------------------------
# RUN 2 — without bug (corrected Hellinger distances)
#   Skip preprocessing and ensemble (stages 1-2) since we copied the artefacts.
# ---------------------------------------------------------------------------

echo ""
echo "==========================================================="
echo " RUN 2/2  [fixed / corrected]  ->  $OUT_FIX"
echo "==========================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

uv run python -m legacy.run_pipeline \
    "${COMMON[@]}" \
    --output_dir "$OUT_FIX" \
    --skip_preprocess \
    --skip_ensemble \
    | tee "$OUT_FIX/pipeline_run.log"

echo ""
echo "RUN 2 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "==========================================================="
echo " ALL RUNS COMPLETE"
echo "  With bug : $OUT_BUG"
echo "  Fixed    : $OUT_FIX"
echo "==========================================================="
