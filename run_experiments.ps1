#
# run_experiments.ps1 — Reproduce CAMSAP 2025 results both with and without
# the known legacy FAISS Hellinger conversion bug.
#
# Run from the MSTML repo root:
#   .\run_experiments.ps1
#
# The script runs the full pipeline once with --reproduce_legacy_bug (exact
# reproduction of thesis/CAMSAP 2025 figures), then shares the expensive
# preprocessing and ensemble artefacts with the second run, which re-runs
# only the bug-affected stages 3-6 (distributions, manifold, scoring, viz)
# without the flag, producing corrected results.
#
# Data file:
#   data\arxiv\original\arxiv-metadata-oai-snapshot-atoms-lp.json
#   (the 3.79 GB snapshot used for the original AToMS-LP / CAMSAP 2025 runs)
#
# Output directories:
#   experiments\arxiv\camsap2025_with_bug\   (legacy / exact replication)
#   experiments\arxiv\camsap2025_fixed\      (corrected Hellinger distances)

$ErrorActionPreference = "Stop"
$env:VIRTUAL_ENV = ""   # prevent uv warning when a stale venv is activated

$ROOT    = $PSScriptRoot
$DATA    = "$ROOT\data\arxiv\original\arxiv-metadata-oai-snapshot-atoms-lp.json"
$OUT_BUG = "$ROOT\experiments\arxiv\camsap2025_with_bug"
$OUT_FIX = "$ROOT\experiments\arxiv\camsap2025_fixed"

# Create output directories so Tee-Object can open the log files immediately
New-Item -ItemType Directory -Force $OUT_BUG | Out-Null
New-Item -ItemType Directory -Force $OUT_FIX | Out-Null

# Common CLI arguments (primary hyperparameters matching CAMSAP 2025)
$COMMON = @(
    "--input_file",  $DATA,
    "--categories",  "cs.LG", "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH",
    "--year_start",  "2012",
    "--year_end",    "2023",
    "--distance_metric",  "hellinger",
    "--embedding_method", "phate",
    "--linkage_method",   "ward",
    "--cut_height",       "0.68",
    "--log_level",        "INFO"
)

# ---------------------------------------------------------------------------
# RUN 1 — with legacy bug (exact CAMSAP 2025 / thesis replication)
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "==========================================================="
Write-Host " RUN 1/2  [with legacy bug]  ->  $OUT_BUG"
Write-Host "==========================================================="
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

uv run python -m legacy.run_pipeline @COMMON --output_dir $OUT_BUG --reproduce_legacy_bug

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: RUN 1 failed (exit code $LASTEXITCODE). Aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "RUN 1 complete: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ---------------------------------------------------------------------------
# Share preprocessing + ensemble artefacts with the fixed run
#   Stages 1-2 produce identical output regardless of the bug flag.
#   The bug only affects stages 3-6 (distributions, manifold, scoring, viz).
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "Copying stages 1-2 artefacts to fixed-run directory ..."

$SHARED = @(
    "main_df.pkl",
    "id2word.pkl",
    "name_to_id.pkl",
    "id_to_names.pkl",
    "topic_vectors.pkl",
    "ntopics_by_chunk.pkl",
    "inds_by_chunk.pkl",
    "doc_topic_distns.pkl",
    "expanded_doc_topic_distns.pkl"
)
foreach ($f in $SHARED) {
    $src = "$OUT_BUG\$f"
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination "$OUT_FIX\$f" -Force
        Write-Host "  Copied: $f"
    } else {
        Write-Warning "  Not found (skipping): $f"
    }
}

# ---------------------------------------------------------------------------
# RUN 2 — without bug (corrected Hellinger distances)
#   Skip preprocessing and ensemble (stages 1-2) since we copied the artefacts.
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "==========================================================="
Write-Host " RUN 2/2  [fixed / corrected]  ->  $OUT_FIX"
Write-Host "==========================================================="
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

uv run python -m legacy.run_pipeline @COMMON --output_dir $OUT_FIX --skip_preprocess --skip_ensemble

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: RUN 2 failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "RUN 2 complete: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "==========================================================="
Write-Host " ALL RUNS COMPLETE"
Write-Host "  With bug : $OUT_BUG"
Write-Host "  Fixed    : $OUT_FIX"
Write-Host "==========================================================="
