#!/usr/bin/env bash
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# NOTE: This script is only there for failsafe. The main runner is run_pipeline.sh
# NOTE: Unexpected behavior may occur if you run this
set -euo pipefail

############################################
# Args: --start-from N --end-at N --reuse-run RUN_ID_OR_PATH
############################################
START_FROM=1
END_AT=999
REUSE_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-from)
      START_FROM="${2:-1}"; shift 2 ;;
    --end-at)
      END_AT="${2:-999}"; shift 2 ;;
    --reuse-run)
      REUSE_RUN="${2:-}"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

############################################
# Config
############################################

# Toggle transformer fine tuning
RUN_TRANSFORMER="${RUN_TRANSFORMER:-true}"

# Datasets
BASELINE_CSV="data/baseline_spam-ham.csv"
SA_CSV="data/spam_assassin_cleaned.csv"
ZENODO_CSV="data/zenodo_cleaned.csv"
ENRON_CSV="data/enron_emails.csv"  # optional holdout (ham-heavy)

# One demo text for LIME
SAMPLE_TEXT="Your account requires a brief settings check. Please sign in to continue using secure session."

# Transformer config
TRANSFORMER_MODEL="xlm-roberta-large"
TRANSFORMER_EPOCHS=3
TRANSFORMER_MAXLEN=256
TRANSFORMER_BATCH_SIZE=4
TRANSFORMER_GRAD_ACCUM_STEPS=4
TRANSFORMER_WARMUP_RATIO=0.06

############################################
# Repo root / env
############################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT=""
for d in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
  if [ -d "$d/src" ]; then REPO_ROOT="$(cd "$d" && pwd)"; break; fi
done
[ -z "$REPO_ROOT" ] && echo "Error: Could not find repository root." && exit 1
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

############################################
# Output dirs (support reuse)
############################################
if [ -n "$REUSE_RUN" ]; then
  # Accept either an ID or a full path
  if [ -d "$REUSE_RUN" ]; then
    RUN_DIR="$REUSE_RUN"
  else
    RUN_DIR="runs/$REUSE_RUN"
  fi
  if [ ! -d "$RUN_DIR" ]; then
    echo "Error: --reuse-run points to missing directory: $RUN_DIR"
    exit 1
  fi
  RUN_ID="$(basename "$RUN_DIR")"
  ART_DIR="$RUN_DIR/artifacts"
  LOG_DIR="$RUN_DIR/logs"
  mkdir -p "$ART_DIR" "$LOG_DIR"
else
  RUN_ID="$(date +%Y%m%d-%H%M%S)"
  RUN_DIR="runs/run_${RUN_ID}"
  ART_DIR="$RUN_DIR/artifacts"
  LOG_DIR="$RUN_DIR/logs"
  mkdir -p "$ART_DIR" "$LOG_DIR" "models" "feedback"
fi

echo "=== Starting pipeline run ==="
echo "REPO:    $REPO_ROOT"
echo "RUN ID:  $RUN_ID"
echo "RUN DIR: $RUN_DIR"
echo "ARGS:    --start-from $START_FROM  --end-at $END_AT  --reuse-run ${REUSE_RUN:-<none>}"
echo

############################################
# Resolve datasets
############################################
DATASETS=()
for f in "$BASELINE_CSV" "$SA_CSV" "$ZENODO_CSV"; do
  [ -f "$f" ] && DATASETS+=("$f")
done
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "No training datasets found. Checked:"
  echo " - $BASELINE_CSV"
  echo " - $SA_CSV"
  echo " - $ZENODO_CSV"
  exit 1
fi
if [ ! -f "$ENRON_CSV" ]; then
  echo "Enron holdout not found ($ENRON_CSV). Skipping cross-domain test."
  ENRON_CSV=""
fi

echo "[DATASETS]"
printf ' - %s\n' "${DATASETS[@]}"
[ -n "$ENRON_CSV" ] && echo " - (holdout) $ENRON_CSV"
echo

############################################
# Model sweeps
############################################
# Names must match src/models/baselines.py AVAILABLE
TFIDF_MODELS=(
  logistic_regression
  ridge_classifier
  sgd_log
  sgd_hinge
  complement_nb
  bernoulli_nb
  decision_tree
  dummy_mf
)

SBERT_MODELS=(
  logistic_regression
  ridge_classifier
  sgd_log
)

m_out() { echo "runs/$RUN_ID/$1"; }

############################################
# 1) Train TF-IDF-only models
############################################
if [ "$START_FROM" -le 1 ] && [ "$END_AT" -ge 1 ]; then
  echo "[1/5] Training TF-IDF models…"
  for model in "${TFIDF_MODELS[@]}"; do
    TAG="tfidf_${model}"
    OUT_TAG="$(m_out "$TAG")"
    LOG="$LOG_DIR/train_${TAG}.log"
    echo " -> $model"
    python3 -m src.models.train \
      --datasets "${DATASETS[@]}" \
      --model "$model" \
      --use_tfidf --no_sbert --no_fc \
      --out "$OUT_TAG" | tee "$LOG" >/dev/null
  done
  echo "TF-IDF models trained."
  echo
else
  echo "[1/5] Skipping TF-IDF training (start-from/end-at gates)."
fi

############################################
# 2) Train SBERT(+FactChecker) models
############################################
if [ "$START_FROM" -le 2 ] && [ "$END_AT" -ge 2 ]; then
  echo "[2/5] Training SBERT(+FactChecker) models…"
  for model in "${SBERT_MODELS[@]}"; do
    TAG="sbertfc_${model}"
    OUT_TAG="$(m_out "$TAG")"
    LOG="$LOG_DIR/train_${TAG}.log"
    echo " -> $model"
    python3 -m src.models.train \
      --datasets "${DATASETS[@]}" \
      --model "$model" \
      --out "$OUT_TAG" | tee "$LOG" >/dev/null
  done
  echo "SBERT(+FC) models trained."
  echo
else
  echo "[2/5] Skipping SBERT(+FC) training (start-from/end-at gates)."
fi

############################################
# 3) Optional: fine-tune transformer
############################################
TRANS_DIR=""
if [ "$START_FROM" -le 3 ] && [ "$END_AT" -ge 3 ]; then
  if [ "$RUN_TRANSFORMER" = "true" ]; then
    echo "[3/5] Fine-tuning transformer ($TRANSFORMER_MODEL)…"
    TRANS_OUT="runs/$RUN_ID/transformer_${TRANSFORMER_MODEL//\//_}"
    python3 -m src.models.transformer_train \
      --datasets "${DATASETS[@]}" \
      --model_name "$TRANSFORMER_MODEL" \
      --out_dir "models/$TRANS_OUT" \
      --epochs "$TRANSFORMER_EPOCHS" \
      --max_len "$TRANSFORMER_MAXLEN" \
      --batch_size "$TRANSFORMER_BATCH_SIZE" \
      --grad_accum_steps "$TRANSFORMER_GRAD_ACCUM_STEPS" \
      --warmup_ratio "$TRANSFORMER_WARMUP_RATIO" | tee "$LOG_DIR/transformer_train.log" >/dev/null
    TRANS_DIR="models/$TRANS_OUT"
    echo "✔ Transformer saved at: $TRANS_DIR"
    echo
  else
    echo "[3/5] Skipping transformer (set RUN_TRANSFORMER=true to enable)."
    echo
  fi
else
  echo "[3/5] Skipping step 3 (start-from/end-at gates)."
fi

############################################
# 4) Predict on Enron (for every trained model)
############################################
if [ "$START_FROM" -le 4 ] && [ "$END_AT" -ge 4 ]; then
  echo "[4/5] Enron predictions…"
  if [ -n "$ENRON_CSV" ]; then
    mkdir -p "$ART_DIR/preds"
    MODELS_ROOT="models/runs/$RUN_ID"
    if [ ! -d "$MODELS_ROOT" ]; then
      echo "Error: expected trained models under $MODELS_ROOT"
      echo "Hint: pass --reuse-run <run_id_or_path> if you want to reuse a previous run."
      exit 1
    fi

    # loop through every trained model dir under models/runs/<RUN_ID>/*
    while IFS= read -r -d '' dir; do
      NAME="$(basename "$dir")"
      OUT_CSV="$ART_DIR/preds/enron_${NAME}.csv"
      LOG="$LOG_DIR/predict_enron_${NAME}.log"
      echo " -> $NAME"
      python3 -m src.models.predict \
        --model_dir "$dir" \
        --input_csv "$ENRON_CSV" \
        --out_csv "$OUT_CSV" | tee "$LOG" >/dev/null
    done < <(find "$MODELS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0)

    if [ -n "$TRANS_DIR" ]; then
      echo "NOTE: transformer inference on Enron not wired into this script yet."
    fi
  else
    echo "Skipped (no Enron holdout)."
  fi
  echo
else
  echo "[4/5] Skipping step 4 (start-from/end-at gates)."
fi

############################################
# 5) Explainability artifacts (one per family)
############################################
if [ "$START_FROM" -le 5 ] && [ "$END_AT" -ge 5 ]; then
  echo "[5/5] Explainability artifacts…"
  mkdir -p "$ART_DIR/explain"

  # pick the first TF-IDF model dir
  TFIDF_DIR="$(find "models/runs/$RUN_ID" -maxdepth 1 -type d -name 'tfidf_ridge_classifier*' | head -n 1 || true)"
[ -z "$TFIDF_DIR" ] && TFIDF_DIR="$(find "models/runs/$RUN_ID" -maxdepth 1 -type d -name 'tfidf_logistic_regression*' | head -n 1 || true)"
  if [ -n "$TFIDF_DIR" ]; then
    python3 -m src.explain.explain \
      --model_dir "$TFIDF_DIR" \
      --out_dir "$ART_DIR/explain/tfidf" \
      --text "$SAMPLE_TEXT" \
      --sample_csv "${DATASETS[0]}" \
      --sample_size 150 \
      --save_global_shap | tee "$LOG_DIR/explain_tfidf.log" >/dev/null
  fi

  # pick the first SBERT(+FC) model dir
  SBERT_DIR="$(find "models/runs/$RUN_ID" -maxdepth 1 -type d -name 'sbertfc_logistic_regression*' | head -n 1 || true)"
[ -z "$SBERT_DIR" ] && SBERT_DIR="$(find "models/runs/$RUN_ID" -maxdepth 1 -type d -name 'sbertfc_ridge_classifier*' | head -n 1 || true)"
  if [ -n "$SBERT_DIR" ]; then
    python3 -m src.explain.explain \
      --model_dir "$SBERT_DIR" \
      --out_dir "$ART_DIR/explain/sbertfc" \
      --text "$SAMPLE_TEXT" \
      --sample_csv "${DATASETS[0]}" \
      --sample_size 150 \
      --save_global_shap | tee "$LOG_DIR/explain_sbertfc.log" >/dev/null
  fi

  echo
  echo "=== ✅ Pipeline finished ==="

  echo "Summary CSV: $RUN_DIR/artifacts/summary/metrics_summary.csv"
  echo "Artifacts:   $RUN_DIR"
else
  echo "[5/5] Skipping step 5 (start-from/end-at gates)."
fi
