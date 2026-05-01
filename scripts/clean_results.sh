#!/usr/bin/env bash
# Remove experiment outputs under results/ and figures/ (models, predictions,
# logs, tables, figures). Does not touch .venv or downloaded Fed-Heart-Disease data.
#
# Usage:
#   bash scripts/clean_results.sh           # delete artifacts
#   bash scripts/clean_results.sh --dry-run # print what would be removed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

rm_rf() {
  if [[ -e "$1" ]]; then
    if $DRY_RUN; then
      echo "[dry-run] rm -rf $1"
    else
      rm -rf -- "$1"
    fi
  fi
}

rm_f() {
  if [[ -f "$1" ]]; then
    if $DRY_RUN; then
      echo "[dry-run] rm -f $1"
    else
      rm -f -- "$1"
    fi
  fi
}

shopt -s nullglob

rm_f "${ROOT}/run_log.txt"

rm_rf "${ROOT}/results/models"
rm_rf "${ROOT}/results/predictions"

for f in "${ROOT}/results"/diagnostics_*.log; do
  rm_f "$f"
done
rm_f "${ROOT}/results/run_log.txt"
rm_f "${ROOT}/results/exp5_failures.txt"

for f in "${ROOT}/results"/exp*.json "${ROOT}/results"/exp*.csv "${ROOT}/results"/exp5_paper_*.tex; do
  rm_f "$f"
done

for f in "${ROOT}/figures"/*.png "${ROOT}/figures"/*.pdf; do
  rm_f "$f"
done

if $DRY_RUN; then
  echo "[dry-run] done (no files removed)."
else
  echo "Cleaned generated results and figures under ${ROOT}"
fi
