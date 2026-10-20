#!/usr/bin/env bash
# rerun_forward_check_tight.sh
# Repeat the forward check of the hp-VPINN final runs with a tighter shooting
# tolerance of the exact solver (--itol 1e-5 instead of frontx's default 1e-3),
# WITHOUT overwriting the results already obtained.
#
# Needs frontx (see requirements-conventional.txt). Run with the environment active:
#     bash scripts/rerun_forward_check_tight.sh            # all four tags
#     DRY=1 bash scripts/rerun_forward_check_tight.sh      # only print the solver commands
#     CASES="vangenuchten" bash scripts/rerun_forward_check_tight.sh   # only Case 2
#
# What it does
#   0. Renames the existing results of the default-tolerance run:
#        results/conventional/forward_check_<case>[_kpct_noise].json      -> ..._itol1e-3.json
#        results/conventional/forward_check_summary_<tag>.csv             -> ..._itol1e-3.csv
#      (skipped for files already renamed; it never overwrites anything).
#   1. Runs conventional.benchmark --forward-check ... --itol 1e-5 and moves each
#      output to ..._itol1e-5.json (the driver always writes the unsuffixed name).
#   2. Summarises each tolerance separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root

ITOL=1e-5
NEW=_itol1e-5
OLD=_itol1e-3
DRY=${DRY:-0}
CASES=${CASES:-"grenoblesand1 grenoblesand2 vangenuchten letd"}
export JAX_PLATFORMS=cpu

run() { if [[ "$DRY" == 1 ]]; then echo "[dry] $*"; else "$@"; fi; }
mvsafe() { # mv without ever overwriting
  if [[ -e "$2" ]]; then echo "  refusing to overwrite $2 (keeping $1)"; else mv "$1" "$2"; echo "  $1 -> $2"; fi
}

# tag | driver args | forward-check file stem | input jsonl
declare -A ARGS=( [grenoblesand1]="--case grenoblesand --noise-level 0.01"
                  [grenoblesand2]="--case grenoblesand --noise-level 0.02"
                  [vangenuchten]="--case vangenuchten"
                  [letd]="--case letd" )
declare -A STEM=( [grenoblesand1]=grenoblesand_1pct_noise [grenoblesand2]=grenoblesand_2pct_noise
                  [vangenuchten]=vangenuchten [letd]=letd )
declare -A TAG=(  [grenoblesand1]=grenoblesand_gb_1pct_noise [grenoblesand2]=grenoblesand_gb_2pct_noise
                  [vangenuchten]=vangenuchten_es [letd]=letd_gb )

[[ -f pyproject.toml ]] || { echo "run this from the repo root"; exit 1; }
for k in $CASES; do
  [[ -f results/conventional/forward_check_inputs/hpvpinn_${TAG[$k]}.jsonl ]] || \
    { echo "missing results/conventional/forward_check_inputs/hpvpinn_${TAG[$k]}.jsonl: run  python3 -m hpvpinn_capillarity.conventional.forward_check_bridge  first"; exit 1; }
done

echo "== 0. keep the default-tolerance results =="
for k in $CASES; do
  f=results/conventional/forward_check_${STEM[$k]}.json
  [[ -f $f ]] && mvsafe "$f" "${f%.json}${OLD}.json"
  c=results/conventional/forward_check_summary_${TAG[$k]}.csv
  [[ -f $c ]] && mvsafe "$c" "${c%.csv}${OLD}.csv"
done

echo "== 1. forward check with --itol $ITOL =="
for k in $CASES; do
  echo "-- ${TAG[$k]}"
  # shellcheck disable=SC2086
  run python3 -m hpvpinn_capillarity.conventional.benchmark ${ARGS[$k]} --itol $ITOL \
      --forward-check results/conventional/forward_check_inputs/hpvpinn_${TAG[$k]}.jsonl
  f=results/conventional/forward_check_${STEM[$k]}.json
  if [[ "$DRY" != 1 ]]; then mvsafe "$f" "${f%.json}${NEW}.json"; fi
done

echo "== 2. summaries =="
if [[ "$DRY" != 1 ]]; then
  python3 -m hpvpinn_capillarity.conventional.forward_check_bridge --summarise --suffix $OLD
  python3 -m hpvpinn_capillarity.conventional.forward_check_bridge --summarise --suffix $NEW
fi
echo
echo "Send back: results/conventional/forward_check_*${NEW}.json and results/conventional/forward_check_summary_*${NEW}.csv"
