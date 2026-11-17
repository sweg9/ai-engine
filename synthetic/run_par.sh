#!/bin/zsh
# One of several render loops run side by side: seeds START, START+STEP, ... until TOTAL labels.
#   synthetic/run_par.sh 21 3 & synthetic/run_par.sh 22 3 & synthetic/run_par.sh 23 3
# OUT (default ../synthetic/evidence from the repo root), BLENDER and TOTAL can be set in the environment.
cd "$(dirname "$0")/.."
OUT=${OUT:-../synthetic/evidence}; TOTAL=${TOTAL:-3000}
BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
mkdir -p "$OUT"
s=$1; step=$2
while [ $(find "$OUT" -name labels.jsonl -exec cat {} + | wc -l) -lt $TOTAL ]; do
  nice -n 19 "$BLENDER" -b --factory-startup -P synthetic/synth_evidence.py -- --out "$OUT/syn-e$s" --scenes 40 --seed $s > "$OUT/log_e$s.txt" 2>&1
  echo "syn-e$s: $(tail -1 "$OUT/log_e$s.txt") / total $(find "$OUT" -name labels.jsonl -exec cat {} + | wc -l)"
  s=$((s + step))
done
