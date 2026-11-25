#!/bin/bash
# Max-command skate eval of one checkpoint over a set of floors.
# usage: eval_sweep.sh <ckpt> ["CRR=.. BEARING=.." ...]
set -uo pipefail
cd "$(dirname "$0")/.."
export XIAOXIAOBAI_BASE=24996 VX="${VX:-0.5}" NUM_ENVS="${NUM_ENVS:-256}" DURATION="${DURATION:-8}"
ckpt=$1
shift
specs=("$@")
if [ ${#specs[@]} -eq 0 ]; then
  specs=("CRR=0.01 BEARING=1e-4" "CRR=0.08 BEARING=1e-3" "CRR=0.15 BEARING=1e-3" "CRR=0.25 BEARING=1e-3" "CRR=0.35 BEARING=2e-3")
fi
echo "== $ckpt"
for spec in "${specs[@]}"; do
  echo "-- $spec"
  env $spec .venv/bin/python scripts/skate_eval.py "$ckpt" 2>&1 | grep -E "hip yaw|hip roll|  all"
done
