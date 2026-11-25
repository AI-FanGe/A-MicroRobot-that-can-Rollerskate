#!/usr/bin/env bash
# Skate checkpoints at full and low supply voltage: skating at 0.3 m/s (7.5 V, 5.5 V) and
# standing (5.5 V), one summary line each.
# usage: scripts/skate_vin_sweep.sh <ckpt> [<ckpt> ...]
# env:   NUM_ENVS (512), RUNS="<vin> <vx>,..." ("7.5 0.3,5.5 0.3,5.5 0"),
#        TASK (Mjlab-Skate-Xiaoxiaobai, with XIAOXIAOBAI_SKATE_BASE=s5 unless set)
cd "$(dirname "$0")/.." || exit 1
export XIAOXIAOBAI_BASE=24996 NUM_ENVS=${NUM_ENVS:-512}
export TASK=${TASK:-Mjlab-Skate-Xiaoxiaobai}
[ "$TASK" = Mjlab-Skate-Xiaoxiaobai ] && export XIAOXIAOBAI_SKATE_BASE=${XIAOXIAOBAI_SKATE_BASE:-s5}
IFS=, read -ra runs <<< "${RUNS:-7.5 0.3,5.5 0.3,5.5 0}"
for ckpt in "$@"; do
    name=$(basename "$ckpt")
    for run in "${runs[@]}"; do
        set -- $run
        (VIN=$1 VX=$2 .venv/bin/python scripts/skate_eval.py "$ckpt" 2>&1 \
            | grep -aE "^  (hip roll|all)" | sed -E 's/ +n= *[0-9]+//; s/\| one skate.*touchdowns/| touchdowns/; s/ none .*side slip/ | side slip/' \
            | paste -sd' ' | sed "s|^|$name ${1} V vx ${2}: |")
    done
done
