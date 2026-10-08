#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

n_iters=40
stamp=${2:-$(date +%Y%m%d-%H%M%S)}

for config in config_E.jl config_Erho.jl config_Erhotau.jl; do
    variant=$(basename "$config" .jl)
    rundir="runs/${variant}_$stamp"
    mkdir -p "$rundir"
    echo "=== train $config -> $rundir"
    julia +1.12.5 -t 8 --project train.jl "$config" "$rundir" "$n_iters" 2>&1 | tee "$rundir/train.log"
done
