#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=1
python self_check_arg.py --cuda-required
exec python train_synapse.py --arg-experiment E2 --aug-preset v4 --mix-refinement balance --work-dir results/arg_e2_balance/ "$@"
