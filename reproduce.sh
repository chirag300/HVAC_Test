#!/usr/bin/env bash
# Reproduce the CS 886 report "Sum Instead of AND" (see README.md for what each step produces).
# Every experiment caches its cells as JSON, so the script can be interrupted and re-run.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Checks: model gradients and every closed-form result of the report
python transformer_numpy.py
python theory/attention_checks.py
python verify_all.py

# 2. Synthetic data (about 1.5 hours on two CPU cores)
python run_transformer.py --out results/transformer_toy --seeds 0,1,2

# 3. LBNL fan-coil unit (download first; about 2 hours on two CPU cores)
python download.py fcu
if [ -n "$(find data/raw/fcu -name '*.csv' 2>/dev/null | head -1)" ]; then
  python run_transformer.py --data data/raw/fcu --out results/transformer_fcu \
    --target FCU_FanOutletBlockage --seeds 0,1,2 --rhos 0,0.1,0.2,0.4 --snr-scales 0.5,1
else
  echo "No fan-coil CSVs yet: follow the instructions printed by download.py, then re-run this script."
fi
