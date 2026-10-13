#!/usr/bin/env bash
# Fast end-to-end check on a fake LBNL-layout dataset with a known answer
# (target SupplyFanCapacitor with SF_WAT, SF_SPD, SA_CFM dropping together).
set -euo pipefail
cd "$(dirname "$0")"
python lstm_numpy.py
python make_fake_lbnl.py --out data/raw/fake_lbnl/extracted --days 84
python run_real.py --data data/raw/fake_lbnl --out results/fake_lbnl --nonsig clean \
  --n-train 2000 --n-test 600 --n-other 200 --sweep-sizes 500,2000 --lstm-epochs 15
echo "Expected: target SupplyFanCapacitor_030, signature SF_WAT / SF_SPD / SA_CFM (all negative)."
