#!/usr/bin/env bash
# Fast end-to-end check on a fake LBNL-layout dataset with a known answer
# (target SupplyFanCapacitor with SF_WAT, SF_SPD, SA_CFM dropping together).
set -euo pipefail
cd "$(dirname "$0")"
python lstm_numpy.py
python ci_numpy.py
python transformer_numpy.py
python theory/attention_checks.py D
python make_fake_lbnl.py --out data/raw/fake_lbnl/extracted --days 84
python run_real.py --data data/raw/fake_lbnl --out results/fake_lbnl --nonsig clean \
  --n-train 2000 --n-test 600 --n-other 200 --sweep-sizes 500,2000 --lstm-epochs 15 \
  --steps main,ident,sweep,fix,fixtime,fixboth,fixhelly,nmsweep,scale --nm-counts 0,10,100 --scales 0.5,1,1.5
python run_transformer.py --out results/smoke_transformer --seeds 0 --quick --n-train 1000 --n-test 300 \
  --models "Attention pool (sensor tokens),PatchTST + RevIN,Channel-indep. (raw series)" --rhos 0,0.3 --snr-scales 1 --strengths 1,3
echo "Expected: target SupplyFanCapacitor_030, signature SF_WAT / SF_SPD / SA_CFM (all negative)."
