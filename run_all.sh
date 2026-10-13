#!/usr/bin/env bash
# Full pipeline: sanity checks -> toy replica -> download -> inspect + run each LBNL dataset.
# Re-runnable: every step caches its results.
set -euo pipefail
cd "$(dirname "$0")"

python lstm_numpy.py
python run_toy.py --out results/toy

python download.py sdahu fcu rtu nist   # follow the printed manual step if LBNL shows a form

for ds in sdahu fcu rtu; do
  if [ -n "$(find "data/raw/$ds" -name '*.csv' 2>/dev/null | head -1)" ]; then
    python run_real.py --data "data/raw/$ds" --out "results/$ds" --inspect
    python run_real.py --data "data/raw/$ds" --out "results/$ds"
  else
    echo "skip $ds: no CSVs yet (see download.py output)"
  fi
done

if [ -f data/raw/nist/fdd_heat_pump_cooling_1416seer_longshort_alldata_2019.xlsx ]; then
  python run_nist.py --xlsx data/raw/nist/fdd_heat_pump_cooling_1416seer_longshort_alldata_2019.xlsx --inspect
  echo "NIST: choose --fault-col / --cond-cols from the listing above, then run run_nist.py (see README)."
fi
echo "Reports: results/*/*/*/REPORT.md"
