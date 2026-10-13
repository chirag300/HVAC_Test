"""
tep_to_csv.py -- Tennessee Eastman Process (TEP) benchmark -> LBNL-style CSVs, so run_real.py
works on it unchanged.

Source: the classic Braatz TEP data (one fault-free and 21 faulty test runs, 960 samples each,
3-minute sampling, 41 measured XMEAS + 11 manipulated XMV variables; in every faulty test run the
fault starts at sample 160 = hour 8). Mirrored on GitHub (camaramm/tennessee-eastman-profBraatz).

    python tep_to_csv.py                      # -> data/raw/tep/extracted/*.csv
    python run_real.py --data data/raw/tep --out results/tep --resample none --block 4h \
        --target IDV04 --nonsig clean

Why TEP: it is a non-HVAC process with real closed-loop dynamics, and faults IDV4, IDV11 and
IDV14 all act on the reactor cooling water loop (step, random variation, sticking valve) --
the "other faults partly overlap the signature" situation from the slides, in a public benchmark.

Caveat (stated in the report too): unlike the LBNL files, the TEP fault-free and faulty runs use
different random seeds, so  D = faulty - fault_free  contains the fault effect plus one extra
run's process noise. Spliced constructs are still real process states (fault data on some
channels, fault-free data on the others), but they are not exact counterfactuals.
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master"
COLS = [f"XMEAS_{i:02d}" for i in range(1, 42)] + [f"XMV_{i:02d}" for i in range(1, 12)]
# letters (not digits) in names so severity-stripping never merges two different faults
IDV = {
    1: "AC_FeedRatio_Step", 2: "B_Composition_Step", 3: "D_FeedTemp_Step",
    4: "ReactorCoolingWaterInletTemp_Step", 5: "CondenserCoolingWaterInletTemp_Step",
    6: "A_FeedLoss_Step", 7: "C_HeaderPressureLoss_Step", 8: "ABC_FeedComposition_Random",
    9: "D_FeedTemp_Random", 10: "C_FeedTemp_Random", 11: "ReactorCoolingWaterInletTemp_Random",
    12: "CondenserCoolingWaterInletTemp_Random", 13: "ReactionKinetics_Drift",
    14: "ReactorCoolingWaterValve_Sticking", 15: "CondenserCoolingWaterValve_Sticking",
    16: "Unknown_A", 17: "Unknown_B", 18: "Unknown_C", 19: "Unknown_D", 20: "Unknown_E",
    21: "Stream4Valve_Fixed",
}
ONSET = 160  # fault introduced at sample 160 in the *_te test runs


def fetch(name, raw):
    dest = raw / name
    if not dest.exists():
        print(f"  downloading {name}")
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)
    return np.loadtxt(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/tep")
    ap.add_argument("--faults", default="1-21", help="e.g. 1-21 or 4,11,14")
    a = ap.parse_args()
    root = Path(a.root)
    raw, out = root / "dat", root / "extracted"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if "-" in a.faults:
        lo, hi = map(int, a.faults.split("-"))
        faults = list(range(lo, hi + 1))
    else:
        faults = [int(x) for x in a.faults.split(",")]

    def write(name, arr, gt):
        if arr.shape[1] != len(COLS) and arr.shape[0] == len(COLS):
            arr = arr.T
        df = pd.DataFrame(arr, columns=COLS)
        df.insert(0, "Datetime", pd.date_range("2000-01-01", periods=len(df), freq="3min")
                  .strftime("%Y-%m-%d %H:%M"))
        df["Fault Detection Ground Truth"] = gt
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"  wrote {name}.csv  {df.shape[0]} rows")

    write("TEP_FaultFree", fetch("d00_te.dat", raw), 0)
    for i in faults:
        arr = fetch(f"d{i:02d}_te.dat", raw)
        gt = (np.arange(len(arr)) >= ONSET).astype(int)
        write(f"TEP_IDV{i:02d}_{IDV[i]}", arr, gt)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
