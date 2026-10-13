"""
make_fake_lbnl.py -- small fake dataset with the LBNL file layout, for smoke-testing
run_real.py before the real download (and after any code change).

Layout written to --out:
    FaultFree.csv
    <FaultName>_<severity>.csv     (same weather / schedule / noise as FaultFree, fault imposed)
Columns: 'Datetime' (M/D/YYYY H:MM, 1-min), ~22 AHU-like points, and a
'Fault Detection Ground Truth' column that the loader must exclude.

Built-in structure (so the expected answer is known):
    target  SupplyFanCapacitor   -> SF_WAT, SA_CFM, SF_SPD drop together  (narrow AND)
    other   DuctLeak             -> SF_WAT, SA_CFM drop                  (2 of 3)
    other   BeltSlip             -> SA_CFM drops a lot                   (1 of 3, big)
    other   FanOverspeed         -> the same three channels rise         (UP)
    other   OADamperStuck        -> OA_DMPR, MA_TEMP, CHWC_VLV           (no overlap)
    other   SATSensorBias        -> SA_TEMP                              (no overlap)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POINTS = ["OA_TEMP", "RA_TEMP", "MA_TEMP", "SA_TEMP", "SA_TEMP_SPT", "SA_SP", "SA_SPSPT",
          "SF_SPD", "RF_SPD", "SF_WAT", "RF_WAT", "SA_CFM", "RA_CFM", "OA_DMPR", "CHWC_VLV",
          "HWC_VLV", "ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "OCC", "SYS_CTL"]

FAULTS = {  # name: {channel: shift in units of that channel's noise-free scale}
    "SupplyFanCapacitor": {"SF_WAT": -1.0, "SA_CFM": -1.0, "SF_SPD": -1.0},
    "DuctLeak": {"SF_WAT": -1.0, "SA_CFM": -1.0, "SA_SP": -0.6},
    "BeltSlip": {"SA_CFM": -2.5},
    "FanOverspeed": {"SF_WAT": 1.0, "SA_CFM": 1.0, "SF_SPD": 1.0},
    "OADamperStuck": {"OA_DMPR": 1.5, "MA_TEMP": 1.2, "CHWC_VLV": 1.0},
    "SATSensorBias": {"SA_TEMP": 1.5},
}
SEVERITIES = {"SupplyFanCapacitor": [10, 30], "DuctLeak": [20], "BeltSlip": [40],
              "FanOverspeed": [15], "OADamperStuck": [100], "SATSensorBias": [2]}


def ar1(rng, n, phi, sd):
    e = rng.normal(scale=sd * np.sqrt(1 - phi ** 2), size=n)
    x = np.empty(n)
    x[0] = rng.normal(scale=sd)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def base_signals(n_days, seed):
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = np.arange(n)
    hour = (t % 1440) / 60.0
    day = t // 1440
    occ = ((hour >= 6) & (hour < 19) & (day % 7 < 5)).astype(float)
    oa = 75 + 12 * np.sin(2 * np.pi * (hour - 9) / 24) + 5 * np.sin(2 * np.pi * day / 30) + ar1(rng, n, 0.999, 3)
    load = np.clip((oa - 60) / 30, 0, 1.3) * occ + ar1(rng, n, 0.995, 0.08) * occ
    d = {}
    d["OA_TEMP"] = oa
    d["RA_TEMP"] = 72 + 2 * load + ar1(rng, n, 0.99, 0.4)
    d["OA_DMPR"] = occ * (20 + 10 * (oa < 65)) + ar1(rng, n, 0.98, 2)
    frac = d["OA_DMPR"] / 100
    d["MA_TEMP"] = frac * oa + (1 - frac) * d["RA_TEMP"] + ar1(rng, n, 0.95, 0.5)
    d["SA_TEMP_SPT"] = np.where(occ > 0, 55.0, 60.0)
    d["SA_TEMP"] = d["SA_TEMP_SPT"] + ar1(rng, n, 0.9, 0.6)
    d["SF_SPD"] = occ * (45 + 35 * load) + ar1(rng, n, 0.97, 2.5) * occ
    d["RF_SPD"] = 0.9 * d["SF_SPD"] + ar1(rng, n, 0.97, 1.5) * occ
    d["SA_CFM"] = 120 * d["SF_SPD"] + ar1(rng, n, 0.97, 1500) * occ
    d["RA_CFM"] = 0.85 * d["SA_CFM"] + ar1(rng, n, 0.9, 200) * occ
    d["SF_WAT"] = 0.0035 * d["SF_SPD"] ** 3 / 10 + ar1(rng, n, 0.9, 120) * occ
    d["RF_WAT"] = 0.6 * d["SF_WAT"] + ar1(rng, n, 0.9, 80) * occ
    d["SA_SPSPT"] = np.where(occ > 0, 1.2, 0.0)
    d["SA_SP"] = d["SA_SPSPT"] * (0.8 + 0.3 * load) + ar1(rng, n, 0.9, 0.08) * occ
    d["CHWC_VLV"] = np.clip(100 * (d["MA_TEMP"] - 55) / 25, 0, 100) * occ + ar1(rng, n, 0.95, 4) * occ
    d["HWC_VLV"] = np.clip(100 * (58 - d["MA_TEMP"]) / 15, 0, 100) * occ
    for z in range(1, 5):
        d[f"ZONE_TEMP_{z}"] = 72 + 1.5 * (1 - occ) * np.sin(2 * np.pi * hour / 24) + ar1(rng, n, 0.995, 0.5)
    d["OCC"] = occ
    d["SYS_CTL"] = occ
    return pd.DataFrame(d)[POINTS], occ, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/fake_lbnl/extracted")
    ap.add_argument("--days", type=int, default=56)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, occ, rng = base_signals(a.days, a.seed)
    stamps = pd.date_range("2018-01-01", periods=len(df), freq="1min")
    ts = [f"{s.month}/{s.day}/{s.year} {s.hour}:{s.minute:02d}" for s in stamps]

    def write(name, frame, gt):
        frame = frame.copy()
        frame.insert(0, "Datetime", ts)
        frame["Fault Detection Ground Truth"] = gt
        frame.round(4).to_csv(out / f"{name}.csv", index=False)
        print("wrote", out / f"{name}.csv")

    write("FaultFree", df, 0)
    # shifts are in units of each channel's 15-min-averaged SD in the fault-free file
    # (the same units run_real.py standardizes with)
    tmp = df.copy()
    tmp.index = stamps
    scale = tmp.resample("15min").mean().std()
    for fname, shifts in FAULTS.items():
        for sev in SEVERITIES[fname]:
            g = df.copy()
            # fault manifests when the fan runs, on ~70% of days, ramping in over 20 min
            days = len(df) // 1440
            on_day = np.repeat(rng.random(days) < 0.7, 1440)[:len(df)]
            act = occ * on_day
            act = np.convolve(act, np.ones(20) / 20, mode="same")
            mult = 1.0 if fname != "SupplyFanCapacitor" else sev / 20
            for ch, sh in shifts.items():
                g[ch] = g[ch] + mult * sh * scale[ch] * act
            write(f"{fname}_{sev:03d}", g, 1)


if __name__ == "__main__":
    main()
