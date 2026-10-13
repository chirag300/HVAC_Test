"""
run_real.py -- run the shortcut-learning test battery on a real LBNL FDD dataset.

  # 1. look before you leap: detected files, pairing, channels, fault profiles, target ranking
  python run_real.py --data data/raw/sdahu --out results/sdahu --inspect

  # 2. full run (auto-picked target, or name one with --target <substring of file name>)
  python run_real.py --data data/raw/sdahu --out results/sdahu
  python run_real.py --data data/raw/rtu   --out results/rtu --target <fault_file_substring>

Outputs per variant in <out>/<target>/<clean|real>/ :
  REPORT.md (hypothesis verdicts + all tables), main.json, ident.json, sweep.json, fix.json,
  fig_flag_rates.png, fig_unknown_overlap.png, fig_sweep.png, fig_targeted_negatives.png
Plus <out>/inspect/: fault_profiles_sd_units.csv, target_ranking.csv, dataset_summary.json

Variants (--nonsig):
  clean  only the k signature channels carry the target fault's real deviation; all other
         channels stay normal.  This is the slide's mechanism on real signals (default first).
  real   training positives are the full real faulty windows (all channels deviate).
         Partial-match constructs keep the real non-signature deviations.
  both   run clean, then real.
Every step is cached (JSON); re-running resumes. --force recomputes.
"""
import argparse
import re
import time

import shortcut_lib as L
from experiments import run_suite
from real_data import LBNLSource, NORMAL_RE, DROP_COLS_RE, SETPOINT_RE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing the dataset CSVs (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="auto")
    ap.add_argument("--k", type=int, default=3, help="signature size (slide: 3)")
    ap.add_argument("--nonsig", default="both", choices=["clean", "real", "both"])
    ap.add_argument("--resample", default="15min", help="pandas offset; 48 steps x 15min = 12 h windows")
    ap.add_argument("--base-freq", default="1min", help="assumed spacing if timestamps can't be parsed")
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--onset", default="10,25")
    ap.add_argument("--min-dev", type=float, default=0.5,
                    help="deviation (in fault-free SD units) that counts as 'channel moved'")
    ap.add_argument("--dedup-corr", type=float, default=0.98,
                    help="skip a signature candidate if |corr| with an already chosen one exceeds this")
    ap.add_argument("--normal-regex", default=NORMAL_RE)
    ap.add_argument("--drop-cols-regex", default=DROP_COLS_RE)
    ap.add_argument("--keep-setpoints", action="store_true",
                    help="keep setpoint columns (dropped by default: controller targets, not sensors)")
    ap.add_argument("--mismatch-frac", type=float, default=0.8,
                    help="drop a channel that moves the same way in this fraction of fault families")
    ap.add_argument("--include-regex", default=None, help="only use CSVs whose path matches")
    ap.add_argument("--exclude-regex", default=None)
    ap.add_argument("--family-regex", default=None,
                    help="regex on file stem; group(1) = fault family (severities share a family)")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-other", type=int, default=300, help="windows per unknown-fault file")
    ap.add_argument("--sweep-sizes", default="500,2000,8000")
    ap.add_argument("--sweep-prev", default="0.1,0.5,0.9")
    ap.add_argument("--steps", default="main,ident,sweep,fix")
    ap.add_argument("--models", default="all", help="comma list of model names or 'all'")
    ap.add_argument("--lstm-epochs", type=int, default=30)
    ap.add_argument("--lstm-hidden", type=int, default=32)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="print/save dataset diagnostics and stop")
    a = ap.parse_args()

    L.DETECTORS["LSTM (raw series)"] = ("raw48", lambda s: L.NumpyLSTMClassifier(
        hidden=a.lstm_hidden, epochs=a.lstm_epochs, seed=s))
    models = list(L.DETECTORS) if a.models == "all" else [m.strip() for m in a.models.split(",")]
    variants = ["clean", "real"] if a.nonsig == "both" else [a.nonsig]

    t0 = time.time()
    src = None
    for v in variants:
        print(f"\n=== loading {a.data} (non-signature mode: {v}) ===")
        if src is None:
            src = LBNLSource(a.data, target=a.target, k=a.k, resample=a.resample,
                             base_freq=a.base_freq, T=a.window,
                             onset=tuple(int(x) for x in a.onset.split(",")),
                             min_dev=a.min_dev, dedup_corr=a.dedup_corr, nonsig=v,
                             normal_regex=a.normal_regex, drop_cols_regex=a.drop_cols_regex,
                             include_regex=a.include_regex, exclude_regex=a.exclude_regex,
                             family_regex=a.family_regex, cache_dir=a.cache, seed=a.seed,
                             setpoint_regex=None if a.keep_setpoints else SETPOINT_RE,
                             mismatch_frac=a.mismatch_frac)
            print(src.describe(f"{a.out}/inspect"))
            print(f"  diagnostics written to {a.out}/inspect/")
            if a.inspect:
                return
        else:  # same data, switch mode
            src.nonsig = v
            src.name = re.sub(r"\((clean|real|normal) non", f"({v} non", src.name)
        cfg = dict(n_train=a.n_train, n_val=2000, n_test=a.n_test, n_other=a.n_other, seed=a.seed,
                   force=a.force, models=models, nonsig=v,
                   sweep_sizes=[int(x) for x in a.sweep_sizes.split(",")],
                   sweep_prev=[float(x) for x in a.sweep_prev.split(",")],
                   target=src.target, signature=[src.channel_names[c] for c in src.sig])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.target)[:80]
        run_suite(src, f"{a.out}/{safe}/{v}", cfg, steps=tuple(a.steps.split(",")))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
