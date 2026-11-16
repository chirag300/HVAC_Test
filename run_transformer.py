"""
run_transformer.py -- transformer tests of the theory (sum vs. AND), on the toy/synthetic source
or on an LBNL dataset. NumPy only, same sources and constructs as run_toy.py / run_real.py.

    python run_transformer.py --out results/transformer_toy                     # synthetic (toy)
    python run_transformer.py --out results/transformer_toy --seeds 0,1,2 --quick
    python run_transformer.py --data data/raw/fcu --out results/transformer_fcu  # LBNL (after download.py)

Steps (--steps, comma list; every cell cached in JSON, re-runs resume):

  battery   Train every model on four training sets -- normal-vs-target ("plain"), plus channel
            near-misses ("+NM", all-but-one negatives), plus channel and pair-staggered near-misses
            ("+NM+PS"), and the same plus duration-matched positives ("+NM+PS+H": CAPH windows
            labelled 1, so pair-staggered windows cannot be rejected by duration alone) -- and measure flag rates on: NORMAL, CAP (target), P1..P{k-1} (partial
            look-alikes), BIG1 (one loud sensor), ASYNC (fully staggered), PAIRSTAG (pair-staggered),
            CAPH (duration-matched target: same per-sensor duration as PAIRSTAG).
            Also the false-alarm law: P{m} predicted from each model's own detection rate,
            Phi(m/k (Phi^-1(det) + z) - z)  (theory: Cor. 3 / R4).
  softand   Soft-AND sweep (theory: Theorem E). The attention-pooling transformer is trained with
            near-miss share rho in --rhos and fault scale s in --snr-scales. Measures the learned
            focus of the pooling attention: beta_hat (slope of -log attention on per-sensor
            evidence, within the signature sensors; 0 = averaging, large = attends to the weakest
            sensor) and the left-out-sensor ratio on near-miss windows. Theory: beta* finite,
            rising with rho and with signal strength.
  strength  Look-alike strength sweep (theory: R3 / Theorem 4(d)). Models from the battery
            ("plain" and "+NM") are tested on all-but-one look-alikes and single-sensor look-alikes
            whose shift is multiplied by M in --strengths. Theory: additive detectors (PatchTST,
            channel-indep.) rise toward 1 as M grows; bounded-evidence (RevIN, capped) and
            attention-AND detectors do not.

Outputs: <out>/battery.json, softand.json, strength.json, REPORT_transformer.md, fig_*.png
"""
import argparse
import json
import os
import time
from statistics import NormalDist

import numpy as np

import shortcut_lib as L
from experiments import make_train

_N = NormalDist()
TF_MODELS = ["Transformer (sensor tokens)", "Attention pool (sensor tokens)",
             "PatchTST (channel-indep.)", "PatchTST + RevIN"]
BASELINES = ["MLP (raw series)", "Channel-indep. (raw series)", "Capped CI (raw series)"]
POOL = "Attention pool (sensor tokens)"


def _log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def law(det, m, k, alpha=L.FPR):
    """Flag rate of an m-of-k look-alike predicted from the detection rate alone (sum rule)."""
    det = min(max(det, 1e-6), 1 - 1e-6)
    z = _N.inv_cdf(1 - alpha)
    return _N.cdf(m / k * (_N.inv_cdf(det) + z) - z)


# ----------------------------------------------------------------------------- attention focus
def evidence(src, X, last=20):
    """Per-sensor fault evidence: mean of the last `last` steps (all post-onset), in fault direction."""
    return X[:, -last:, :][:, :, src.sig].mean(1) * np.asarray(src.signs)[None, :]


def focus_stats(src, model, X_cap, X_nm):
    """beta_hat: pooled within-window slope of -log(pi) on evidence over the signature sensors
    (0 = uniform/averaging; > 0 = more weight on sensors with LESS evidence, i.e. AND-like).
    left_ratio: on all-but-one windows, pi(left-out sensor) / mean pi(other signature sensors).
    sig_mass: total pooling weight on the signature sensors."""
    k = len(src.sig)
    out = {}
    num = den = 0.0
    for X in (X_cap, X_nm):
        pi = model.attention(X)
        lp = np.log(pi[:, src.sig] + 1e-12)
        a = evidence(src, X)
        lp = lp - lp.mean(1, keepdims=True)
        a = a - a.mean(1, keepdims=True)
        num += float((a * lp).sum())
        den += float((a * a).sum())
    out["beta_hat"] = -num / den if den > 0 else float("nan")
    pi = model.attention(X_nm)
    left = np.array([src.sig[n % k] for n in range(len(X_nm))])
    pl = pi[np.arange(len(X_nm)), left]
    others = (pi[:, src.sig].sum(1) - pl) / (k - 1)
    out["left_ratio"] = float(np.mean(pl / np.maximum(others, 1e-12)))
    out["sig_mass"] = float(model.attention(X_cap)[:, src.sig].sum(1).mean())
    return out


# ----------------------------------------------------------------------------- steps
# training sets: extra label-0 constructs (fraction of n_train); "+NM+PS+H" also adds duration-matched
# positives (CAPH, label 1) so that "short shift" stops being a cue for "pair-staggered = normal"
TRAIN_SETS = {"plain": [], "+NM": [("ALLBUT1", 0.5)], "+NM+PS": [("ALLBUT1", 0.25), ("PAIRSTAG", 0.25)],
              "+NM+PS+H": [("ALLBUT1", 0.25), ("PAIRSTAG", 0.25)]}
EXTRA_POS = {"+NM+PS+H": [("CAPH", 0.25)]}


def make_train_sets(src, n, tset, rng):
    X, y = make_train(src, n, 0.5, rng, extras=[(kd, int(f * n)) for kd, f in TRAIN_SETS[tset]])
    for kd, f in EXTRA_POS.get(tset, []):
        Xp = src.construct("train", kd, int(f * n), rng)
        X, y = np.concatenate([X, Xp]), np.concatenate([y, np.ones(len(Xp))])
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
    return X, y


def step_battery(src, cfg, out, models_cache, keep_models=False):
    path = os.path.join(out, "battery.json")
    res = _load(path)
    k = len(src.sig)
    for seed in cfg["seeds"]:
        rng_t = np.random.default_rng(1000 + seed)
        kinds = ["CAP", "CAPH"] + [f"P{m}" for m in range(1, k)] + ["ALLBUT1", "BIG1", "ASYNC", "PAIRSTAG"]
        tests = {"NORMAL": src.normal("test", cfg["n_test"], rng_t)}
        for kind in kinds:
            tests[kind] = src.construct("test", kind, cfg["n_test"], rng_t)
        for tset, extras in TRAIN_SETS.items():
            key = f"seed={seed}|{tset}"
            cell = res.setdefault(key, {})
            # refit a cached model only when a later step (strength) needs the fitted object
            todo = [m for m in cfg["models"]
                    if m not in cell or (keep_models and tset in ("plain", "+NM") and (m, seed, tset) not in models_cache)]
            if not todo:
                continue
            rng = np.random.default_rng(seed)
            n = cfg["n_train"]
            Xtr, ytr = make_train_sets(src, n, tset, rng)
            Xval = src.normal("val", cfg["n_val"], rng)
            _log(f"battery: seed {seed}, train set {tset}: fitting {len(todo)} models")
            dets = L.fit_detectors(Xtr, ytr, Xval, todo, seed, log=lambda *_: None)
            for name, d in dets.items():
                models_cache[(name, seed, tset)] = d
                r = {t: float(d.flag(X).mean()) for t, X in tests.items()}
                r["law"] = {f"P{m}": law(r["CAP"], m, k) for m in range(1, k)}
                if name == POOL:
                    r["focus"] = focus_stats(src, d.model, tests["CAP"], tests["ALLBUT1"])
                cell[name] = r
            if "AND rule (true)" not in cell:
                andr = L.AndRule(src.sig, src.signs, Xval, **getattr(src, "and_kwargs", {}))
                cell["AND rule (true)"] = {t: float(andr.flag(X).mean()) for t, X in tests.items()}
            _save(path, res)
    return res


def step_softand(src, cfg, out):
    path = os.path.join(out, "softand.json")
    res = _load(path)
    n = cfg["n_train"]
    for seed in cfg["seeds"]:
        for s in cfg["snr_scales"]:
            rng_t = np.random.default_rng(2000 + seed)
            ckw = {"dscale": s}
            tests = {"NORMAL": src.normal("test", cfg["n_test"], rng_t),
                     "CAP": src.construct("test", "CAP", cfg["n_test"], rng_t, **ckw),
                     "ALLBUT1": src.construct("test", "ALLBUT1", cfg["n_test"], rng_t, **ckw)}
            for rho in cfg["rhos"]:
                key = f"seed={seed}|scale={s}|rho={rho}"
                if key in res:
                    continue
                rng = np.random.default_rng(seed)
                n_nm = int(round(rho * n / (1 - rho)))      # near-misses are a share rho of all training windows
                Xtr, ytr = make_train(src, n, 0.5, rng, extras=[("ALLBUT1", n_nm)], ckw=ckw)
                Xval = src.normal("val", cfg["n_val"], rng)
                _log(f"softand: seed {seed}, scale {s}, near-miss share {rho}")
                d = L.fit_detectors(Xtr, ytr, Xval, [POOL], seed, log=lambda *_: None)[POOL]
                r = {t: float(d.flag(X).mean()) for t, X in tests.items()}
                r.update(focus_stats(src, d.model, tests["CAP"], tests["ALLBUT1"]))
                res[key] = r
                _save(path, res)
    return res


def step_strength(src, cfg, out, models_cache):
    path = os.path.join(out, "strength.json")
    res = _load(path)
    for seed in cfg["seeds"]:
        rng_t = np.random.default_rng(3000 + seed)
        for tset in ("plain", "+NM"):
            for name in cfg["models"]:
                d = models_cache.get((name, seed, tset))
                if d is None:
                    continue
                key = f"seed={seed}|{tset}|{name}"
                if key in res:
                    continue
                row = {}
                for M in cfg["strengths"]:
                    row[f"ALLBUT1 x{M:g}"] = float(d.flag(src.construct("test", "ALLBUT1", cfg["n_test"] // 2,
                                                                        rng_t, dscale=M)).mean())
                    row[f"ONE x{M:g}"] = float(d.flag(src.construct("test", "SINGLE", cfg["n_test"] // 2, rng_t,
                                                                    channel=src.sig[0], dscale=M)).mean())
                res[key] = row
        _save(path, res)
    return res


# ----------------------------------------------------------------------------- report
def _mean_sd(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if not len(v):
        return "--"
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f}±{v.std(ddof=1):.2f}"


def write_report(src, cfg, out, battery, softand, strength):
    k = len(src.sig)
    seeds = cfg["seeds"]
    lines = [f"# Transformer tests — {src.name}", "",
             f"Signature: {[src.channel_names[c] for c in src.sig]}; seeds {seeds}; "
             f"n_train {cfg['n_train']}, n_test {cfg['n_test']}; thresholds at 1% false alarms on held-out normal windows.",
             "Entries are mean±sd over seeds.", ""]
    if battery:
        cols = ["NORMAL", "CAP", "CAPH"] + [f"P{m}" for m in range(1, k)] + ["BIG1", "ASYNC", "PAIRSTAG"]
        for tset in TRAIN_SETS:
            lines += [f"## Battery — training set: {tset}", "",
                      "| Model | " + " | ".join(cols) + f" | law P{k - 1} |", "|---" * (len(cols) + 2) + "|"]
            names = list(dict.fromkeys(n for s in seeds for n in battery.get(f"seed={s}|{tset}", {})))
            for name in names:
                cells = [battery.get(f"seed={s}|{tset}", {}).get(name) for s in seeds]
                cells = [c for c in cells if c]
                row = [_mean_sd([c.get(col) for c in cells]) for col in cols]
                lawv = _mean_sd([c["law"][f"P{k - 1}"] for c in cells if "law" in c]) if "law" in cells[0] else "--"
                lines.append(f"| {name} | " + " | ".join(row) + f" | {lawv} |")
            lines.append("")
        lines += ["## Learned focus of the attention-pooling transformer (theory: Theorem E)", "",
                  "| Training set | beta_hat | left-out ratio | pooling mass on signature |", "|---|---|---|---|"]
        for tset in TRAIN_SETS:
            fs = [battery.get(f"seed={s}|{tset}", {}).get(POOL, {}).get("focus") for s in seeds]
            fs = [f for f in fs if f]
            if fs:
                lines.append(f"| {tset} | {_mean_sd([f['beta_hat'] for f in fs])} | "
                             f"{_mean_sd([f['left_ratio'] for f in fs])} | {_mean_sd([f['sig_mass'] for f in fs])} |")
        lines.append("")
    if softand:
        lines += ["## Soft-AND sweep (attention pooling; theory: finite focus rising with near-miss share and signal)", "",
                  "| fault scale | near-miss share | beta_hat | left-out ratio | detection | all-but-one flagged |",
                  "|---|---|---|---|---|---|"]
        for s in cfg["snr_scales"]:
            for rho in cfg["rhos"]:
                cs = [softand.get(f"seed={sd}|scale={s}|rho={rho}") for sd in seeds]
                cs = [c for c in cs if c]
                if cs:
                    lines.append(f"| {s:g} | {rho:g} | {_mean_sd([c['beta_hat'] for c in cs])} | "
                                 f"{_mean_sd([c['left_ratio'] for c in cs])} | {_mean_sd([c['CAP'] for c in cs])} | "
                                 f"{_mean_sd([c['ALLBUT1'] for c in cs])} |")
        lines.append("")
    if strength:
        cols = [f"ALLBUT1 x{M:g}" for M in cfg["strengths"]] + [f"ONE x{M:g}" for M in cfg["strengths"]]
        for tset in ("plain", "+NM"):
            lines += [f"## Look-alike strength sweep — training set {tset} (theory: R3 / Theorem 4(d))", "",
                      "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
            for name in cfg["models"]:
                cs = [strength.get(f"seed={s}|{tset}|{name}") for s in seeds]
                cs = [c for c in cs if c]
                if cs:
                    lines.append(f"| {name} | " + " | ".join(_mean_sd([c[col] for c in cs]) for col in cols) + " |")
            lines.append("")
    with open(os.path.join(out, "REPORT_transformer.md"), "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def make_figures(src, cfg, out, battery, softand, strength):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    k = len(src.sig)
    seeds = cfg["seeds"]
    if softand:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for s in cfg["snr_scales"]:
            b, fa = [], []
            for rho in cfg["rhos"]:
                cs = [softand.get(f"seed={sd}|scale={s}|rho={rho}") for sd in seeds]
                cs = [c for c in cs if c]
                b.append(np.mean([c["beta_hat"] for c in cs]) if cs else np.nan)
                fa.append(np.mean([c["ALLBUT1"] for c in cs]) if cs else np.nan)
            ax[0].plot(cfg["rhos"], b, "o-", label=f"fault scale {s:g}")
            ax[1].plot(cfg["rhos"], fa, "o-", label=f"fault scale {s:g}")
        ax[0].set_xlabel("near-miss share of training data"); ax[0].set_ylabel("learned focus beta_hat")
        ax[1].set_xlabel("near-miss share of training data"); ax[1].set_ylabel("all-but-one look-alikes flagged")
        ax[0].legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_softand.png"), dpi=150); plt.close(fig)
    if strength:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for i, tset in enumerate(("plain", "+NM")):
            for name in cfg["models"]:
                cs = [strength.get(f"seed={s}|{tset}|{name}") for s in seeds]
                cs = [c for c in cs if c]
                if cs:
                    ax[i].plot(cfg["strengths"], [np.mean([c[f"ALLBUT1 x{M:g}"] for c in cs]) for M in cfg["strengths"]],
                               "o-", label=name)
            ax[i].set_title(f"training set {tset}"); ax[i].set_xlabel("look-alike strength M (x target shift)")
        ax[0].set_ylabel("all-but-one look-alikes flagged"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_strength.png"), dpi=150); plt.close(fig)
    if battery:
        fig, ax = plt.subplots(figsize=(5, 4))
        grid = np.linspace(0.01, 0.999, 200)
        for m in range(1, k):
            ax.plot(grid, [law(g, m, k) for g in grid], "k-" if m == k - 1 else "k--", lw=1)
        for tset, mk in zip(TRAIN_SETS, "o^sd"):
            for name in cfg["models"]:
                cs = [battery.get(f"seed={s}|{tset}", {}).get(name) for s in seeds]
                cs = [c for c in cs if c]
                if cs:
                    ax.scatter([np.mean([c["CAP"] for c in cs])], [np.mean([c[f"P{k - 1}"] for c in cs])], marker=mk, s=25)
        ax.set_xlabel("detection rate (CAP)"); ax.set_ylabel(f"P{k - 1} look-alikes flagged")
        ax.set_title("false-alarm law (lines) vs models\n(o plain, ^ +NM, s +NM+PS, d +NM+PS+H)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_law.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="LBNL dataset folder; omit for the synthetic toy source")
    ap.add_argument("--target", default="auto")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--nonsig", default="clean", choices=["clean", "real"])
    ap.add_argument("--resample", default="15min")
    ap.add_argument("--block", default="7D")
    ap.add_argument("--min-dev", type=float, default=0.5)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--steps", default="battery,softand,strength")
    ap.add_argument("--models", default=",".join(TF_MODELS + BASELINES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--rhos", default="0,0.05,0.1,0.2,0.4")
    ap.add_argument("--snr-scales", default="0.4,0.7,1")
    ap.add_argument("--strengths", default="0.5,1,1.5,2,3,5")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="fewer epochs and smaller sweeps")
    a = ap.parse_args()
    cfg = dict(models=[m.strip() for m in a.models.split(",")], seeds=[int(s) for s in a.seeds.split(",")],
               n_train=a.n_train, n_val=2000, n_test=a.n_test,
               rhos=[float(x) for x in a.rhos.split(",")], snr_scales=[float(x) for x in a.snr_scales.split(",")],
               strengths=[float(x) for x in a.strengths.split(",")])
    epochs = 12 if a.quick else a.epochs
    if a.quick:
        cfg.update(rhos=[0, 0.1, 0.4], snr_scales=[0.5, 1.0], strengths=[1, 2, 5])
    for name in TF_MODELS:   # epochs override
        kind, fac = L.DETECTORS[name]
        L.DETECTORS[name] = (kind, (lambda f: (lambda s: _with_epochs(f(s), epochs)))(fac))
    if a.data:
        from real_data import LBNLSource
        src = LBNLSource(a.data, target=a.target, k=a.k, block=a.block, min_dev=a.min_dev, nonsig=a.nonsig,
                         cache_dir=a.cache, resample=None if a.resample.lower() == "none" else a.resample)
    else:
        from run_toy import ToySource
        src = ToySource()
    os.makedirs(a.out, exist_ok=True)
    _save(os.path.join(a.out, "config.json"), cfg)
    steps = a.steps.split(",")
    models_cache = {}
    battery = step_battery(src, cfg, a.out, models_cache, keep_models="strength" in steps) if "battery" in steps else _load(os.path.join(a.out, "battery.json"))
    softand = step_softand(src, cfg, a.out) if "softand" in steps else _load(os.path.join(a.out, "softand.json"))
    strength = step_strength(src, cfg, a.out, models_cache) if "strength" in steps else _load(os.path.join(a.out, "strength.json"))
    print(write_report(src, cfg, a.out, battery, softand, strength))
    make_figures(src, cfg, a.out, battery, softand, strength)
    _log(f"report: {os.path.join(a.out, 'REPORT_transformer.md')}")


def _with_epochs(model, epochs):
    model.epochs = epochs
    return model


if __name__ == "__main__":
    main()
