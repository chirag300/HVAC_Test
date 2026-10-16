"""
experiments.py -- the full test battery from the slides, run against any "source"
(the toy generator or a real LBNL dataset).

A source must provide:
    name, C, sig (k signature channel indices), signs (fault direction per sig channel),
    channel_names
    normal(split, n, rng)                 -> (n,T,C) normal windows (baseline-subtracted)
    construct(split, kind, n, rng, **kw)  -> windows of a construct built from target-fault
                                             windows (CAP, P1.., BIG1, ASYNC, ...)
    train_negatives(n, rng)               -> normal windows for training
    unknown_faults(split, n, rng)         -> iterator of (name, X, overlap or None, group)
                                             group in {"other", "same_family"}

Steps (each cached as JSON so the run is resumable; --force recomputes):
    main   train 50/50, calibrate 1% FPR, flag rates on every test, true AND rule, learned-rule
           probes (per-channel singles, GB stat importance, LogReg weight share), unknown
           faults by signature overlap, fleet false-alarm share
    ident  CV accuracy using only some signature channels
    sweep  training size x capacitor prevalence -> partial-signature false alarms
    fix    add all-but-one ("targeted") negatives and re-measure
    fixtime / fixboth / fixhelly   other targeted-negative mixes: fully staggered (time),
           channel + fully staggered, channel + pair-staggered ("Helly" boundary negatives)
    nmsweep  recipe curve: partial false alarms vs number of near-miss negatives per type
             (cfg nm_counts, optional LSTM weight decays in cfg nm_wd)
    scale    signal-strength sweep: multiply the fault deviation by s (cfg scales); theory
             predicts detection AND partial false alarms both rise (detectability paradox)
"""
import json
import os
import time
from collections import defaultdict

import numpy as np

import shortcut_lib as L

EXTRA_KINDS = ["BIG1", "ASYNC", "ASYNC2", "PAIRSTAG", "UP", "OFF"]
GD_MODELS = ["MLP (raw series)", "LSTM (raw series)"]
CI = "Channel-indep. (raw series)"
CAPPED = "Capped CI (raw series)"
SWEEP_MODELS = ["LogReg (stats)", "Grad. boosting (stats)", "MLP (raw series)", "LSTM (raw series)",
                CI, CAPPED]
NM_MODELS = ["MLP (raw series)", "LSTM (raw series)", CI, CAPPED]
SCALE_MODELS = ["LogReg (stats)", "MLP (raw series)", "LSTM (raw series)", CI, CAPPED]
# targeted-negative variants: name -> [(construct kind, fraction of n_train added as label 0)]
FIX_VARIANTS = {
    "fix": [("ALLBUT1", 0.5)],                                   # channel identifiability
    "fixtime": [("ASYNC", 0.25), ("ASYNC2", 0.25)],              # time identifiability
    "fixboth": [("ALLBUT1", 0.25), ("ASYNC", 0.125), ("ASYNC2", 0.125)],
    # theory (Helly): the minimal time negatives are pair-staggered, not fully staggered
    "fixhelly": [("ALLBUT1", 0.25), ("PAIRSTAG", 0.25)],
}
FIX_LABEL = {"fix": "all-but-one (channel) negatives", "fixtime": "fully staggered (time) negatives",
             "fixboth": "all-but-one + fully staggered negatives",
             "fixhelly": "all-but-one + pair-staggered (Helly) negatives"}
FIX_STEPS = ("fixtime", "fixboth", "fixhelly")


def _log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def _load(path):
    with open(path) as f:
        return json.load(f)


def make_train(src, n, prev, rng, extra_kind=None, n_extra=0, extras=None, ckw=None):
    """extras: list of (construct kind, count) added as label-0 targeted negatives.
    ckw: extra keyword arguments for every construct (e.g. dscale for the signal sweep)."""
    ckw = ckw or {}
    n_pos = int(round(n * prev))
    Xp = src.construct("train", "CAP", n_pos, rng, **ckw)
    Xn = src.train_negatives(n - n_pos, rng)
    parts, ys = [Xp, Xn], [np.ones(len(Xp)), np.zeros(len(Xn))]
    extras = list(extras or []) + ([(extra_kind, n_extra)] if extra_kind else [])
    for kind, cnt in extras:
        if cnt <= 0:
            continue
        Xe = src.construct("train", kind, cnt, rng, **ckw)
        parts.append(Xe)
        ys.append(np.zeros(len(Xe)))
    X = np.concatenate(parts)
    y = np.concatenate(ys)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def construct_tests(src, split, n, rng, ckw=None, extra=True):
    ckw = ckw or {}
    k = len(src.sig)
    tests = {"NORMAL": src.normal(split, n, rng), "CAP": src.construct(split, "CAP", n, rng, **ckw)}
    for m in range(1, k):
        tests[f"P{m}"] = src.construct(split, f"P{m}", n, rng, **ckw)
    if not extra:
        return tests
    for kind in getattr(src, "test_kinds", EXTRA_KINDS):
        tests[kind] = src.construct(split, kind, n, rng, **ckw)
    for c in src.sig:
        tests[f"ONLY:{src.channel_names[c]}"] = src.construct(split, "SINGLE", n, rng, channel=c, **ckw)
    return tests


def eval_unknown(src, dets, split, n, rng):
    """Flag rates of real unknown faults, overall and bucketed by signature overlap."""
    k = len(src.sig)
    per_fault = {}
    bucket = {d.name: defaultdict(lambda: [0, 0]) for d in dets}
    for name, X, overlap, group in src.unknown_faults(split, n, rng):
        if len(X) == 0:
            continue
        row = {"group": group, "n": int(len(X))}
        if overlap is not None:
            row["mean_overlap"] = float(np.mean(overlap))
            row["frac_full_match"] = float(np.mean(overlap == k))
        for d in dets:
            fl = d.flag(X)
            row[d.name] = float(fl.mean())
            if overlap is not None and group == "other":
                for ov in range(k + 1):
                    sel = overlap == ov
                    bucket[d.name][ov][0] += int(fl[sel].sum())
                    bucket[d.name][ov][1] += int(sel.sum())
        per_fault[name] = row
    by_overlap = {dn: {str(ov): {"flag_rate": (v[0] / v[1]) if v[1] else None, "n": v[1]}
                       for ov, v in sorted(b.items())} for dn, b in bucket.items()}
    return per_fault, by_overlap


def fleet_share(rates_normal, rate_other, p_norm=0.89, p_other=0.10):
    num = p_other * rate_other
    den = p_norm * rates_normal + num
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------------------ steps
def step_main(src, cfg, out):
    path = os.path.join(out, "main.json")
    if os.path.exists(path) and not cfg["force"]:
        _log("main: cached")
        return _load(path)
    rng = np.random.default_rng(cfg["seed"])
    _log(f"main: building train set n={cfg['n_train']}")
    Xtr, ytr = make_train(src, cfg["n_train"], 0.5, rng)
    Xval = src.normal("val", cfg["n_val"], rng)
    dets = list(L.fit_detectors(Xtr, ytr, Xval, cfg["models"], cfg["seed"], log=_log).values())
    andr = L.AndRule(src.sig, src.signs, Xval, **getattr(src, "and_kwargs", {}))
    _log("main: building test constructs")
    tests = construct_tests(src, "test", cfg["n_test"], rng)
    rates = L.flag_rates(dets + [andr], tests)

    res = {"source": src.name, "k": len(src.sig),
           "signature": [src.channel_names[c] for c in src.sig],
           "signature_signs": [int(s) for s in src.signs],
           "signature_info": getattr(src, "sig_info", {}),
           "n_train": int(len(Xtr)), "test_sizes": {t: int(len(X)) for t, X in tests.items()},
           "rates": rates}
    named = {d.name: d for d in dets}
    if "Grad. boosting (stats)" in named:
        res["gb_stat_importance_sig"] = L.gb_stat_importance(named["Grad. boosting (stats)"], src.C, src.sig)
    if "LogReg (stats)" in named:
        share = L.logreg_channel_weight_share(named["LogReg (stats)"], src.C, src.sig)
        res["logreg_weight_share_sig"] = {src.channel_names[c]: v for c, v in share.items()}

    _log("main: unknown faults")
    per_fault, by_ov = eval_unknown(src, dets + [andr], "test", cfg["n_other"], rng)
    res["unknown_faults"] = per_fault
    res["unknown_by_overlap"] = by_ov
    others = [r for r in per_fault.values() if r["group"] == "other"]
    if others:
        res["fleet_false_alarm_share"] = {
            d: fleet_share(rates[d]["NORMAL"], float(np.mean([r[d] for r in others])))
            for d in rates}
    _save(path, res)
    return res


def step_ident(src, cfg, out):
    path = os.path.join(out, "ident.json")
    if os.path.exists(path) and not cfg["force"]:
        _log("ident: cached")
        return _load(path)
    rng = np.random.default_rng(cfg["seed"] + 1)
    X, y = make_train(src, min(cfg["n_train"], 4000), 0.5, rng)
    nm = src.channel_names
    subsets = {f"{nm[c]} only": [c] for c in src.sig}
    if len(src.sig) > 2:
        for i in range(len(src.sig)):
            pair = [c for j, c in enumerate(src.sig) if j != i]
            subsets["all sig but " + nm[src.sig[i]]] = pair
    subsets["all signature channels"] = list(src.sig)
    subsets[f"all {src.C} channels"] = list(range(src.C))
    _log("ident: channel-subset CV")
    res = L.subset_cv_accuracy(X, y, src.C, subsets, cfg["seed"])
    _save(path, res)
    return res


def step_sweep(src, cfg, out):
    path = os.path.join(out, "sweep.json")
    res = _load(path) if os.path.exists(path) and not cfg["force"] else {}
    rng = np.random.default_rng(cfg["seed"] + 2)
    kp = f"P{len(src.sig) - 1}"
    Xval = src.normal("val", cfg["n_val"], rng)
    Xpart = src.construct("test", kp, cfg["n_test"], rng)
    Xcap = src.construct("test", "CAP", cfg["n_test"], rng)
    for n in cfg["sweep_sizes"]:
        for prev in cfg["sweep_prev"]:
            key = f"n={n}|prev={prev}"
            if key in res:
                continue
            _log(f"sweep: {key}")
            Xtr, ytr = make_train(src, n, prev, rng)
            dets = L.fit_detectors(Xtr, ytr, Xval, [m for m in SWEEP_MODELS if m in cfg["models"]],
                                   cfg["seed"], log=lambda *_: None)
            res[key] = {d.name: {"partial_FA": float(d.flag(Xpart).mean()),
                                 "CAP": float(d.flag(Xcap).mean())} for d in dets.values()}
            _save(path, res)  # resumable per cell
    return res


def step_fix(src, cfg, out, name="fix"):
    path = os.path.join(out, f"{name}.json")
    if os.path.exists(path) and not cfg["force"]:
        _log(f"{name}: cached")
        return _load(path)
    rng = np.random.default_rng(cfg["seed"] + 3)
    n = cfg["n_train"]
    _log(f"{name}: training with {FIX_LABEL[name]}")
    extras = [(kind, int(frac * n)) for kind, frac in FIX_VARIANTS[name]]
    Xtr, ytr = make_train(src, n, 0.5, rng, extras=extras)
    Xval = src.normal("val", cfg["n_val"], rng)
    dets = list(L.fit_detectors(Xtr, ytr, Xval, cfg["models"], cfg["seed"], log=_log).values())
    tests = construct_tests(src, "test", cfg["n_test"], np.random.default_rng(cfg["seed"]))
    res = {"rates": L.flag_rates(dets, tests)}
    per_fault, by_ov = eval_unknown(src, dets, "test", cfg["n_other"], rng)
    res["unknown_faults"] = per_fault
    res["unknown_by_overlap"] = by_ov
    _save(path, res)
    return res


def _wd_variant(name, wd):
    kind, base = L.DETECTORS[name]

    def fac(seed):
        m = base(seed)
        m.wd = wd
        return m
    return kind, fac


def step_nmsweep(src, cfg, out):
    """Recipe curve (theory: near-miss supervision). Same normal-vs-fault training set every
    time; only the number of all-but-one negatives per near-miss type changes."""
    path = os.path.join(out, "nmsweep.json")
    res = _load(path) if os.path.exists(path) and not cfg["force"] else {}
    k = len(src.sig)
    kp = f"P{k - 1}"
    models = [m for m in NM_MODELS if m in cfg["models"]]
    for wd in cfg.get("nm_wd", []):
        name = f"LSTM wd={wd:g}"
        L.DETECTORS[name] = _wd_variant("LSTM (raw series)", wd)
        models.append(name)
    rng = np.random.default_rng(cfg["seed"] + 4)
    Xval = src.normal("val", cfg["n_val"], rng)
    tests = {"NORMAL": src.normal("test", cfg["n_test"], rng),
             "CAP": src.construct("test", "CAP", cfg["n_test"], rng),
             kp: src.construct("test", kp, cfg["n_test"], rng)}
    for c in cfg.get("nm_counts", [0, 3, 10, 30, 100, 300, 1000]):
        key = f"per_type={c}"
        todo = [m for m in models if m not in res.get(key, {})]
        if not todo:
            continue
        _log(f"nmsweep: {c} near-misses per type ({k * c} in total)")
        Xtr, ytr = make_train(src, cfg["n_train"], 0.5, np.random.default_rng(cfg["seed"] + 40),
                              extras=[("ALLBUT1", k * c)])
        dets = L.fit_detectors(Xtr, ytr, Xval, todo, cfg["seed"], log=lambda *_: None)
        cell = res.setdefault(key, {})
        for d in dets.values():
            cell[d.name] = {t: float(d.flag(X).mean()) for t, X in tests.items()}
        _save(path, res)
    return res


def step_scale(src, cfg, out):
    """Signal-strength sweep (theory: detectability paradox). The fault deviation D is
    multiplied by s in training AND test; nothing else changes."""
    path = os.path.join(out, "scale.json")
    res = _load(path) if os.path.exists(path) and not cfg["force"] else {}
    models = [m for m in SCALE_MODELS if m in cfg["models"]]
    for s in cfg.get("scales", [0.5, 0.75, 1.0, 1.5, 2.0]):
        key = f"scale={s:g}"
        todo = [m for m in models if m not in res.get(key, {})]
        if not todo:
            continue
        _log(f"scale: fault deviation x{s:g}")
        rng = np.random.default_rng(cfg["seed"] + 5)
        ckw = {"dscale": float(s)}
        Xtr, ytr = make_train(src, cfg["n_train"], 0.5, rng, ckw=ckw)
        Xval = src.normal("val", cfg["n_val"], rng)
        tests = construct_tests(src, "test", cfg["n_test"], rng, ckw=ckw, extra=False)
        dets = L.fit_detectors(Xtr, ytr, Xval, todo, cfg["seed"], log=lambda *_: None)
        cell = res.setdefault(key, {})
        for d in dets.values():
            cell[d.name] = {t: float(d.flag(X).mean()) for t, X in tests.items()}
        _save(path, res)
    return res


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(y) == 0 or np.ptp(x) == 0:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def theory_predictions(main):
    """Per model: measured P_m vs the theory's predictions (Cor. power-specificity curve) from
    (a) the detection rate alone, equal channels, and (b) detection + single-channel rates."""
    k = main["k"]
    n = main.get("test_sizes", {}).get("CAP")
    out = {}
    for model, r in main["rates"].items():
        singles = [r.get(f"ONLY:{c}") for c in main["signature"]]
        row = {"CAP": r["CAP"]}
        for m in range(1, k):
            row[f"P{m}"] = r.get(f"P{m}")
            row[f"P{m} pred (CAP only)"] = L.predict_from_detection(r["CAP"], m / k, n=n)
            row[f"P{m} pred (CAP+singles)"] = (L.predict_partial_additive(r["CAP"], singles, m, n=n)
                                               if None not in singles else float("nan"))
        out[model] = row
    return out


# ---------------------------------------------------------------------- report
def _verdict(ok, n_ok=True):
    if not n_ok:
        return "INCONCLUSIVE"
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def hypothesis_checks(main, ident, sweep, fix, fixes=None, extra=None):
    k = main["k"]
    kp = f"P{k - 1}"
    R = main["rates"]
    # gradient-descent model to test: the LSTM, or the MLP when there is no time axis
    lstm = "LSTM (raw series)" if "LSTM (raw series)" in R else "MLP (raw series)"
    gdn = lstm.split(" (")[0]
    andn = "AND rule (true)"
    out = []

    single = [v for kk, v in ident.items() if kk.endswith(" only")]
    allc = [v for kk, v in ident.items() if kk.startswith("all ") and kk.endswith("channels")]
    if single and allc:
        best, full = max(single), max(allc)
        out.append(("H1 one signature channel nearly separates the training data",
                    _verdict(best >= 0.90 and full - best <= 0.08),
                    f"best single-channel CV acc {best:.3f} vs all channels {full:.3f}"))

    if lstm in R:
        r = R[lstm]
        out.append((f"H2 {gdn} flags partial-signature faults ({kp})",
                    _verdict(r[kp] >= 0.30 and r[kp] - R[andn][kp] >= 0.20),
                    f"{gdn} {kp} flagged {r[kp]:.2f}; true AND rule {R[andn][kp]:.2f}; "
                    f"{gdn} CAP recall {r['CAP']:.2f}"))
        fooled = [m for m in R if m != andn and R[m][kp] >= 0.30]
        out.append(("H2b every model family is fooled the same way",
                    _verdict(len(fooled) == len(R) - 1),
                    f"{len(fooled)}/{len(R) - 1} models flag >=30% of {kp}: {', '.join(fooled)}"))
        learned_cap = np.mean([R[m]["CAP"] for m in R if m != andn])
        out.append(("H3 the true rule scores lower on the training task",
                    _verdict(R[andn]["CAP"] < learned_cap),
                    f"AND rule CAP recall {R[andn]['CAP']:.2f} vs learned mean {learned_cap:.2f}"))

    if sweep:
        vals = [cell[lstm]["partial_FA"] for cell in sweep.values() if lstm in cell]
        if vals:
            out.append((f"H4 more data / rebalancing does not remove partial false alarms ({gdn})",
                        _verdict(min(vals) >= 0.30),
                        f"{gdn} {kp} false-alarm rate across sweep: min {min(vals):.2f}, max {max(vals):.2f}"))

    if fix and lstm in fix["rates"] and lstm in R:
        before, after = R[lstm][kp], fix["rates"][lstm][kp]
        cap_drop = R[lstm]["CAP"] - fix["rates"][lstm]["CAP"]
        out.append((f"H5 targeted all-but-one negatives fix it ({gdn})",
                    _verdict(after <= 0.5 * before and cap_drop <= 0.05),
                    f"{kp}: {before:.2f} -> {after:.2f}; CAP recall change {-cap_drop:+.2f}"))

    fixes = fixes or {}
    time_cols = [c for c in ("ASYNC", "ASYNC2") if c in R.get(lstm, {})]
    ft = fixes.get("fixtime")
    if ft and lstm in ft["rates"] and lstm in R and time_cols:
        b = float(np.mean([R[lstm][c] for c in time_cols]))
        a = float(np.mean([ft["rates"][lstm][c] for c in time_cols]))
        cap_drop = R[lstm]["CAP"] - ft["rates"][lstm]["CAP"]
        out.append((f"H5t staggered (time) negatives fix simultaneity ({gdn})",
                    _verdict(a <= 0.5 * b and cap_drop <= 0.05, b >= 0.10),
                    f"ASYNC/ASYNC2 mean {b:.2f} -> {a:.2f}; {kp} {R[lstm][kp]:.2f} -> "
                    f"{ft['rates'][lstm][kp]:.2f} (time negatives alone should not fix it); "
                    f"CAP change {-cap_drop:+.2f}"))
    fb = fixes.get("fixboth")
    if fb and lstm in fb["rates"] and lstm in R and time_cols:
        cols = [kp] + time_cols
        b = float(np.mean([R[lstm][c] for c in cols]))
        a = float(np.mean([fb["rates"][lstm][c] for c in cols]))
        cap_drop = R[lstm]["CAP"] - fb["rates"][lstm]["CAP"]
        out.append((f"H5b channel + time negatives fix both ({gdn})",
                    _verdict(a <= 0.5 * b and cap_drop <= 0.05),
                    f"mean of {', '.join(cols)}: {b:.2f} -> {a:.2f}; CAP change {-cap_drop:+.2f}"))
    if fix and CI in fix["rates"] and lstm in fix["rates"] and CI in R:
        ci_a, ls_a = fix["rates"][CI][kp], fix["rates"][lstm][kp]
        out.append(("H8 additive (channel-independent) model stays fooled after the fix",
                    _verdict(ci_a >= ls_a + 0.10),
                    f"after all-but-one negatives: channel-indep. {kp} {R[CI][kp]:.2f} -> {ci_a:.2f} "
                    f"vs {gdn} {R[lstm][kp]:.2f} -> {ls_a:.2f}"))

    # ---- theory checks (theory/theory.pdf)
    extra = extra or {}
    if lstm in R and all(f"ONLY:{c}" in R[lstm] for c in main["signature"]):
        tp = theory_predictions(main)[lstm]
        meas, pred = tp[kp], tp[f"{kp} pred (CAP+singles)"]
        pred0 = tp[f"{kp} pred (CAP only)"]
        out.append((f"HA partial-match rate follows the additive law ({gdn})",
                    _verdict(abs(meas - pred) <= 0.15, not np.isnan(pred)),
                    f"{kp} measured {meas:.2f}; predicted from CAP + single-channel rates {pred:.2f}; "
                    f"from CAP alone (equal channels) {pred0:.2f}"))
    fh, ft = fixes.get("fixhelly"), fixes.get("fixtime")
    if fh and lstm in fh["rates"] and lstm in R:
        base = R[lstm].get("PAIRSTAG")
        a_h = fh["rates"][lstm].get("PAIRSTAG")
        a_t = ft["rates"][lstm].get("PAIRSTAG") if ft and lstm in ft["rates"] else None
        known = base is not None and a_h is not None and a_t is not None
        if known:
            asy = float(np.mean([ft["rates"][lstm][c] for c in time_cols])) if time_cols else float("nan")
            ok = a_h <= 0.5 * base and a_t >= a_h + 0.15
            msg = (f"PAIRSTAG flagged: before {base:.2f}; + fully staggered negatives {a_t:.2f} "
                   f"(their own ASYNC/ASYNC2 {asy:.2f}); + pair-staggered negatives {a_h:.2f}")
        else:
            ok, msg = False, "PAIRSTAG missing in main/fixtime results: rerun them with --force"
        out.append((f"H5h pair-staggered negatives teach simultaneity, fully staggered do not ({gdn})",
                    _verdict(ok, known and base >= 0.10), msg))
    if fix and CAPPED in fix["rates"] and CI in fix["rates"] and CAPPED in R and CI in R:
        cb, ca = R[CAPPED][kp], fix["rates"][CAPPED][kp]
        ub, ua = R[CI][kp], fix["rates"][CI][kp]
        out.append(("H9 capped per-channel evidence can be fixed, uncapped cannot",
                    _verdict(ca <= ua - 0.10 and ca <= 0.5 * cb, cb >= 0.10),
                    f"after all-but-one negatives {kp}: capped {cb:.2f} -> {ca:.2f} "
                    f"(CAP {R[CAPPED]['CAP']:.2f} -> {fix['rates'][CAPPED]['CAP']:.2f}); "
                    f"uncapped {ub:.2f} -> {ua:.2f}; BIG1 without any fix: capped "
                    f"{R[CAPPED].get('BIG1', float('nan')):.2f} vs uncapped {R[CI].get('BIG1', float('nan')):.2f}"))
    sc = extra.get("scale")
    if sc:
        ss = sorted(float(key.split("=")[1]) for key in sc)
        cells = [sc[f"scale={v:g}"] for v in ss]
        if all(lstm in c for c in cells):
            cap = [c[lstm]["CAP"] for c in cells]
            part = [c[lstm][kp] for c in cells]
            pred = [L.predict_from_detection(c, (k - 1) / k) for c in cap]
            rho = _spearman(ss, part)
            saturated = min(part) >= 0.95
            out.append((f"HP stronger target fault -> more partial false alarms ({gdn})",
                        _verdict(rho >= 0.8 and part[-1] - part[0] >= 0.10, not saturated),
                        "scale " + ", ".join(f"x{v:g}: CAP {c_:.2f} / {kp} {p_:.2f} (law {q:.2f})"
                                             for v, c_, p_, q in zip(ss, cap, part, pred))
                        + (" (saturated)" if saturated else "")))
    nm = extra.get("nmsweep")
    if nm:
        cs = sorted(int(key.split("=")[1]) for key in nm)
        for model in [lstm, CI, CAPPED] + [m for m in next(iter(nm.values())) if m.startswith("LSTM wd=")]:
            cells = [nm[f"per_type={c}"].get(model) for c in cs]
            if None in cells or len(cs) < 3:
                continue
            part = [c[kp] for c in cells]
            cap = [c["CAP"] for c in cells]
            reach = next((c for c, p_, q in zip(cs, part, cap)
                          if p_ <= 0.5 * part[0] and cap[0] - q <= 0.05), None)
            rho = _spearman(cs, part)
            short = model.split(" (")[0]
            out.append((f"HR more near-miss negatives -> fewer partial false alarms ({short})",
                        _verdict(rho <= -0.8 and part[-1] <= 0.5 * part[0], part[0] >= 0.10),
                        f"{kp} by near-misses per type: " +
                        ", ".join(f"{c}:{p_:.2f}" for c, p_ in zip(cs, part)) +
                        f"; halved (CAP kept) at {reach if reach is not None else '>' + str(cs[-1])} per type"))

    # H6 / H7 per model: real unknown faults (never trained on)
    others = [r for r in main.get("unknown_faults", {}).values() if r["group"] == "other"]
    for model, bo in main.get("unknown_by_overlap", {}).items():
        if model == andn:
            continue
        short = model.split(" (")[0]
        lo = bo.get("0", {})
        part = [bo[str(o)] for o in range(1, k) if bo.get(str(o), {}).get("n")]
        n_part = sum(p["n"] for p in part)
        enough = lo.get("n", 0) >= 30 and n_part >= 30
        if enough:
            r_part = sum(p["flag_rate"] * p["n"] for p in part) / n_part
            r0 = lo["flag_rate"]
            # flagged often, and clearly more often than faults that miss the signature
            ok = r_part >= 0.20 and r_part - r0 >= 0.15 and r_part >= 2 * max(r0, 0.01)
            per = ", ".join(f"ov{o} {bo[str(o)]['flag_rate']:.2f} (n={bo[str(o)]['n']})"
                            for o in range(1, k) if bo.get(str(o), {}).get("n"))
            msg = f"partial overlap {r_part:.2f} vs none {r0:.2f} (n={lo['n']}); {per}"
        else:
            ok, msg = False, (f"need >=30 windows in both groups: partial n={n_part}, "
                              f"none n={lo.get('n', 0)}")
        out.append((f"H6 real unknown faults with partial overlap are flagged ({short})",
                    _verdict(ok, enough), msg))
        sh = main.get("fleet_false_alarm_share", {}).get(model)
        if sh is not None and others:
            r_other = float(np.mean([r[model] for r in others]))
            meaningful = r_other >= 0.05  # below this the share is a ratio of tiny numbers
            out.append((f"H7 most fleet false alarms come from unknown faults ({short})",
                        _verdict(sh >= 0.5, meaningful),
                        f"{sh:.0%} of false alarms in an 89/1/10 fleet mix; "
                        f"unknown faults flagged {r_other:.2f} on average"
                        + ("" if meaningful else " (too rare to judge)")))
    return out


def write_report(src_name, out, main, ident, sweep, fix, cfg, fixes=None, extra=None):
    fixes, extra = fixes or {}, extra or {}
    k = main["k"]
    present = next(iter(main["rates"].values()))
    cols = [c for c in ["NORMAL", "CAP"] + [f"P{m}" for m in range(1, k)] + EXTRA_KINDS if c in present]
    only_cols = [c for c in next(iter(main["rates"].values())) if c.startswith("ONLY:")]
    lines = [f"# Shortcut-learning test: {src_name}", "",
             f"Signature (k={k}): " + ", ".join(
                 f"{n} ({'+' if s > 0 else '-'})" for n, s in
                 zip(main["signature"], main["signature_signs"])),
             f"Train windows: {main['n_train']} (50/50); thresholds at {L.FPR:.0%} false alarms "
             f"on held-out normal; nonsig mode: {cfg.get('nonsig', 'n/a')}", ""]
    if main.get("signature_info"):
        lines += ["Signature selection details:", "```",
                  json.dumps(main["signature_info"], indent=1)[:3000], "```", ""]
    lines += ["## Hypothesis checks", "", "| Check | Verdict | Evidence |", "|---|---|---|"]
    for name, v, ev in hypothesis_checks(main, ident, sweep, fix, fixes, extra):
        lines.append(f"| {name} | **{v}** | {ev} |")
    lines += ["", "## Flag rates (fraction flagged as the target fault)", "",
              L.md_table(main["rates"], cols), "",
              "## Which single channel carries the rule (only one signature channel deviates)", "",
              L.md_table(main["rates"], only_cols), ""]
    if "gb_stat_importance_sig" in main:
        lines += ["GB importance by statistic (summed over signature channels): " +
                  ", ".join(f"{s}={v:.3f}" for s, v in main["gb_stat_importance_sig"].items()), ""]
    if "logreg_weight_share_sig" in main:
        lines += ["LogReg |weight| share on signature channels: " +
                  ", ".join(f"{s}={v:.2f}" for s, v in main["logreg_weight_share_sig"].items()), ""]
    tp = theory_predictions(main)
    if any(not np.isnan(r.get(f"P{k - 1} pred (CAP+singles)", float("nan"))) for r in tp.values()):
        pcols = ["CAP"] + [c for m in range(1, k) for c in
                           (f"P{m}", f"P{m} pred (CAP only)", f"P{m} pred (CAP+singles)")]
        lines += ["## Theory check: partial-match rates predicted from the detection rate", "",
                  "Law (theory/theory.pdf, Cor. 'power-specificity curve'): "
                  "P_m = Phi(r (Phi^-1(CAP) + z) - z), z = Phi^-1(0.99). "
                  "'CAP only' assumes equal channels, r = m/k; 'CAP+singles' reads each channel's "
                  "share r_j from its single-channel flag rate and averages over channel subsets. "
                  "Close agreement = the model behaves like an additive (sum-of-evidence) detector.", "",
                  L.md_table(tp, pcols), ""]
    lines += ["## Identifiability: CV accuracy with channel subsets", "",
              "| Channels used | CV acc |", "|---|---:|"] + \
             [f"| {kk} | {v:.3f} |" for kk, v in ident.items()] + [""]
    uf = main.get("unknown_faults", {})
    if uf:
        det_names = list(main["rates"].keys())
        lines += ["## Real unknown faults (never seen in training)", "",
                  "| Fault file | group | n | mean overlap | full match | " +
                  " | ".join(det_names) + " |",
                  "|---|---|---:|---:|---:|" + "---:|" * len(det_names)]
        for fname, r in sorted(uf.items(), key=lambda kv: (kv[1]["group"], -kv[1].get("mean_overlap", 0))):
            lines.append(f"| {fname} | {r['group']} | {r['n']} | {r.get('mean_overlap', float('nan')):.2f} | "
                         f"{r.get('frac_full_match', float('nan')):.2f} | " +
                         " | ".join(f"{r[d]:.2f}" for d in det_names) + " |")
        lines += ["", "Flag rate of other faults by number of signature channels they move "
                      "in the fault direction:", "",
                  "| Model | " + " | ".join(f"overlap {o}" for o in range(k + 1)) + " |",
                  "|---|" + "---:|" * (k + 1)]
        for d, b in main["unknown_by_overlap"].items():
            cells = []
            for o in range(k + 1):
                v = b.get(str(o), {})
                cells.append(f"{v['flag_rate']:.2f} (n={v['n']})" if v.get("n") else "–")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        if "fleet_false_alarm_share" in main:
            lines += ["", "Share of fleet false alarms caused by unknown faults "
                          "(89% normal / 1% target / 10% other):", ""] + \
                     [f"- {d}: {v:.0%}" for d, v in main["fleet_false_alarm_share"].items()]
        lines.append("")
    if sweep:
        lines += ["## More data / balancing (partial-signature false-alarm rate)", "",
                  "| Train size | Prevalence | " + " | ".join(SWEEP_MODELS) + " |",
                  "|---:|---:|" + "---:|" * len(SWEEP_MODELS)]
        for key, cell in sweep.items():
            n, p = key.split("|")
            lines.append(f"| {n[2:]} | {p[5:]} | " + " | ".join(
                f"{cell[m]['partial_FA']:.2f}" if m in cell else "–" for m in SWEEP_MODELS) + " |")
        lines.append("")
    if fix:
        lines += ["## Targeted all-but-one negatives added to training", "",
                  L.md_table(fix["rates"], cols), ""]
        if fix.get("unknown_by_overlap"):
            lines += ["Other-fault flag rate by overlap after the fix:", ""]
            for d, b in fix["unknown_by_overlap"].items():
                lines.append(f"- {d}: " + ", ".join(
                    f"ov{o}={b[str(o)]['flag_rate']:.2f}" for o in range(k + 1)
                    if b.get(str(o), {}).get("n")))
            lines.append("")
    for name in FIX_STEPS:
        fx = fixes.get(name)
        if not fx:
            continue
        lines += [f"## Targeted {FIX_LABEL[name]} added to training", "",
                  L.md_table(fx["rates"], cols), ""]
        if fx.get("unknown_by_overlap"):
            lines += ["Other-fault flag rate by overlap:", ""]
            for d, b in fx["unknown_by_overlap"].items():
                lines.append(f"- {d}: " + ", ".join(
                    f"ov{o}={b[str(o)]['flag_rate']:.2f}" for o in range(k + 1)
                    if b.get(str(o), {}).get("n")))
            lines.append("")
    nm = extra.get("nmsweep")
    if nm:
        kp = f"P{k - 1}"
        cs = sorted(nm, key=lambda key: int(key.split("=")[1]))
        ms = list(dict.fromkeys(m for key in cs for m in nm[key]))
        lines += ["## Recipe curve: near-miss (all-but-one) negatives per type", "",
                  f"Same normal-vs-fault training set; only the number of near-miss negatives changes. "
                  f"Cells: {kp} flagged / CAP flagged.", "",
                  "| Near-misses per type | " + " | ".join(ms) + " |", "|---:|" + "---:|" * len(ms)]
        for key in cs:
            lines.append(f"| {key.split('=')[1]} | " + " | ".join(
                f"{nm[key][m][kp]:.2f} / {nm[key][m]['CAP']:.2f}" if m in nm[key] else "–" for m in ms) + " |")
        lines.append("")
    sc = extra.get("scale")
    if sc:
        kp = f"P{k - 1}"
        keys = sorted(sc, key=lambda key: float(key.split("=")[1]))
        ms = list(dict.fromkeys(m for key in keys for m in sc[key]))
        lines += ["## Signal-strength sweep (fault deviation x s, train and test)", "",
                  f"Cells: CAP / {kp} / {kp} predicted from CAP by the law (equal channels).", "",
                  "| Scale | " + " | ".join(ms) + " |", "|---:|" + "---:|" * len(ms)]
        for key in keys:
            cells = []
            for m in ms:
                c = sc[key].get(m)
                cells.append(f"{c['CAP']:.2f} / {c[kp]:.2f} / "
                             f"{L.predict_from_detection(c['CAP'], (k - 1) / k):.2f}" if c else "–")
            lines.append(f"| {key.split('=')[1]} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))


def make_figures(out, main, sweep, fix, extra=None):
    extra = extra or {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = main["k"]
    R = main["rates"]
    present = next(iter(R.values()))
    cols = [c for c in ["CAP"] + [f"P{m}" for m in range(1, k)] + EXTRA_KINDS if c in present]
    M = np.array([[R[m][c] for c in cols] for m in R])
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3, 0.5 * len(R) + 1.5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(R)), list(R))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction flagged")
    ax.set_title("Flagged as target fault (threshold: 1% false alarms on normal)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_flag_rates.png"), dpi=150)
    plt.close(fig)

    if main.get("unknown_by_overlap"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for d, b in main["unknown_by_overlap"].items():
            xs = [o for o in range(k + 1) if b.get(str(o), {}).get("n")]
            ax.plot(xs, [b[str(o)]["flag_rate"] for o in xs], marker="o", label=d)
        ax.set_xlabel(f"signature channels moved by the unknown fault (of {k})")
        ax.set_ylabel("fraction flagged as target")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(k + 1))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title("Real unknown faults vs. signature overlap")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_unknown_overlap.png"), dpi=150)
        plt.close(fig)

    if sweep:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, mk in zip(SWEEP_MODELS, "os^v"):
            pts = [(int(kk.split("|")[0][2:]), float(kk.split("|")[1][5:]), c[m]["partial_FA"])
                   for kk, c in sweep.items() if m in c]
            if not pts:
                continue
            bal = sorted((n, v) for n, p, v in pts if abs(p - 0.5) < 1e-9)
            line, = ax.plot([n for n, _ in bal], [v for _, v in bal], marker=mk, label=f"{m} · 50/50")
            ax.scatter([n for n, p, v in pts if abs(p - 0.5) > 1e-9],
                       [v for n, p, v in pts if abs(p - 0.5) > 1e-9], marker="x", color=line.get_color())
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("training windows (log)")
        ax.set_ylabel(f"false alarms on unseen P{k - 1} fault")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title("x = 10% / 90% target prevalence")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_sweep.png"), dpi=150)
        plt.close(fig)

    if fix:
        kp = f"P{k - 1}"
        ms = [m for m in fix["rates"] if m in R]
        x = np.arange(len(ms))
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col in zip(axs, ["CAP", kp]):
            ax.bar(x - 0.2, [R[m][col] for m in ms], 0.4, label="normal vs fault only")
            ax.bar(x + 0.2, [fix["rates"][m][col] for m in ms], 0.4, label="+ all-but-one negatives")
            ax.set_xticks(x, [m.split(" (")[0] for m in ms], rotation=20)
            ax.set_ylim(0, 1)
            ax.set_title(f"{col} flagged")
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_targeted_negatives.png"), dpi=150)
        plt.close(fig)

    kp = f"P{k - 1}"
    nm = extra.get("nmsweep")
    if nm:
        cs = sorted(int(key.split("=")[1]) for key in nm)
        ms = list(dict.fromkeys(m for c in cs for m in nm[f"per_type={c}"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ms:
            pts = [(c, nm[f"per_type={c}"][m][kp]) for c in cs if m in nm[f"per_type={c}"]]
            ax.plot([max(c, 0.5) for c, _ in pts], [v for _, v in pts], marker="o", label=m)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("near-miss (all-but-one) negatives per type (0 plotted at 0.5)")
        ax.set_ylabel(f"{kp} flagged as target")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title("Recipe curve")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_recipe_curve.png"), dpi=150)
        plt.close(fig)
    sc = extra.get("scale")
    if sc:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        b = np.linspace(0.02, 0.9999, 200)
        for m_ in range(1, k):
            ax.plot(b, [L.predict_from_detection(v, m_ / k) for v in b], color="0.6", lw=1,
                    ls="-" if m_ == k - 1 else "--")
            ax.text(0.03, L.predict_from_detection(0.9999, m_ / k), f"law, P{m_}", fontsize=7,
                    color="0.4", va="bottom")
        ms = list(dict.fromkeys(m for c in sc.values() for m in c))
        for m, mk in zip(ms, "os^vDP"):
            pts = sorted((float(key.split("=")[1]), c[m]["CAP"], c[m][kp]) for key, c in sc.items() if m in c)
            ax.plot([p_[1] for p_ in pts], [p_[2] for p_ in pts], marker=mk, label=m)
        ax.set_xlabel("detection rate of the target fault (CAP)")
        ax.set_ylabel(f"{kp} flagged as target")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title("Detectability paradox: signal-strength sweep")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_paradox.png"), dpi=150)
        plt.close(fig)


DEFAULT_STEPS = ("main", "ident", "sweep", "fix", "fixtime", "fixboth", "fixhelly")
THEORY_STEPS = ("nmsweep", "scale")


def run_suite(src, out, cfg, steps=DEFAULT_STEPS):
    # fail early (not mid-run) if a split has no windows where the target fault is present
    for split in ("train", "test"):
        if len(src.construct(split, "CAP", 5, np.random.default_rng(0))) == 0:
            raise SystemExit(
                f"No windows in the '{split}' split where the target's signature is present. "
                "The data is too short for this target or split: try a smaller --block, "
                "a lower --min-dev, another --target, or a dataset with more runs.")
    os.makedirs(out, exist_ok=True)
    _save(os.path.join(out, "config.json"), cfg)
    main = step_main(src, cfg, out) if "main" in steps else _load(os.path.join(out, "main.json"))
    ident = step_ident(src, cfg, out) if "ident" in steps else {}
    sweep = step_sweep(src, cfg, out) if "sweep" in steps else {}
    fix = step_fix(src, cfg, out) if "fix" in steps else {}
    has_time = "ASYNC" in getattr(src, "test_kinds", EXTRA_KINDS)  # tabular sources have no time axis
    fixes = {}
    for name in FIX_STEPS:
        if name in steps and has_time:
            fixes[name] = step_fix(src, cfg, out, name)
        elif os.path.exists(os.path.join(out, f"{name}.json")):  # keep earlier results in the report
            fixes[name] = _load(os.path.join(out, f"{name}.json"))
    extra = {}
    for name, fn in (("nmsweep", step_nmsweep), ("scale", step_scale)):
        if name in steps:
            extra[name] = fn(src, cfg, out)
        elif os.path.exists(os.path.join(out, f"{name}.json")):
            extra[name] = _load(os.path.join(out, f"{name}.json"))
    write_report(src.name, out, main, ident, sweep, fix, cfg, fixes, extra)
    make_figures(out, main, sweep, fix, extra)
    k = main["k"]
    present = next(iter(main["rates"].values()))
    cols = [c for c in ["NORMAL", "CAP"] + [f"P{m}" for m in range(1, k)] + EXTRA_KINDS if c in present]
    print("\n" + L.fmt_table(main["rates"], cols))
    print("\nHypothesis checks:")
    for name, v, ev in hypothesis_checks(main, ident, sweep, fix, fixes, extra):
        print(f"  [{v}] {name}\n      {ev}")
    print(f"\nReport: {os.path.join(out, 'REPORT.md')}")
    return main, ident, sweep, fix
