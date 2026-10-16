"""
make_figures.py -- figures for theory/theory.tex from theory/checks_out/results.json.

    python theory/make_figures.py          # writes theory/figs/*.pdf and *.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "checks_out", "results.json")))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
P_DARK, P_MAG, P_LIGHT, GOLD, GREY = "#3F0A6A", "#AE3ED3", "#BCA5CC", "#C98A12", "#8C8C8C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold"})
Z = norm.ppf(0.99)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def law(beta, r):
    return norm.cdf(r * (norm.ppf(beta) + Z) - Z)


# ---------------------------------------------------------------- 1. trade-off / paradox
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    lo = 1e-4                       # x = miss rate 1 - beta, log scale, reversed
    miss = np.logspace(np.log10(0.98), np.log10(lo), 400)
    k = 3
    for m, ls, col in ((2, "-", P_DARK), (1, "--", P_MAG)):
        ax.plot(miss, law(1 - miss, m / k), ls=ls, color=col, lw=1.6, label=f"law: {m} of {k} channels")
    c2 = R.get("C2", {}).get("rows", [])
    mx = lambda pw: max(1 - pw, lo)
    if c2:
        ax.scatter([mx(r["pw"]) for r in c2], [r["p2"] for r in c2], s=18, color=P_DARK, marker="o",
                   zorder=3, label="GD logistic reg. (signal sweep)")
        ax.scatter([mx(r["pw"]) for r in c2], [r["p1"] for r in c2], s=18, color=P_MAG, marker="o", zorder=3)
    slide = {"LogReg": 0.99, "Random forest": 0.99, "Grad. boosting": 0.99, "MLP": 0.98, "LSTM": 0.98}
    c8 = R.get("C8", {})
    for i, (m, v) in enumerate(c8.items()):
        cap = slide.get(m)
        if cap is None:
            continue
        jitter = 1 + (i - 2) * 0.04
        ax.scatter([(1 - cap) * jitter], [v[3]], marker="*", s=70, color=GOLD, edgecolor="k", lw=0.4,
                   zorder=4, label="toy study, slide 12 (5 models)" if i == 0 else None)
        ax.scatter([(1 - cap) * jitter], [v[1]], marker="*", s=70, color=GOLD, edgecolor="k", lw=0.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1.0, lo * 0.8)
    ax.set_xticks([1, 0.1, 0.01, 1e-3, 1e-4], ["100%", "10%", "1%", "0.1%", "0.01%"])
    ax.set_xlabel("miss rate of the target fault (log scale; false alarms on normal fixed at 1%)")
    ax.set_ylabel("partial-match fault flagged")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title("Detectability paradox (k = 3)")
    save(fig, "fig_tradeoff")


# ---------------------------------------------------------------- 2. composite null
def fig_composite():
    rows = R.get("C3", [])
    if not rows:
        return
    d = [r["delta"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axs[0]
    labels = ["add.\nnormal\ncal.", "add.\noverlap\ncal.", "inters.-\nunion", "capped"]
    sizes = [max(r["a1"][1] for r in rows), max(r["a2"][1] for r in rows),
             max(r["iu"][0] for r in rows), max(r["cp"][0] for r in rows)]
    cols = [GREY, P_LIGHT, P_DARK, P_MAG]
    ax.bar(range(4), sizes, color=cols)
    ax.axhline(0.01, color="k", lw=0.8, ls=":")
    for i, v in enumerate(sizes):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4), labels, fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("worst false-alarm rate over\npartial faults (unbounded size)")
    ax.set_title("Validity (budget 1%, dotted)")
    ax = axs[1]
    ax.plot(d, [r["bound"] for r in rows], color="k", ls=":", lw=1.2, label="upper bound (any valid test)")
    ax.plot(d, [r["iu"][1] for r in rows], color=P_DARK, marker="o", label="intersection-union (valid)")
    ax.plot(d, [r["cp"][1] for r in rows], color=P_MAG, marker="s", label="capped evidence (valid)")
    ax.plot(d, [r["a2"][2] for r in rows], color=P_LIGHT, marker="^",
            label="additive, valid only if overlap bounded")
    ax.set_xlabel("per-channel shift δ/σ")
    ax.set_ylabel("detection rate of the target")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("Power (k = 3)")
    save(fig, "fig_composite")


# ---------------------------------------------------------------- 3. separated regime
def fig_separated():
    rows = R.get("C4", [])
    if not rows:
        return
    k = np.array([r["k"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axs[0]
    ax.plot(k, [r["pos"] for r in rows], color=P_DARK, marker="o", label="measured (GD, 2-layer ReLU)")
    ax.plot(k, [r["linear"] for r in rows], color=P_MAG, ls="--", label="linear max-margin (k−1)/k")
    ax.plot(k, [r["bound"] for r in rows], color=GREY, ls=":", label="bound 1 − κ/√k")
    ax.set_xlabel("signature size k")
    ax.set_ylabel("near-miss position p")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Near-misses sit next to the fault")
    ax = axs[1]
    ax.plot(k, [r["ratio"] for r in rows], color=P_DARK, marker="o", label="measured norm ratio")
    ax.plot(k, np.sqrt(k), color=GREY, ls=":", label="√k")
    ax.plot(k, k / 2, color=P_MAG, ls="--", label="k/2 (Thm, g = G/2)")
    ax.set_xlabel("signature size k")
    ax.set_ylabel("path norm with / without near-misses")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Rejecting near-misses costs norm")
    save(fig, "fig_separated")


# ---------------------------------------------------------------- 4. recipe curve
def fig_recipe():
    c5, counts = R.get("C5"), R.get("C5_counts")
    if not c5:
        return
    x = [max(c, 0.5) for c in counts]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for (wd, rows), col, mk in zip(c5.items(), (P_DARK, P_MAG, GOLD), "os^"):
        ax.plot(x, [r[0] for r in rows], color=col, marker=mk, label=f"weight decay {float(wd):g}")
    ax.set_xscale("log")
    ax.set_xticks(x, ["0"] + [str(c) for c in counts[1:]])
    ax.set_xlabel("near-miss negatives per type (k = 3 types)")
    ax.set_ylabel("2-of-3 partial fault flagged")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Recipe curve: supervision vs. regularization")
    save(fig, "fig_recipe")


# ---------------------------------------------------------------- 5. Helly
def fig_helly():
    raw = R.get("C6", {}).get("raw")
    if not raw:
        return
    conds = list(raw)
    tests = ["target", "fully staggered", "pair-staggered"]
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    w = 0.26
    for j, (t, col) in enumerate(zip(tests, (P_DARK, P_LIGHT, P_MAG))):
        vals = [raw[c][j] for c in conds]
        ax.bar(np.arange(len(conds)) + (j - 1) * w, vals, w, color=col, label=t)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(conds)), ["normal vs fault", "+ fully staggered\nnegatives",
                                      "+ pair-staggered\nnegatives"], fontsize=7.5)
    ax.set_ylabel("flagged as target")
    ax.set_ylim(0, 1.32)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center")
    ax.set_title("Which timing negatives teach 'simultaneous'")
    save(fig, "fig_helly")


# ---------------------------------------------------------------- 6. p*(lambda)
def fig_pstar():
    c9 = R.get("C9")
    if not c9:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for (k, rows), col in zip(sorted(c9.items(), key=lambda kv: int(kv[0])), (P_DARK, P_MAG, GOLD, GREY)):
        lam = [r[0] for r in rows]
        ax.plot(lam, [max(r[1], 1e-5) for r in rows], marker="o", color=col, label=f"k = {k}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weight decay λ")
    ax.set_ylabel("necessary near-miss fraction p*")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Below p*, every minimizer flags a near-miss")
    save(fig, "fig_pstar")


if __name__ == "__main__":
    for f in (fig_tradeoff, fig_composite, fig_separated, fig_recipe, fig_helly, fig_pstar):
        f()
