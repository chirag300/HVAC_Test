"""
make_paper_assets.py -- figures and LaTeX tables for the paper.

Inputs: paper/synth_out/results.json (paper_synth.py), theory/checks_out/results.json
(theory_checks.py), results/paper_toy/seed*/ (run_toy.py), and the real-data numbers below,
copied from the REPORT.md files of the LBNL runs (one seed each).
Outputs: <out>/figs/fig_main.pdf, <out>/tables/*.tex
    python paper/make_paper_assets.py --out ../paper
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
Z = norm.ppf(0.99)
DARK, MAG, LIGHT, GOLD, GREY = "#3F0A6A", "#AE3ED3", "#BCA5CC", "#C98A12", "#8C8C8C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8})


def law(beta, r):
    return norm.cdf(r * (norm.ppf(np.clip(beta, 1e-6, 1 - 1e-6)) + Z) - Z)


# ----------------------------------------------------------------- real data (single seed, from REPORT.md)
REAL_HELLY = {  # PAIRSTAG flag rate (LSTM): no timing negatives / fully staggered / CIS
    "fan\nclean": (0.92, 0.51, 0.00), "fan\nreal": (0.99, 0.37, 0.00), "valve": (0.96, 0.59, 0.06)}
REAL_RECIPE = {  # LSTM wd=1e-3, 2-of-3 flag rate vs near-misses per type
    "counts": [0, 3, 10, 30, 100, 300, 1000],
    "FCU fan (clean)": [0.96, 0.89, 0.75, 0.57, 0.26, 0.06, 0.01],
    "FCU fan (real)": [1.00, 1.00, 0.69, 0.65, 0.41, 0.14, 0.00],
    "FCU valve": [1.00, 1.00, 0.89, 0.72, 0.51, 0.11, 0.09],
    "AHU coil valve": [0.81, 0.78, 0.84, 0.68, 0.60, 0.49, 0.50]}
REAL_RECIPE_PLAIN = {  # LSTM without weight decay
    "FCU fan (clean)": [0.95, 0.92, 0.82, 0.62, 0.54, 0.44, 0.08],
    "FCU fan (real)": [0.99, 1.00, 0.78, 0.74, 0.55, 0.32, 0.12],
    "FCU valve": [1.00, 1.00, 1.00, 0.86, 0.81, 0.64, 0.06],
    "AHU coil valve": [0.74, 0.79, 0.83, 0.67, 0.53, 0.54, 0.47]}
AHU_SCALE = {  # capped CI, AHU coil valve: scale -> (detection, 2-of-3)
    0.05: (0.06, 0.03), 0.1: (0.37, 0.19), 0.25: (0.66, 0.29), 0.5: (0.95, 0.47), 1.0: (1.00, 0.90)}


def load_toy():
    seeds = sorted(glob.glob(os.path.join(ROOT, "results", "paper_toy", "seed*")))
    out = {}
    for sd in seeds:
        for step in ("main", "fix", "fixtime", "fixhelly", "fixoe"):
            p = os.path.join(sd, f"{step}.json")
            if os.path.exists(p):
                out.setdefault(step, []).append(json.load(open(p))["rates"])
    return out, len(seeds)


def fig_main(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    # (a) power-specificity curve
    ax = axs[0]
    lo = 1e-4
    miss = np.logspace(np.log10(0.99), np.log10(lo), 300)
    ax.plot(miss, law(1 - miss, 2 / 3), color=DARK, lw=1.3, label="law, 2 of 3")
    ax.plot(miss, law(1 - miss, 1 / 3), color=MAG, lw=1.3, ls="--", label="law, 1 of 3")
    for model, col, lab in (("logistic", DARK, "GD logistic"), ("mlp", GOLD, "GD 2-layer ReLU")):
        rows = S["S1"][model]
        m = [max(1 - r["cap"][0], lo) for r in rows]
        ax.scatter(m, [r["p2"][0] for r in rows], s=12, color=col, marker="o", zorder=3, label=lab)
        ax.scatter(m, [r["p1"][0] for r in rows], s=12, color=col, marker="s", zorder=3)
    am = [max(1 - v[0], lo) for v in AHU_SCALE.values()]
    ax.scatter(am, [v[1] for v in AHU_SCALE.values()], s=40, marker="*", color=MAG, edgecolor="k", lw=0.3,
               zorder=4, label="AHU real, capped CI")
    ax.set_xscale("log")
    ax.set_xlim(1.0, lo * 0.8)
    ax.set_xticks([1, 0.1, 0.01, 1e-3, 1e-4], ["100%", "10%", "1%", "0.1%", "0.01%"])
    ax.set_xlabel("target miss rate (1% false alarms)")
    ax.set_ylabel("look-alike flagged")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
    ax.set_title("(a) law and paradox")
    # (b) Helly
    ax = axs[1]
    syn = S["S3"]
    groups = ["Synth."] + list(REAL_HELLY)
    vals = [[syn["normal vs fault"]["pair"][0], syn["+ fully staggered"]["pair"][0], syn["+ pair-staggered"]["pair"][0]]]
    errs = [[syn["normal vs fault"]["pair"][1], syn["+ fully staggered"]["pair"][1], syn["+ pair-staggered"]["pair"][1]]]
    vals += [list(v) for v in REAL_HELLY.values()]
    errs += [[0, 0, 0]] * len(REAL_HELLY)
    x = np.arange(len(groups))
    w = 0.26
    for i, (lab, col) in enumerate(zip(("no timing negatives", "fully staggered", "pair-staggered (CIS)"),
                                       (GREY, LIGHT, DARK))):
        ax.bar(x + (i - 1) * w, [v[i] for v in vals], w, color=col, label=lab)
        ax.errorbar([(i - 1) * w], [vals[0][i]], yerr=[errs[0][i]], fmt="none", ecolor="k", lw=0.7, capsize=1.5)
    ax.set_xticks(x, groups, fontsize=7.5)
    ax.axvline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("synthetic | real FCU runs")
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("pair-staggered flagged")
    ax.legend(frameon=False, ncol=1, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    ax.set_title("(b) timing negatives")
    # (c) recipe (real)
    ax = axs[2]
    cs = [max(c, 1) for c in REAL_RECIPE["counts"]]
    for (name, ys), col, mk in zip([(k, v) for k, v in REAL_RECIPE.items() if k != "counts"],
                                   (DARK, MAG, GOLD, GREY), "os^D"):
        ax.plot(cs, ys, marker=mk, ms=3, lw=1.1, color=col, label=name, ls="--" if "AHU" in name else "-")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["0", "3", "10", "30", "100", "300", "1k"])
    ax.set_xlabel("channel near-misses per type")
    ax.set_ylabel("2-of-3 look-alike flagged")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.1)
    ax.set_title("(c) examples needed (real)")
    fig.tight_layout(w_pad=0.8)
    os.makedirs(os.path.join(out, "figs"), exist_ok=True)
    fig.savefig(os.path.join(out, "figs", "fig_main.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "figs", "fig_main.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def pm(vals, d=2):
    v = np.asarray(vals, float)
    if len(v) > 1:
        return f"{v.mean():.{d}f}$\\pm${v.std(ddof=1):.{d}f}"
    return f"{v.mean():.{d}f}"


def tab_toy(T, nseeds, out):
    cols = ["CAP", "P1", "P2", "BIG1", "ASYNC", "PAIRSTAG"]
    heads = ["Detection", "1-of-3", "2-of-3", "BIG1", "Fully stag.", "Pair-stag."]
    rows = [("Normal vs.\\ target only", "main"), ("+ fully staggered negatives", "fixtime"),
            ("+ generic negatives (OE-style)", "fixoe"), ("+ channel near-misses", "fix"),
            ("+ CIS (channel + pair-staggered)", "fixhelly")]
    m = "LSTM (raw series)"
    lines = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Training & " + " & ".join(heads) + "\\\\", "\\midrule"]
    for lab, step in rows:
        rs = T.get(step, [])
        lines.append(lab + " & " + " & ".join(pm([r[m][c] for r in rs]) for c in cols) + "\\\\")
    rs = T["main"]
    lines.append("\\midrule")
    lines.append("Reference all-sensors rule & " + " & ".join(pm([r["AND rule (true)"][c] for r in rs]) for c in cols)
                 + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    open(os.path.join(out, "tables", "tab_toy.tex"), "w").write("\n".join(lines) + "\n")


def tab_appendix(S, C, T, nseeds, out):
    L = []
    # law table
    L += ["\\begin{table}[h]\\centering\\small",
          "\\caption{Synthetic law check (S1; mean$\\pm$s.d., 5 seeds). \\emph{Law}: \\cref{cor:paradox} evaluated at the "
          "model's own measured detection rate (no fitting).}\\label{tab:law}",
          "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llccccc}\\toprule",
          "Model & $\\delta/\\sigma$ & Detection & 2-of-3 & law & 1-of-3 & law\\\\\\midrule"]
    for model, name in (("logistic", "GD logistic"), ("mlp", "GD 2-layer ReLU")):
        for r in S["S1"][model]:
            L.append(f"{name} & {r['delta']:.1f} & {r['cap'][0]:.3f}$\\pm${r['cap'][1]:.3f} & "
                     f"{r['p2'][0]:.3f}$\\pm${r['p2'][1]:.3f} & {r['p2_law'][0]:.3f} & "
                     f"{r['p1'][0]:.3f}$\\pm${r['p1'][1]:.3f} & {r['p1_law'][0]:.3f}\\\\")
            name = ""
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    # composite
    L += ["\\begin{table}[h]\\centering\\small",
          "\\caption{Composite null (C3, $k=3$, $4\\times10^5$ draws). \\emph{Size}: worst false-alarm rate over "
          "look-alikes, with shifts bounded by $\\delta$ or unbounded ($50\\delta$). Additive detectors are calibrated "
          "either on normal data or for the bounded null (\\cref{thm:composite}(c)).}\\label{tab:composite}",
          "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{ccccccccc}\\toprule",
          " & \\multicolumn{3}{c}{Additive, normal-calibrated} & \\multicolumn{2}{c}{Additive, bounded-null} & "
          "\\multicolumn{2}{c}{Intersection-union} & Bound\\\\",
          "$\\delta/\\sigma$ & size ($\\le\\delta$) & size ($\\infty$) & power & size ($\\infty$) & power & size ($\\infty$)"
          " & power (pred.) & (a)\\\\\\midrule"]
    for r in C["C3"]:
        L.append(f"{r['delta']:.0f} & {r['a1'][0]:.3f} & {r['a1'][1]:.3f} & {r['a1'][2]:.3f} & {r['a2'][1]:.3f} & "
                 f"{r['a2'][2]:.3f} & {r['iu'][0]:.3f} & {r['iu'][1]:.3f} ({r['iu'][2]:.3f}) & {r['bound']:.3f}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    # separated
    L += ["\\begin{table}[h]\\centering\\small",
          "\\caption{Separated regime (S2; mean$\\pm$s.d., 3 seeds). Left block: normal-vs-target training. "
          "\\emph{Position}: $p(x^{(j)})$ of channel near-misses at the prototypes. \\emph{Bound}: $1-(R/R_{\\min})/\\sqrt k$, "
          "a valid lower bound on the position because $\\kappa\\le R/R_{\\min}$ (\\cref{lem:lip}). \\emph{Flagged}: near-miss "
          "flag rate at 1\\% false alarms. Right block: retrained with 100 near-misses per type. \\emph{Norm ratio}: "
          "$R/G$ after over $R/G$ before. \\cref{thm:sep}(b) requires $R/R_{\\min}\\ge k\\gamma/G$ with $\\gamma$ the "
          "smallest target-minus-near-miss score gap; it held in every run. Detection after retraining is 1.000 throughout.}"
          "\\label{tab:sep}",
          "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{cccccc|cccc}\\toprule",
          "$k$ & linear $\\frac{k-1}{k}$ & position & bound & $R/R_{\\min}$ & flagged & norm ratio & $R/R_{\\min}$ & "
          "$k\\gamma/G$ & flagged\\\\\\midrule"]
    for r in S["S2"]:
        L.append(f"{r['k']} & {r['linear']:.3f} & {r['pos'][0]:.3f}$\\pm${r['pos'][1]:.3f} & {r['bound'][0]:.3f} & "
                 f"{r['kappa'][0]:.2f} & {r['flag'][0]:.3f} & {r['ratio'][0]:.2f}$\\pm${r['ratio'][1]:.2f} & "
                 f"{r['rnorm_after'][0]:.2f} & {r['kgamma_after'][0]:.2f} & {r['flag_after'][0]:.3f}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    # recipe
    L += ["\\begin{table}[h]\\centering\\small",
          "\\caption{Recipe curves: 2-of-3 look-alike flag rate vs.\\ channel near-misses per type. Top: synthetic "
          "two-layer ReLU (S4; mean$\\pm$s.d., 3 seeds). Bottom: real data, LSTM, one seed (detection $\\ge0.99$ "
          "throughout).}\\label{tab:recipe}",
          "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{l" + "c" * 7 + "}\\toprule"]
    s4 = S["S4"]
    cs = [r["c"] for r in s4["0.0"]]
    L.append("Synthetic, per type & " + " & ".join(str(c) for c in cs) + " & & \\\\\\midrule")
    for wd in ("0.0", "0.001"):
        L.append(f"weight decay {float(wd):g} & " + " & ".join(f"{r['p2'][0]:.2f}$\\pm${r['p2'][1]:.2f}" for r in s4[wd])
                 + " & & \\\\")
    L.append("\\midrule Real, per type & " + " & ".join(str(c) for c in REAL_RECIPE["counts"]) + "\\\\\\midrule")
    for name in ("FCU fan (clean)", "FCU fan (real)", "FCU valve", "AHU coil valve"):
        L.append(f"{name}, wd $10^{{-3}}$ & " + " & ".join(f"{v:.2f}" for v in REAL_RECIPE[name]) + "\\\\")
        L.append(f"{name}, wd 0 & " + " & ".join(f"{v:.2f}" for v in REAL_RECIPE_PLAIN[name]) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    # toy, all models, with CAP-only law predictions
    cols = ["CAP", "P1", "P2", "BIG1", "ASYNC", "PAIRSTAG", "UP", "OFF"]
    L += ["\\begin{table}[h]\\centering\\small",
          f"\\caption{{Toy replica, all models, normal-vs-target training (mean$\\pm$s.d., {nseeds} seeds). "
          "Columns \\emph{law} give \\cref{cor:paradox} at each model's mean detection rate with equal shares "
          "($r=m/k$).}\\label{tab:toyall}",
          "\\setlength{\\tabcolsep}{2.5pt}\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * (len(cols) + 2) + "}\\toprule",
          "Model & Detection & 1-of-3 & law & 2-of-3 & law & BIG1 & Fully st. & Pair-st. & UP & OFF\\\\\\midrule"]
    for m in T["main"][0]:
        rs = T["main"]
        cap = np.mean([r[m]["CAP"] for r in rs])
        cells = [pm([r[m]["CAP"] for r in rs]), pm([r[m]["P1"] for r in rs]), f"{law(cap, 1/3):.2f}",
                 pm([r[m]["P2"] for r in rs]), f"{law(cap, 2/3):.2f}"] + \
                [pm([r[m][c] for r in rs]) for c in ("BIG1", "ASYNC", "PAIRSTAG", "UP", "OFF")]
        L.append(m.replace(" (raw series)", "").replace(" (stats)", " (stats)").replace("AND rule (true)", "Reference rule")
                 + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}}\\end{table}", ""]
    # toy remedies, other models
    L += ["\\begin{table}[h]\\centering\\small",
          f"\\caption{{Toy replica: 2-of-3 / pair-staggered flag rates per model and remedy (mean over {nseeds} seeds).}}"
          "\\label{tab:toyremedy}",
          "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccccc}\\toprule",
          "Model & normal only & + fully stag. & + generic (OE) & + channel NM & + CIS\\\\\\midrule"]
    for m in T["main"][0]:
        if m == "AND rule (true)":
            continue
        cells = []
        for step in ("main", "fixtime", "fixoe", "fix", "fixhelly"):
            rs = T.get(step, [])
            cells.append(f"{np.mean([r[m]['P2'] for r in rs]):.2f} / {np.mean([r[m]['PAIRSTAG'] for r in rs]):.2f}")
        L.append(m.replace(" (raw series)", "") + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    L.append(REAL_TABLES)
    open(os.path.join(out, "tables", "tab_appendix.tex"), "w").write("\n".join(L) + "\n")


REAL_TABLES = r"""
\begin{table}[h]\centering\small
\caption{Real data (LBNL), normal-vs-target training, all models, one seed. Clean mode. \emph{Pairwise law}: 2-of-3 rate predicted from the detection rate and the single-sensor rates (each sensor's share $r_j$ read from its single-sensor flag rate), a test of additivity. FCU values saturate near 1; the AHU coil is informative.}\label{tab:realall}
\setlength{\tabcolsep}{2.4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
Target & Model & Det. & 1-of-3 & 2-of-3 & pairwise law & BIG1 & Fully st. & Pair-st. & UP & OFF\\
\midrule
FCU fan & LogReg (stats) & 1.00 & 0.81 & 1.00 & 1.00 & 0.98 & 0.95 & 0.97 & 0.00 & 0.22\\
 & Random forest & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 & 1.00 & 0.98 & 0.33\\
 & Grad.\ boosting & 1.00 & 0.65 & 1.00 & 1.00 & 0.65 & 0.94 & 0.94 & 0.13 & 0.07\\
 & MLP & 0.99 & 0.69 & 0.95 & 1.00 & 0.96 & 0.76 & 0.87 & 0.00 & 0.17\\
 & LSTM & 1.00 & 0.79 & 0.97 & 1.00 & 0.98 & 0.85 & 0.92 & 0.48 & 0.61\\
 & Channel-indep. & 1.00 & 0.88 & 0.99 & 1.00 & 1.00 & 0.95 & 0.95 & 0.00 & 0.00\\
 & Capped CI & 1.00 & 0.96 & 1.00 & 1.00 & 1.00 & 0.99 & 0.98 & 0.41 & 0.14\\
 & Reference rule & 1.00 & 0.08 & 0.35 & 0.25 & 0.06 & 0.33 & 0.82 & 0.01 & 0.19\\
\midrule
AHU coil & LogReg (stats) & 1.00 & 0.76 & 1.00 & 1.00 & 1.00 & 0.97 & 1.00 & 0.00 & 0.01\\
 & Random forest & 1.00 & 0.66 & 0.98 & 1.00 & 0.81 & 0.86 & 0.89 & 0.05 & 0.01\\
 & Grad.\ boosting & 1.00 & 0.45 & 0.81 & 0.98 & 0.52 & 0.57 & 0.54 & 0.03 & 0.00\\
 & MLP & 1.00 & 0.39 & 0.79 & 0.81 & 0.63 & 0.40 & 0.71 & 0.00 & 0.01\\
 & LSTM & 1.00 & 0.38 & 0.78 & 0.63 & 0.58 & 0.30 & 0.73 & 0.00 & 0.01\\
 & Channel-indep. & 1.00 & 0.59 & 0.93 & 0.96 & 0.76 & 0.81 & 0.95 & 0.00 & 0.00\\
 & Capped CI & 1.00 & 0.47 & 0.96 & 0.98 & 0.41 & 0.31 & 0.50 & 0.01 & 0.06\\
 & Reference rule & 0.97 & 0.37 & 0.74 & 0.78 & 0.38 & 0.74 & 0.90 & 0.01 & 0.01\\
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[h]\centering\small
\caption{Real never-seen faults (LSTM vs.\ reference rule), flag rate by the number of signature sensors the fault moves in the fault direction; one seed.}\label{tab:realov}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
Target, detector & 0 of 3 & 1 of 3 & 2 of 3 & 3 of 3\\
\midrule
FCU fan (clean), LSTM & 0.22 & 0.63 & 0.79 & 0.93\\
FCU fan (clean), reference rule & 0.06 & 0.13 & 0.36 & 0.44\\
FCU fan (clean), LSTM + channel near-misses & 0.15 & 0.11 & 0.22 & 0.80\\
FCU valve, LSTM & 0.18 & 0.59 & 0.65 & 1.00\\
FCU valve, reference rule & 0.05 & 0.06 & 0.23 & 0.19\\
AHU coil valve, LSTM & 0.00 & 0.04 & 0.01 & --\\
\bottomrule
\end{tabular}
\end{table}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "paper"))
    a = ap.parse_args()
    S = json.load(open(os.path.join(HERE, "synth_out", "results.json")))
    C = json.load(open(os.path.join(ROOT, "theory", "checks_out", "results.json")))
    T, ns = load_toy()
    fig_main(S, a.out)
    tab_toy(T, ns, a.out)
    tab_appendix(S, C, T, ns, a.out)
    print("assets written to", a.out, "toy seeds:", ns)
