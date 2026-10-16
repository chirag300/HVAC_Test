"""
theory_checks.py -- numerical checks of every theorem in theory/theory.pdf.

Each check simulates the exact setting of one result, trains the relevant learner with
gradient descent where the result is about gradient descent, and prints prediction vs.
measurement. Runtime: a few minutes on a laptop CPU. NumPy + SciPy only (scikit-learn is
used only for the window-statistics logistic regression in C6).

    python theory/theory_checks.py            # all checks, writes theory/checks_out/
    python theory/theory_checks.py C2 C4      # selected checks
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks_out")
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.01
Z = norm.ppf(1 - ALPHA)
RESULTS = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ learners
def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def logistic_gd(X, y, iters=3000, lr=0.5, wd=0.0):
    """Plain full-batch gradient descent on the logistic loss (linear model + bias)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w + b)
        g = p - y
        w -= lr * (X.T @ g / n + wd * w)
        b -= lr * g.mean()
    return lambda Z_: Z_ @ w + b


class TwoLayer:
    """f(x) = sum_i v_i relu(w_i.x + b_i) + c, trained by full-batch GD (or Adam) on logistic loss."""

    def __init__(self, d, h=64, init=1e-2, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, init, (h, d))
        self.b = np.zeros(h)
        self.v = r.normal(0, init, h)
        self.c = 0.0

    def __call__(self, X):
        return np.maximum(X @ self.W.T + self.b, 0) @ self.v + self.c

    def path_norm(self):
        return float(np.sum(np.abs(self.v) * np.linalg.norm(self.W, axis=1)))

    def fit(self, X, y, iters=4000, lr=0.05, wd=0.0, adam=True):
        yy = 2 * y - 1
        P = [self.W, self.b, self.v]
        m = [np.zeros_like(p) for p in P] + [0.0]
        s = [np.zeros_like(p) for p in P] + [0.0]
        for t in range(1, iters + 1):
            H = X @ self.W.T + self.b
            A = np.maximum(H, 0)
            f = A @ self.v + self.c
            g = -yy * sigmoid(-yy * f) / len(X)          # dL/df
            gv = A.T @ g + wd * self.v
            gc = g.sum()
            dH = np.outer(g, self.v) * (H > 0)
            gW = dH.T @ X + wd * self.W
            gb = dH.sum(0)
            grads = [gW, gb, gv, gc]
            if adam:
                for i, gr in enumerate(grads):
                    m[i] = 0.9 * m[i] + 0.1 * gr
                    s[i] = 0.999 * s[i] + 0.001 * gr * gr
                    step = lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(s[i] / (1 - 0.999 ** t)) + 1e-8)
                    if i < 3:
                        P[i] -= step
                    else:
                        self.c -= step
            else:
                self.W -= lr * gW
                self.b -= lr * gb
                self.v -= lr * gv
                self.c -= lr * gc
        return self


def calibrate(f, Xnorm):
    return np.quantile(f(Xnorm), 1 - ALPHA)


# ------------------------------------------------------------------ C1
def C1(rng):
    """Thm 1 + Thm 2: GD logistic regression converges to the additive NP detector; the
    partial-overlap false-alarm law holds and does not depend on n or class prevalence."""
    d, k, delta, sig = 20, 3, 1.5, 1.0
    S = np.arange(k)
    mu = np.zeros(d); mu[S] = delta
    D = delta * np.sqrt(k) / sig
    tests = {"CAP (all 3)": mu, "P2 (2 of 3)": np.r_[delta, delta, np.zeros(d - 2)],
             "P1 (1 of 3)": np.r_[delta, np.zeros(d - 1)], "BIG1 (one ×3)": np.r_[3 * delta, np.zeros(d - 1)],
             "UP (sign flipped)": -mu, "OFF (moved)": np.r_[np.zeros(k), delta, delta, delta, np.zeros(d - 2 * k)]}
    pred = {t: norm.cdf(m_ @ mu / (sig ** 2 * D) - Z) for t, m_ in tests.items()}
    Xn_cal = rng.normal(0, sig, (200000, d))
    rows = []
    for n in (200, 2000, 20000):
        for prev in (0.1, 0.5, 0.9):
            n1 = int(n * prev)
            X = np.r_[rng.normal(0, sig, (n - n1, d)), rng.normal(0, sig, (n1, d)) + mu]
            y = np.r_[np.zeros(n - n1), np.ones(n1)]
            f = logistic_gd(X, y)
            thr = calibrate(f, Xn_cal)
            meas = {t: float((f(rng.normal(0, sig, (20000, d)) + m_) > thr).mean()) for t, m_ in tests.items()}
            rows.append({"n": n, "prevalence": prev, **meas})
    # flexible model, overlap regime: cross-entropy consistency
    n = 20000
    X = np.r_[rng.normal(0, sig, (n // 2, d)), rng.normal(0, sig, (n // 2, d)) + mu]
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    net = TwoLayer(d, h=64, init=0.1, seed=1).fit(X, y, iters=1500, lr=0.01)
    thr = calibrate(net, Xn_cal)
    mlp = {t: float((net(rng.normal(0, sig, (20000, d)) + m_) > thr).mean()) for t, m_ in tests.items()}
    Dh = norm.ppf(mlp["CAP (all 3)"]) + Z
    mlp_tradeoff = {"P2 (2 of 3)": norm.cdf(2 * Dh / 3 - Z), "P1 (1 of 3)": norm.cdf(Dh / 3 - Z)}
    log("\nC1  Partial-overlap false-alarm law (k=3, δ/σ=1.5 per channel, α=1%)")
    log(f"{'test':20s} {'predicted':>9s} " + " ".join(f"n={r['n']},π={r['prevalence']}"[:14].rjust(14) for r in rows[::4]) + "   2-layer ReLU")
    for t in tests:
        log(f"{t:20s} {pred[t]:9.3f} " + " ".join(f"{r[t]:14.3f}" for r in rows[::4]) + f"   {mlp[t]:10.3f}")
    spread = {t: (min(r[t] for r in rows if r['n'] >= 2000), max(r[t] for r in rows if r['n'] >= 2000)) for t in tests}
    log("range over n∈{2000,20000} × prevalence∈{.1,.5,.9}: " +
        ", ".join(f"{t.split()[0]} {a:.3f}–{b:.3f}" for t, (a, b) in spread.items()))
    log("2-layer net, predicted from its own detection rate (trade-off curve): " +
        ", ".join(f"{t.split()[0]} pred {v:.3f} meas {mlp[t]:.3f}" for t, v in mlp_tradeoff.items()))
    RESULTS["C1"] = {"predicted": pred, "logistic_gd": rows, "two_layer": mlp, "two_layer_tradeoff": mlp_tradeoff}


# ------------------------------------------------------------------ C2
def C2(rng):
    """Cor. 2: detectability paradox. Stronger target signal => more partial-match false alarms."""
    d, k, sig = 20, 3, 1.0
    Xn_cal = rng.normal(0, sig, (200000, d))
    log("\nC2  Detectability paradox: P2 false alarms vs per-channel signal δ/σ (k=3)")
    log(f"{'δ/σ':>5s} {'power pred':>10s} {'power meas':>10s} {'P2 pred':>8s} {'P2 meas':>8s} {'P1 pred':>8s} {'P1 meas':>8s}")
    rows = []
    for delta in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        mu = np.zeros(d); mu[:k] = delta
        X = np.r_[rng.normal(0, sig, (5000, d)), rng.normal(0, sig, (5000, d)) + mu]
        y = np.r_[np.zeros(5000), np.ones(5000)]
        f = logistic_gd(X, y)
        thr = calibrate(f, Xn_cal)
        D = delta * np.sqrt(k)
        def meas(m):
            sh = np.zeros(d); sh[:m] = delta
            return float((f(rng.normal(0, sig, (20000, d)) + sh) > thr).mean())
        r = dict(delta=delta, pw_p=norm.cdf(D - Z), pw=meas(3), p2_p=norm.cdf(2 * delta / np.sqrt(k) - Z), p2=meas(2),
                 p1_p=norm.cdf(delta / np.sqrt(k) - Z), p1=meas(1))
        rows.append(r)
        log(f"{delta:5.1f} {r['pw_p']:10.3f} {r['pw']:10.3f} {r['p2_p']:8.3f} {r['p2']:8.3f} {r['p1_p']:8.3f} {r['p1']:8.3f}")
    beta = np.array([0.5, 0.9, 0.98, 0.999])
    tradeoff = norm.cdf((1 - 1 / k) * (norm.ppf(beta) + Z) - Z)
    log("trade-off curve, k=3: power " + ", ".join(f"{b:.3f}→P2 {t:.2f}" for b, t in zip(beta, tradeoff)))
    RESULTS["C2"] = {"rows": rows, "tradeoff": dict(zip(map(float, beta), map(float, tradeoff)))}


# ------------------------------------------------------------------ C3
def C3(rng):
    """Thm 3: composite null. Additive vs intersection-union (min) vs capped-evidence GAM, as signal grows."""
    k, sig, n = 3, 1.0, 400000
    log("\nC3  Composite null, k=3. Size = worst false-alarm rate over partial faults; power = target detection")
    log(f"{'δ/σ':>4s} | {'additive (simple-null cal.)':>27s} | {'additive (bounded-null cal.)':>28s} | "
        f"{'intersection-union (min)':>25s} | {'capped Σ min(a,δ)':>18s} | {'bound':>5s}")
    log(f"{'':4s} | {'size≤δ  size∞   power':>27s} | {'size≤δ  size∞   power':>28s} | {'size∞   power  pred':>25s} | {'size∞   power':>18s} |")
    rows = []
    for delta in (2.0, 3.0, 4.0, 5.0):
        a0 = rng.normal(0, sig, (n, k))
        def sh(m, size):
            v = np.zeros(k); v[:m] = size
            return rng.normal(0, sig, (n, k)) + v
        part_d, part_inf, tgt = sh(2, delta), sh(2, 50 * delta), sh(3, delta)
        add = lambda a: a.sum(1)
        t1 = np.sqrt(k) * Z * sig
        t2 = (k - 1) * delta + np.sqrt(k) * Z * sig
        iut = lambda a: a.min(1)
        cap = lambda a: np.minimum(a, delta).sum(1)
        worst = np.c_[np.full(n, 1e9), np.full(n, 1e9), a0[:, 2]]
        t4 = np.quantile(cap(worst), 1 - ALPHA)
        r = dict(delta=delta,
                 a1=((add(part_d) > t1).mean(), (add(part_inf) > t1).mean(), (add(tgt) > t1).mean()),
                 a2=((add(part_d) > t2).mean(), (add(part_inf) > t2).mean(), (add(tgt) > t2).mean()),
                 iu=((iut(part_inf) > Z).mean(), (iut(tgt) > Z).mean(), norm.cdf(delta - Z) ** k),
                 cp=((cap(part_inf) > t4).mean(), (cap(tgt) > t4).mean()),
                 bound=norm.cdf(delta - Z))
        rows.append(r)
        log(f"{delta:4.1f} | {r['a1'][0]:6.3f} {r['a1'][1]:6.3f} {r['a1'][2]:7.3f} | {r['a2'][0]:7.3f} {r['a2'][1]:6.3f} {r['a2'][2]:7.3f} | "
            f"{r['iu'][0]:7.3f} {r['iu'][1]:7.3f} {r['iu'][2]:5.3f} | {r['cp'][0]:7.3f} {r['cp'][1]:7.3f}  | {r['bound']:5.3f}")
    RESULTS["C3"] = rows


def C8(rng):
    """Cor. 2: predict the toy-study partial-match rates (your slide 12) from each model's detection rate alone."""
    k = 3
    slide12 = {"LogReg": (0.99, 0.35, 0.89), "Random forest": (0.99, 0.22, 0.83), "Grad. boosting": (0.99, 0.21, 0.87),
               "MLP": (0.98, 0.19, 0.73), "LSTM": (0.98, 0.20, 0.73)}
    log("\nC8  Toy study (slide 12): partial-match rates predicted from detection rate alone, no fitting")
    log(f"{'model':15s} {'CAP':>5s} {'P1 pred':>8s} {'P1 obs':>7s} {'P2 pred':>8s} {'P2 obs':>7s}")
    out = {}
    for m, (cap, p1, p2) in slide12.items():
        Dhat = norm.ppf(cap) + Z
        pp1, pp2 = norm.cdf(Dhat / k - Z), norm.cdf(2 * Dhat / k - Z)
        out[m] = (pp1, p1, pp2, p2)
        log(f"{m:15s} {cap:5.2f} {pp1:8.2f} {p1:7.2f} {pp2:8.2f} {p2:7.2f}")
    RESULTS["C8"] = out


# ------------------------------------------------------------------ C4
def C4(rng):
    """Thm 4: separated regime. GD on a 2-layer ReLU net places near-misses at ≥ 1−1/√k of the way
    from normal to fault (× R/R_min); rejecting them needs path norm ≥ √k × the minimum."""
    log("\nC4  Separated regime, 2-layer ReLU nets trained by gradient descent (d=10, noise σ=0.15, δ=1)")
    log(f"{'k':>3s} {'linear (k-1)/k':>14s} {'bound 1-1/√k·ρ':>15s} {'measured pos.':>13s} {'ρ=R/Rmin':>9s} "
        f"{'flagged@1%':>10s} {'R ratio with/without near-misses':>32s} {'√k':>5s}")
    d, sig, delta = 10, 0.15, 1.0
    rows = []
    for k in (2, 3, 4, 5, 6, 8):
        res = []
        for seed in range(3):
            r = np.random.default_rng(100 * k + seed)
            mu = np.zeros(d); mu[:k] = delta
            X = np.r_[r.normal(0, sig, (500, d)), r.normal(0, sig, (500, d)) + mu]
            y = np.r_[np.zeros(500), np.ones(500)]
            net = TwoLayer(d, h=64, init=1e-3, seed=seed).fit(X, y, iters=3000, lr=0.01)
            f0, fp = net(np.zeros((1, d)))[0], net(mu[None])[0]
            nms = [mu - delta * np.eye(d)[j] for j in range(k)]
            pos = np.mean([(net(x[None])[0] - f0) / (fp - f0) for x in nms])
            Rmin = (fp - f0) / (delta * np.sqrt(k))
            rho = net.path_norm() / Rmin
            thr = calibrate(net, r.normal(0, sig, (20000, d)))
            flag = np.mean([(net(r.normal(0, sig, (4000, d)) + x) > thr).mean() for x in nms])
            # same data + all-but-one negatives
            Xn = np.r_[X] ; yn = y
            for x in nms:
                Xn = np.r_[Xn, r.normal(0, sig, (100, d)) + x]; yn = np.r_[yn, np.zeros(100)]
            net2 = TwoLayer(d, h=64, init=1e-3, seed=seed).fit(Xn, yn, iters=3000, lr=0.01)
            f0b, fpb = net2(np.zeros((1, d)))[0], net2(mu[None])[0]
            ratio = (net2.path_norm() / (fpb - f0b)) / (net.path_norm() / (fp - f0))
            res.append((pos, rho, flag, ratio))
        pos, rho, flag, ratio = np.mean(res, 0)
        rows.append(dict(k=k, linear=(k - 1) / k, bound=1 - rho / np.sqrt(k), pos=pos, rho=rho, flag=flag, ratio=ratio))
        log(f"{k:3d} {(k-1)/k:14.3f} {1 - rho/np.sqrt(k):15.3f} {pos:13.3f} {rho:9.2f} {flag:10.3f} {ratio:32.2f} {np.sqrt(k):5.2f}")
    RESULTS["C4"] = rows


# ------------------------------------------------------------------ C5
def C5(rng):
    """Prop. 5: how many near-miss examples. Partial false alarms vs near-miss count per type, with weight decay."""
    log("\nC5  Recipe: P2 false alarms vs near-miss negatives per type (k=3, 2-layer ReLU, 1000+1000 normal/fault)")
    d, k, sig, delta = 10, 3, 0.3, 1.0
    mu = np.zeros(d); mu[:k] = delta
    counts = (0, 1, 2, 4, 8, 16, 32, 64, 128)
    out = {}
    for wd in (0.0, 1e-3, 1e-2):
        row = []
        for c in counts:
            vals = []
            for seed in range(3):
                r = np.random.default_rng(seed)
                X = np.r_[r.normal(0, sig, (1000, d)), r.normal(0, sig, (1000, d)) + mu]
                y = np.r_[np.zeros(1000), np.ones(1000)]
                for j in range(k):
                    if c:
                        X = np.r_[X, r.normal(0, sig, (c, d)) + mu - delta * np.eye(d)[j]]
                        y = np.r_[y, np.zeros(c)]
                net = TwoLayer(d, h=64, init=1e-3, seed=seed).fit(X, y, iters=2500, lr=0.01, wd=wd)
                thr = calibrate(net, r.normal(0, sig, (20000, d)))
                p2 = np.mean([(net(r.normal(0, sig, (4000, d)) + mu - delta * np.eye(d)[j]) > thr).mean() for j in range(k)])
                cap = (net(r.normal(0, sig, (4000, d)) + mu) > thr).mean()
                vals.append((p2, cap))
            row.append(np.mean(vals, 0))
        out[wd] = row
        log(f"wd={wd:<6g} " + " ".join(f"{c}:{p2:.2f}" for c, (p2, _) in zip(counts, row)) +
            f"   (detection min {min(cp for _, cp in row):.2f})")
    RESULTS["C5"] = {str(k_): [list(map(float, v)) for v in vals] for k_, vals in out.items()}
    RESULTS["C5_counts"] = counts


# ------------------------------------------------------------------ C6
def windows(rng, n, kind, T=60, C=6, k=3, sig=0.5, delta=1.0, L=None):
    X = rng.normal(0, sig, (n, T, C))
    for i in range(n):
        Li = L or rng.integers(12, 19)
        o = rng.integers(4, T - 3 * Li)
        if kind == "full":
            X[i, o:o + Li, :k] += delta
        elif kind == "stag":          # every channel in its own segment (empty overlap graph)
            for j in range(k):
                X[i, o + j * Li:o + (j + 1) * Li, j] += delta
        elif kind == "pair":          # one pair disjoint, the rest overlap both (K_k minus one edge);
            a, b = rng.choice(k, 2, replace=False)   # every channel keeps the same duration Li
            X[i, o:o + Li, a] += delta
            X[i, o + Li:o + 2 * Li, b] += delta
            for j in range(k):
                if j not in (a, b):
                    X[i, o + Li // 2:o + Li // 2 + Li, j] += delta
    return X


def C6(rng):
    """Lemma 6 + Thm 5(b): window statistics are timing-blind; pair-staggered (Helly boundary)
    negatives teach simultaneity, fully staggered ones do not."""
    from sklearn.linear_model import LogisticRegression
    T, C = 60, 6
    def stats(X):
        return np.c_[X.mean(1), X.std(1), X.min(1), X.max(1)]
    n = 3000
    Xn, Xf = windows(rng, n, "none"), windows(rng, n, "full", delta=0.25)
    clf = LogisticRegression(max_iter=3000).fit(stats(np.r_[Xn, Xf]), np.r_[np.zeros(n), np.ones(n)])
    s = lambda X: clf.decision_function(stats(X))
    thr = np.quantile(s(windows(rng, 20000, "none")), 1 - ALPHA)
    blind = {kind: float((s(windows(rng, 4000, kind, delta=0.25)) > thr).mean()) for kind in ("full", "stag", "pair")}
    log("\nC6a Window statistics (mean, std, min, max) + logistic regression, weak signal δ=0.25: fraction flagged")
    log("     " + ", ".join(f"{k_} {v:.3f}" for k_, v in blind.items()) + "   (lemma: stag = pair = full)")

    log("C6b Raw-series 2-layer ReLU net (gradient descent), trained with different negatives: fraction flagged")
    flat = lambda X: X.reshape(len(X), -1)
    res = {}
    for train in ("normal vs fault", "+ fully staggered", "+ pair-staggered"):
        vals = []
        for seed in range(2):
            r = np.random.default_rng(seed + 7)
            Xs = [windows(r, 1500, "none"), windows(r, 1500, "full")]
            ys = [np.zeros(1500), np.ones(1500)]
            if train == "+ fully staggered":
                Xs.append(windows(r, 600, "stag")); ys.append(np.zeros(600))
            if train == "+ pair-staggered":
                Xs.append(windows(r, 600, "pair")); ys.append(np.zeros(600))
            X, y = flat(np.concatenate(Xs)), np.concatenate(ys)
            net = TwoLayer(X.shape[1], h=64, init=1e-2, seed=seed).fit(X, y, iters=1200, lr=0.003, wd=1e-4)
            thr = calibrate(net, flat(windows(r, 6000, "none")))
            vals.append([float((net(flat(windows(r, 2000, kk))) > thr).mean()) for kk in ("full", "stag", "pair")])
        res[train] = np.mean(vals, 0)
        log(f"     {train:20s} target {res[train][0]:.2f} · fully staggered {res[train][1]:.2f} · pair-staggered {res[train][2]:.2f}")
    RESULTS["C6"] = {"window_stats": blind, "raw": {k_: list(map(float, v)) for k_, v in res.items()}}


# ------------------------------------------------------------------ C7
def C7(rng):
    """Lemma 7: temporal dependence enters only through the effective noise of the shift statistic."""
    log("\nC7  AR(1) effective noise of the baseline-corrected post-onset mean (σ_x=1, B=8 baseline steps)")
    for phi in (0.0, 0.5, 0.7, 0.9):
        for L in (10, 30):
            B, gap = 8, 2
            T = B + gap + L
            n = 100000
            x = np.empty((n, T)); x[:, 0] = rng.normal(0, 1, n)
            e = rng.normal(0, np.sqrt(1 - phi ** 2), (n, T))
            for t in range(1, T):
                x[:, t] = phi * x[:, t - 1] + e[:, t]
            stat = x[:, B + gap:].mean(1) - x[:, :B].mean(1)
            def vmean(m):
                i = np.arange(m)
                return (phi ** np.abs(i[:, None] - i[None, :])).sum() / m ** 2
            i1, i2 = np.arange(B), np.arange(B + gap, T)
            cov = (phi ** np.abs(i1[:, None] - i2[None, :])).sum() / (B * L)
            pred = vmean(L) + vmean(B) - 2 * cov
            log(f"  φ={phi:.1f} L={L:2d}: Var measured {stat.var():.4f}  formula {pred:.4f}")
    RESULTS["C7"] = "see log"


def C9(rng):
    """Prop. 4.5: near-miss fraction p* below which no weight-decay (path-norm) minimizer rejects all
    near-misses. Rigorous comparison: explicit linear 'cheap' solution vs the path-norm lower bound
    R >= max(sqrt(k) m+/delta, |m+m0|/(delta sqrt(k))) for any net that rejects every near-miss."""
    from scipy.optimize import brentq, minimize
    ell = lambda z: np.logaddexp(0, -z)
    log("\nC9  Supervision–regularization threshold p*(λ) (δ=1): below p*, every minimizer flags some near-miss")
    out = {}
    starts = [(a, b) for a in (0.5, 2, 5, 9) for b in (0.5, 2, 5, 9)]

    def best(fun):
        return min(minimize(fun, x0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000)).fun
                   for x0 in starts)

    for k in (3, 4, 6, 8):
        row = []
        for lam in (1e-4, 1e-3, 1e-2, 3e-2):
            def gap(p):
                q = (1 - p) / 2
                def cheap(z):
                    m0, m1 = abs(z[0]), abs(z[1])
                    fnm = m1 - (m0 + m1) / k
                    return q * ell(m0) + q * ell(m1) + p * ell(-fnm) + lam * (m0 + m1) / np.sqrt(k)
                def narrow(z):
                    m, m0 = z
                    R = max(np.sqrt(k) * max(m, 0), abs(m + m0) / np.sqrt(k))
                    return q * ell(m) + q * ell(m0) + lam * R
                return best(narrow) - best(cheap)
            lo, hi = 1e-6, 0.95
            pstar = brentq(gap, lo, hi, xtol=1e-6) if gap(lo) > 0 and gap(hi) < 0 else (0.0 if gap(lo) <= 0 else float("inf"))
            row.append((lam, pstar))
        out[k] = row
        log(f"  k={k}: " + ", ".join(f"λ={l:g} → p*={p_:.4f}" for l, p_ in row))
    RESULTS["C9"] = {str(k_): v for k_, v in out.items()}


ALL = {"C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "C6": C6, "C7": C7, "C8": C8, "C9": C9}

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    t0 = time.time()
    for w in which:
        ALL[w](np.random.default_rng(0))
    path = os.path.join(OUT, "results.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(json.loads(json.dumps(RESULTS, default=float)))
    with open(path, "w") as f:
        json.dump(old, f, indent=1)
    log(f"\ndone in {time.time() - t0:.0f}s -> {OUT}/results.json")
