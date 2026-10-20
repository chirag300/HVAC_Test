"""
paper_synth.py -- multi-seed synthetic experiments for the paper (mean ± sd over seeds).

S1  partial-overlap law and detectability paradox: GD logistic regression and a GD two-layer
    ReLU net, k=3, d=20, delta/sigma swept; measured P1/P2 vs. the law evaluated at the
    model's own measured detection rate.
S2  separated regime: near-miss position, flag rate and path-norm ratio (k = 2..8).
S3  temporal teaching set (Helly): raw-series two-layer net trained with no timing negatives,
    fully staggered negatives, or pair-staggered negatives.
S4  recipe curve: P2 vs near-miss negatives per type, with and without weight decay.

    python paper/paper_synth.py [S1 S2 S3 S4]      -> paper/synth_out/results.json, log.txt
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "theory"))
import theory_checks as tc  # noqa: E402

OUT = os.path.join(HERE, "synth_out")
os.makedirs(OUT, exist_ok=True)
Z = tc.Z
R = {}
LOG = open(os.path.join(OUT, "log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def law(cap, r, n=None):
    eps = 0.5 / n if n else 1e-4
    b = min(max(cap, eps), 1 - eps)
    return float(norm.cdf(r * (norm.ppf(b) + Z) - Z))


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]


# ------------------------------------------------------------------ S1
def S1(seeds=5):
    d, k, sig = 20, 3, 1.0
    log("\nS1 law + paradox (k=3, d=20, n=4000 balanced, 1% FPR); mean±sd over", seeds, "seeds")
    out = {}
    for model in ("logistic", "mlp"):
        rows = []
        for delta in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            mu = np.zeros(d); mu[:k] = delta
            vals = []
            for sd in range(seeds):
                r = np.random.default_rng(1000 + sd)
                X = np.r_[r.normal(0, sig, (2000, d)), r.normal(0, sig, (2000, d)) + mu]
                y = np.r_[np.zeros(2000), np.ones(2000)]
                if model == "logistic":
                    f = tc.logistic_gd(X, y, iters=2000, lr=0.5)
                else:
                    net = tc.TwoLayer(d, h=64, init=1e-2, seed=sd).fit(X, y, iters=1500, lr=0.01)
                    f = net
                thr = tc.calibrate(f, r.normal(0, sig, (20000, d)))
                m = 20000
                cap = (f(r.normal(0, sig, (m, d)) + mu) > thr).mean()
                p2 = np.mean([(f(r.normal(0, sig, (m, d)) + mu - delta * np.eye(d)[j]) > thr).mean() for j in range(k)])
                p1 = np.mean([(f(r.normal(0, sig, (m, d)) + delta * np.eye(d)[j]) > thr).mean() for j in range(k)])
                vals.append((cap, p2, p1, law(cap, 2 / 3, m), law(cap, 1 / 3, m)))
            v = np.array(vals)
            row = dict(delta=delta, cap=ms(v[:, 0]), p2=ms(v[:, 1]), p1=ms(v[:, 2]), p2_law=ms(v[:, 3]),
                       p1_law=ms(v[:, 4]), p2_oracle=law(norm.cdf(delta * np.sqrt(k) - Z), 2 / 3))
            rows.append(row)
            log(f"  {model:8s} δ/σ={delta:.1f}  CAP {row['cap'][0]:.3f}±{row['cap'][1]:.3f}  "
                f"P2 {row['p2'][0]:.3f}±{row['p2'][1]:.3f} (law {row['p2_law'][0]:.3f})  "
                f"P1 {row['p1'][0]:.3f}±{row['p1'][1]:.3f} (law {row['p1_law'][0]:.3f})")
        out[model] = rows
    R["S1"] = out


# ------------------------------------------------------------------ S2
def S2(seeds=3):
    d, sig, delta = 10, 0.15, 1.0
    log("\nS2 separated regime (d=10, σ=0.15, δ=1, 500+500 train); mean±sd over", seeds, "seeds")
    rows = []
    for k in (2, 3, 4, 6, 8):
        vals = []
        for sd in range(seeds):
            r = np.random.default_rng(200 * k + sd)
            mu = np.zeros(d); mu[:k] = delta
            X = np.r_[r.normal(0, sig, (500, d)), r.normal(0, sig, (500, d)) + mu]
            y = np.r_[np.zeros(500), np.ones(500)]
            net = tc.TwoLayer(d, h=64, init=1e-3, seed=sd).fit(X, y, iters=3000, lr=0.01)
            f0, fp = net(np.zeros((1, d)))[0], net(mu[None])[0]
            nms = [mu - delta * np.eye(d)[j] for j in range(k)]
            pos = np.mean([(net(x[None])[0] - f0) / (fp - f0) for x in nms])
            rho = net.path_norm() / ((fp - f0) / (delta * np.sqrt(k)))
            thr = tc.calibrate(net, r.normal(0, sig, (20000, d)))
            flag = np.mean([(net(r.normal(0, sig, (4000, d)) + x) > thr).mean() for x in nms])
            Xn, yn = X, y
            for x in nms:
                Xn = np.r_[Xn, r.normal(0, sig, (100, d)) + x]; yn = np.r_[yn, np.zeros(100)]
            net2 = tc.TwoLayer(d, h=64, init=1e-3, seed=sd).fit(Xn, yn, iters=3000, lr=0.01)
            f0b, fpb = net2(np.zeros((1, d)))[0], net2(mu[None])[0]
            ratio = (net2.path_norm() / (fpb - f0b)) / (net.path_norm() / (fp - f0))
            thr2 = tc.calibrate(net2, r.normal(0, sig, (20000, d)))
            flag2 = np.mean([(net2(r.normal(0, sig, (4000, d)) + x) > thr2).mean() for x in nms])
            cap2 = (net2(r.normal(0, sig, (4000, d)) + mu) > thr2).mean()
            vals.append((pos, rho, 1 - rho / np.sqrt(k), flag, ratio, flag2, cap2))
        v = np.array(vals)
        row = dict(k=k, linear=(k - 1) / k, pos=ms(v[:, 0]), kappa=ms(v[:, 1]), bound=ms(v[:, 2]),
                   flag=ms(v[:, 3]), ratio=ms(v[:, 4]), flag_after=ms(v[:, 5]), cap_after=ms(v[:, 6]))
        rows.append(row)
        log(f"  k={k}  pos {row['pos'][0]:.3f}±{row['pos'][1]:.3f} (linear {(k-1)/k:.3f}, bound "
            f"{row['bound'][0]:.3f})  κ {row['kappa'][0]:.2f}  flagged {row['flag'][0]:.3f}  "
            f"norm ratio {row['ratio'][0]:.2f}±{row['ratio'][1]:.2f} (k={k})  after: flagged {row['flag_after'][0]:.3f} "
            f"CAP {row['cap_after'][0]:.3f}")
    R["S2"] = rows


# ------------------------------------------------------------------ S3
def S3(seeds=5):
    log("\nS3 temporal teaching set (T=60, C=6, k=3, σ=0.5, δ=1; raw-series 2-layer net); mean±sd over",
        seeds, "seeds")
    flat = lambda X: X.reshape(len(X), -1)
    res = {}
    for train in ("normal vs fault", "+ fully staggered", "+ pair-staggered"):
        vals = []
        for sd in range(seeds):
            r = np.random.default_rng(sd + 50)
            Xs = [tc.windows(r, 1500, "none"), tc.windows(r, 1500, "full")]
            ys = [np.zeros(1500), np.ones(1500)]
            if train == "+ fully staggered":
                Xs.append(tc.windows(r, 600, "stag")); ys.append(np.zeros(600))
            if train == "+ pair-staggered":
                Xs.append(tc.windows(r, 600, "pair")); ys.append(np.zeros(600))
            X, y = flat(np.concatenate(Xs)), np.concatenate(ys)
            net = tc.TwoLayer(X.shape[1], h=64, init=1e-2, seed=sd).fit(X, y, iters=1200, lr=0.003, wd=1e-4)
            thr = tc.calibrate(net, flat(tc.windows(r, 6000, "none")))
            vals.append([float((net(flat(tc.windows(r, 2000, kk))) > thr).mean()) for kk in ("full", "stag", "pair")])
        v = np.array(vals)
        res[train] = {"target": ms(v[:, 0]), "stag": ms(v[:, 1]), "pair": ms(v[:, 2])}
        log(f"  {train:20s} target {v[:,0].mean():.3f}±{v[:,0].std(ddof=1):.3f}  fully-stag "
            f"{v[:,1].mean():.3f}±{v[:,1].std(ddof=1):.3f}  pair-stag {v[:,2].mean():.3f}±{v[:,2].std(ddof=1):.3f}")
    R["S3"] = res


# ------------------------------------------------------------------ S4
def S4(seeds=3):
    d, k, sig, delta = 10, 3, 0.3, 1.0
    mu = np.zeros(d); mu[:k] = delta
    counts = (0, 4, 16, 64, 256)
    log("\nS4 recipe (d=10, k=3, σ=0.3; 1000+1000 normal/fault + c near-misses per type); mean±sd over",
        seeds, "seeds")
    out = {}
    for wd in (0.0, 1e-3):
        rows = []
        for c in counts:
            vals = []
            for sd in range(seeds):
                r = np.random.default_rng(300 + sd)
                X = np.r_[r.normal(0, sig, (1000, d)), r.normal(0, sig, (1000, d)) + mu]
                y = np.r_[np.zeros(1000), np.ones(1000)]
                for j in range(k):
                    if c:
                        X = np.r_[X, r.normal(0, sig, (c, d)) + mu - delta * np.eye(d)[j]]
                        y = np.r_[y, np.zeros(c)]
                net = tc.TwoLayer(d, h=64, init=1e-3, seed=sd).fit(X, y, iters=2500, lr=0.01, wd=wd)
                thr = tc.calibrate(net, r.normal(0, sig, (20000, d)))
                p2 = np.mean([(net(r.normal(0, sig, (4000, d)) + mu - delta * np.eye(d)[j]) > thr).mean()
                              for j in range(k)])
                cap = (net(r.normal(0, sig, (4000, d)) + mu) > thr).mean()
                vals.append((p2, cap))
            v = np.array(vals)
            rows.append(dict(c=c, p2=ms(v[:, 0]), cap=ms(v[:, 1])))
        out[str(wd)] = rows
        log(f"  wd={wd:<6g} " + "  ".join(f"{rw['c']}: {rw['p2'][0]:.2f}±{rw['p2'][1]:.2f}" for rw in rows)
            + f"   (CAP min {min(rw['cap'][0] for rw in rows):.3f})")
    R["S4"] = out


if __name__ == "__main__":
    which = sys.argv[1:] or ["S1", "S2", "S3", "S4"]
    t0 = time.time()
    for w in which:
        globals()[w]()
        path = os.path.join(OUT, "results.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update(json.loads(json.dumps(R, default=float)))
        json.dump(old, open(path, "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")
