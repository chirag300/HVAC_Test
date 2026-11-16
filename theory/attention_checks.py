"""Numerical checks of the attention and sample-size results (NumPy + SciPy only).

  A   no near-misses: at the best sum rule the attention-logit gradient is zero (uniform attention is stationary)
  B   with channel near-misses: dL/dq_a = u* pi_nm delta^2 (k-1)/k^2 E_nm[sig f] > 0  (push toward the weakest sensor)
  B'  with C-k irrelevant sensors: targets push attention toward the MOST evidence (OR-like), near-misses push back
  E   soft AND: optimal temperature beta*(rho) ~ kappa * rho, kappa = delta E_nm[sig f0] / (2 k sigma^2 E_b[l''(y f0)])
  D   near-miss sample size: IU learner fails with prob <= eta at the upper bound; lower bound for any learner

Population quantities are large Monte Carlo averages with common random numbers, so finite-difference
gradients are accurate. Logits are q_a * a_j (q_a > 0 = evidence-seeking / OR-like, q_a < 0 = AND-like).

    python theory/attention_checks.py            # all checks (~2-4 min CPU)
    python theory/attention_checks.py E D        # a subset
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm


def _pool(a, qa):
    s = qa * a
    s = s - s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    return (p * a).sum(1)


def _sample(rng, N, C, k, d, s, pi0, pinm, pit):
    """Normal, target (first k sensors shifted by d) and k channel near-miss types."""
    n0, nt, nnm = int(N * pi0), int(N * pit), int(N * pinm)
    mt = np.zeros(C); mt[:k] = d
    A, Y, G = [s * rng.normal(size=(n0, C)), mt + s * rng.normal(size=(nt, C))], [-np.ones(n0), np.ones(nt)], [np.zeros(n0, int), np.ones(nt, int)]
    for j in range(k):
        m = mt.copy(); m[j] = 0.0
        A.append(m + s * rng.normal(size=(nnm // k, C))); Y.append(-np.ones(nnm // k)); G.append(2 * np.ones(nnm // k, int))
    return np.vstack(A), np.concatenate(Y), np.concatenate(G)


def _best_sum(a, y):
    """Newton's method for the best uniform-attention rule u * mean(a) + c."""
    x = a.mean(1); X = np.c_[x, np.ones_like(x)]; w = np.zeros(2)
    for _ in range(40):
        z = y * (X @ w); sg = expit(-z)
        g = -(X * (y * sg)[:, None]).mean(0)
        H = (X * (sg * (1 - sg))[:, None]).T @ X / len(x)
        w -= np.linalg.solve(H, g)
    return w


def _dL_dqa(a, y, u, c, h=1e-3):
    L = lambda qa: np.mean(-log_expit(y * (u * _pool(a, qa) + c)))
    return (L(h) - L(-h)) / (2 * h)


def check_AB(N=1_500_000):
    print("\n[A, B] C = k = 3, delta/sigma = 1.5: gradient on the attention logit at the best sum rule")
    rng = np.random.default_rng(0); k, d, s = 3, 1.5, 1.0
    ok = True
    for pis in [(0.5, 0.0, 0.5), (0.4, 0.2, 0.4), (0.25, 0.5, 0.25)]:
        a, y, g = _sample(rng, N, k, k, d, s, *pis)
        u, c = _best_sum(a, y)
        num = _dL_dqa(a, y, u, c)
        f = u * a.mean(1) + c
        pred = u * pis[1] * d**2 * (k - 1) / k**2 * (np.mean(expit(f[g == 2])) if pis[1] > 0 else 0.0)
        ok &= abs(num - pred) < 0.02 * max(1.0, abs(pred)) + 3e-3
        print(f"  pi_nm={pis[1]:.2f}: u*={u:.3f}  dL/dq_a numeric {num:+.5f}  formula {pred:+.5f}")
    print("  A/B:", "PASS" if ok else "CHECK")
    return ok


def check_Bprime(N=1_200_000):
    print("\n[B'] C = 6 tokens, k = 3 signature, delta/sigma = 1.5")
    rng = np.random.default_rng(1); C, k, d, s = 6, 3, 1.5, 1.0
    Dt, Dn = d**2 * k * (C - k) / C**2, d**2 * (k - 1) * (C - k + 1) / C**2
    ok = True
    for pis in [(0.5, 0.0, 0.5), (0.4, 0.2, 0.4), (0.25, 0.5, 0.25)]:
        a, y, g = _sample(rng, N, C, k, d, s, *pis)
        u, c = _best_sum(a, y)
        num = _dL_dqa(a, y, u, c)
        f = u * a.mean(1) + c
        pred = u * (-pis[2] * Dt * np.mean(expit(-f[g == 1])) + (pis[1] * Dn * np.mean(expit(f[g == 2])) if pis[1] > 0 else 0.0))
        ok &= abs(num - pred) < 0.03 * max(1.0, abs(pred)) + 3e-3
        way = "q_a grows: OR-like (most evidence)" if num < 0 else "q_a falls: AND-like (least evidence)"
        print(f"  pi_nm={pis[1]:.2f}: dL/dq_a numeric {num:+.5f}  formula {pred:+.5f}  -> gradient descent: {way}")
    print("  B':", "PASS" if ok else "CHECK")
    return ok


def check_E(N=400_000):
    print("\n[E] soft AND: beta*(rho)/rho -> kappa as rho -> 0 (temperature model f = u*softmin_beta(a) + c)")
    rng = np.random.default_rng(2)
    ok = True
    for k, d in [(3, 1.0), (3, 1.5), (5, 1.5)]:
        s = 1.0
        A0 = s * rng.normal(size=(N, k)); At = d + s * rng.normal(size=(N, k))
        Anm = np.concatenate([d * (1 - np.eye(k)[j]) + s * rng.normal(size=(N // k, k)) for j in range(k)])

        def L(th, rho):
            b, u, c = th
            base = 0.5 * np.mean(-log_expit(-(u * _pool(A0, -b) + c))) + 0.5 * np.mean(-log_expit(u * _pool(At, -b) + c))
            return (1 - rho) * base + rho * np.mean(-log_expit(-(u * _pool(Anm, -b) + c)))

        u0, c0 = k * d / s**2, -k * d**2 / (2 * s**2)
        fn, ft, fm = u0 * A0.mean(1) + c0, u0 * At.mean(1) + c0, u0 * Anm.mean(1) + c0
        l2 = 0.5 * np.mean(expit(-fn) * expit(fn)) + 0.5 * np.mean(expit(ft) * expit(-ft))
        kappa = d * np.mean(expit(fm)) / (2 * k * s**2 * l2)
        lower = d / (8 * k * s**2) * np.exp(k * d**2 / (8 * s**2))
        th = np.array([0.0, u0, c0]); row = []
        for rho in (0.0, 0.02, 0.1, 0.4):
            th = minimize(L, th, args=(rho,), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-11, maxiter=4000)).x
            row.append((rho, th[0]))
        b0 = row[0][1]                       # Monte Carlo offset of the rho = 0 minimiser (population value: 0)
        slope = (row[1][1] - b0) / 0.02
        ok &= abs(slope - kappa) < 0.15 * kappa and kappa >= lower and all(np.isfinite(b) for _, b in row)
        print(f"  k={k} delta/sigma={d}: kappa={kappa:.3f} (lower bound {lower:.3f}); fitted slope {slope:.3f}; "
              + ", ".join(f"beta*({r})={b:.3f}" for r, b in row))
    print("  E:", "PASS" if ok else "CHECK")
    return ok


def check_D():
    print("\n[D] near-miss sample size (eps = 0.01): IU learner at the upper bound vs. lower bounds")
    rng = np.random.default_rng(3); eps = 0.01; z = norm.ppf(1 - eps); sig = 1.0
    ok = True
    for D, gam, eta, k in [(z, 0.05, 0.05, 3), (z + 1, 0.05, 0.05, 3), (z + 2, 0.05, 0.05, 3), (z, 0.1, 0.05, 3)]:
        nu = -D * sig
        sj = sig * ((D - z) - norm.ppf(norm.cdf(D - z) - gam))
        n_up = int(np.ceil(8 * sig**2 / sj**2 * np.log(2 * k / eta)))
        n_lo = 2 * sig**2 / sj**2 * np.log(1 / (4 * eta))

        def fail(n, R=20000):
            xb = nu + sig * rng.normal(size=(R, k)) / np.sqrt(n)
            c = xb + sig * z + sig * np.sqrt(2 * np.log(2 * k / eta) / n)
            return 1 - np.mean(((c >= nu + sig * z) & (c <= nu + sig * z + sj)).all(1))

        f_up, f_small = fail(n_up), fail(max(1, n_up // 8))
        ok &= f_up <= eta
        print(f"  D-z={D - z:.0f} gamma={gam}: slack s={sj:.4f} (approx {sig * gam / norm.pdf(D - z):.4f}); "
              f"n_upper={n_up}, n_lower={n_lo:.0f}; IU failure at n_upper {f_up:.4f} (eta={eta}), at n_upper/8 {f_small:.3f}")
    print("  D:", "PASS" if ok else "CHECK")
    return ok


CHECKS = {"A": check_AB, "B": check_AB, "Bp": check_Bprime, "E": check_E, "D": check_D}

if __name__ == "__main__":
    want = sys.argv[1:] or ["A", "Bp", "E", "D"]
    done, res = set(), {}
    for w in want:
        fn = CHECKS[w]
        if fn in done:
            continue
        done.add(fn); res[w] = fn()
    print("\nSummary:", ", ".join(f"{k}: {'PASS' if v else 'CHECK'}" for k, v in res.items()))
