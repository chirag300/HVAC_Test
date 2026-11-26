"""Independent numerical checks of every result in the report (NumPy + SciPy).

Each check compares a closed form from the report with a direct computation (Monte Carlo,
quadrature, finite differences or brute-force optimisation). Run:

    python verify_all.py            # all checks, ~3-6 min on a laptop CPU
    python verify_all.py L1 T2      # a subset

Checks: L1 (Lemma 1), L2 (Lemma 2), T1 (Theorem 1), T2 (Theorem 2), T3 (Theorem 3),
FL (full attention layer remark), T4 (Theorem 4), HA (noise-free remark), P1 (Proposition 1(c)),
MD (min-degree interpolator).
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm

RESULTS = []


def report(name, got, want, tol, rel=True):
    err = abs(got - want) / (abs(want) if rel and want != 0 else 1.0)
    ok = err <= tol
    RESULTS.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: computed {got:.4f}, formula {want:.4f}  ({'rel' if rel else 'abs'} err {err:.4f})")


def pool(a, beta):
    """m_beta(a) = sum_j softmax(-beta a)_j a_j, row-wise."""
    s = -beta * a
    s = s - s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    return (p * a).sum(1)


# ----------------------------------------------------------------------------------------- Lemma 1
def check_L1():
    print("\n[L1] Lemma 1: E[S^2 | mean] = s0 + Delta, and Var(S^2) when mu is proportional to 1")
    rng = np.random.default_rng(1)
    C, k, d, s, N = 5, 3, 1.3, 0.8, 2_000_000
    s0 = s**2 * (C - 1) / C
    for name, mu, want in [("normal", np.zeros(C), 0.0),
                           ("fault 1 (C>k)", np.r_[np.full(k, d), np.zeros(C - k)], d**2 * k * (C - k) / C**2),
                           ("near-miss (C>k)", np.r_[np.full(k - 1, d), np.zeros(C - k + 1)], d**2 * (k - 1) * (C - k + 1) / C**2)]:
        a = mu + s * rng.normal(size=(N, C))
        abar = a.mean(1)
        S2 = ((a - abar[:, None]) ** 2).mean(1)
        # conditional mean in two bins of abar: must be the same constant s0 + Delta
        lo = abar < np.median(abar)
        report(f"{name}, lower half of abar", S2[lo].mean(), s0 + want, 0.01)
        report(f"{name}, upper half of abar", S2[~lo].mean(), s0 + want, 0.01)
    a = s * rng.normal(size=(N, C))
    S2 = ((a - a.mean(1, keepdims=True)) ** 2).mean(1)
    report("Var(S^2), mu = 0", S2.var(), 2 * (C - 1) * s**4 / C**2, 0.01)


# ----------------------------------------------------------------------------------------- Lemma 2
def check_L2():
    print("\n[L2] Lemma 2(i): d m_beta / d beta = -Var_pi(a); at beta = 0 it is -S^2(a)")
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 4)) * 2
    for beta in [0.0, 0.7, -1.3]:
        h = 1e-5
        fd = (pool(a, beta + h) - pool(a, beta - h)) / (2 * h)
        s_ = -beta * a; p = np.exp(s_ - s_.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        m = (p * a).sum(1); var = (p * a**2).sum(1) - m**2
        report(f"beta = {beta}: max |fd + Var_pi|", float(np.max(np.abs(fd + var))), 0.0, 1e-6, rel=False)
    S2 = ((a - a.mean(1, keepdims=True)) ** 2).mean(1)
    fd0 = (pool(a, 1e-5) - pool(a, -1e-5)) / 2e-5
    report("beta = 0: max |fd + S^2|", float(np.max(np.abs(fd0 + S2))), 0.0, 1e-6, rel=False)


# ----------------------------------------------------------------------------------------- Theorem 1
def check_T1():
    print("\n[T1] Theorem 1(ii): false-alarm law, and the example numbers (k = 3, delta/sigma = 3, alpha = 1%)")
    rng = np.random.default_rng(3)
    k, d, s, alpha, N = 3, 3.0, 1.0, 0.01, 2_000_000
    za = norm.ppf(1 - alpha)
    # an arbitrary positive weight vector and a correlated covariance: the law holds for any Sigma
    w = np.array([1.0, 0.6, 1.4]); L = np.array([[1, 0, 0], [0.5, 1, 0], [0.2, -0.3, 1.0]]); Sig = L @ L.T
    mu_t = np.full(k, d)
    sd = np.sqrt(w @ Sig @ w); t = za * sd
    z = rng.normal(size=(N, k)) @ L.T
    det = np.mean((mu_t + z) @ w > t)
    for h in [np.array([d, d, 0.0]), np.array([d, 0, 0]), np.array([2.5 * d, 0, 0])]:
        r = (w @ h) / (w @ mu_t)
        got = np.mean((h + z) @ w > t)
        want = norm.cdf(r * (norm.ppf(det) + za) - za)
        report(f"look-alike {h}: alarm rate", got, want, 0.02)
    # example numbers in the paper
    D = np.sqrt(k) * d / s
    report("SUM detection", norm.cdf(D - za), 0.998, 0.001)
    report("SUM flags 2-of-3", norm.cdf(2 / 3 * D - za), 0.87, 0.005)
    report("SUM flags 1-of-3", norm.cdf(1 / 3 * D - za), 0.28, 0.02)
    report("AND detection", norm.cdf(d / s - za) ** 3, 0.42, 0.005)
    report("AND ceiling (Thm 2a)", norm.cdf(d / s - za), 0.75, 0.005)


# ----------------------------------------------------------------------------------------- Theorem 2
def best_linear_power(k, d, M, s, za, rng, n_rand=20000):
    """Brute force over nonnegative weights: max over w of Phi(F(w)/s - za), F from the validity constraint."""
    W = rng.exponential(size=(n_rand, k))
    W = np.vstack([W, np.ones((1, k)), np.eye(k), rng.dirichlet(np.ones(k) * 0.3, size=2000)])
    l1, l2, wmin = W.sum(1), np.sqrt((W**2).sum(1)), W.min(1)
    F = ((d - M) * l1 + M * wmin) / l2
    return norm.cdf(F.max() / s - za), W[F.argmax()]


def check_T2():
    print("\n[T2] Theorem 2: ceiling, AND test, nonnegative linear detectors, unbounded additive detectors")
    rng = np.random.default_rng(4)
    k, d, s, alpha = 3, 2.0, 1.0, 0.01
    za = norm.ppf(1 - alpha)
    # (b) validity of the AND test against random look-alikes (all proper supports, shifts up to M = 10)
    N = 400_000
    worst = 0.0
    for supp in [(0,), (1,), (0, 1), (1, 2), (0, 2)]:
        for M in [d, 5.0, 10.0]:
            mu = np.zeros(k); mu[list(supp)] = M
            a = mu + s * rng.normal(size=(N, k))
            worst = max(worst, np.mean(a.min(1) / s > za))
    report("AND test, worst false-alarm rate over look-alikes (must be <= alpha)", worst, alpha, 0.05)
    a = d + s * rng.normal(size=(4_000_000, k))
    report("AND test power", np.mean(a.min(1) / s > za), norm.cdf(d / s - za) ** k, 0.01)
    q = norm.cdf(za - d / s)
    ratio = (1 - (1 - q) ** k) / q
    RESULTS.append(("AND miss / best possible miss <= k", ratio <= k))
    print(f"  {'PASS' if ratio <= k else 'FAIL'}  AND miss / best possible miss = {ratio:.3f} <= k = {k}")
    # (c) best valid nonnegative linear detector: brute force vs formula, for M below and above k delta/(k-1)
    for M in [0.5 * d, d, 1.2 * d, 1.45 * d, 1.5 * d, 2.0 * d]:
        got, wbest = best_linear_power(k, d, M, s, za, rng)
        want = norm.cdf((k * d - (k - 1) * M) / (s * np.sqrt(k)) - za)
        if M < k / (k - 1) * d:
            report(f"best linear power, M = {M/d:.2f} delta", got, want, 0.002)
        else:
            ok = got <= alpha + 1e-12
            RESULTS.append((f"best linear power <= alpha at M = {M/d:.2f} delta", ok))
            print(f"  {'PASS' if ok else 'FAIL'}  best linear power at M = {M/d:.2f} delta: {got:.5f} <= alpha = {alpha}")
    # (c) check the validity constraint used: the worst look-alike for w is M on all sensors except the smallest weight
    w = np.array([0.5, 1.0, 2.0]); M = 3.0
    worst_mean = max(w @ np.array(v) for v in [(M, M, 0), (M, 0, M), (0, M, M), (M, 0, 0), (0, M, 0), (0, 0, M)])
    report("worst look-alike mean for w = (0.5,1,2)", worst_mean, M * (w.sum() - w.min()), 1e-12)
    # (d) an unbounded additive detector loses validity as a single-sensor look-alike grows
    psi = lambda x: x  # the identity is unbounded and nondecreasing
    t = 6.0
    rates = []
    for A in [2, 5, 10, 20]:
        a = s * rng.normal(size=(200_000, k)); a[:, 0] += A
        rates.append(np.mean(psi(a).sum(1) > t))
    ok = rates[-1] > 0.99
    RESULTS.append(("additive detector false-alarm rate -> 1 as the look-alike grows", ok))
    print(f"  {'PASS' if ok else 'FAIL'}  additive detector (threshold 6) on single-sensor look-alike A e_1, A = 2,5,10,20: {np.round(rates, 3)}")


# ----------------------------------------------------------------------------------------- Theorem 3
def sample(rng, N, C, k, d, s, pi0, pit, pinm):
    n0, nt, nnm = int(N * pi0), int(N * pit), int(N * pinm)
    mt = np.zeros(C); mt[:k] = d
    A = [s * rng.normal(size=(n0, C)), mt + s * rng.normal(size=(nt, C))]
    Y = [-np.ones(n0), np.ones(nt)]; G = [np.zeros(n0, int), np.ones(nt, int)]
    for j in range(k):
        m = mt.copy(); m[j] = 0.0
        A.append(m + s * rng.normal(size=(nnm // k, C))); Y.append(-np.ones(nnm // k)); G.append(np.full(nnm // k, 2))
    return np.vstack(A), np.concatenate(Y), np.concatenate(G)


def best_sum(a, y):
    x = a.mean(1); X = np.c_[x, np.ones_like(x)]; w = np.zeros(2)
    for _ in range(50):
        z = y * (X @ w); sg = expit(-z)
        g = -(X * (y * sg)[:, None]).mean(0)
        H = (X * (sg * (1 - sg))[:, None]).T @ X / len(x)
        w -= np.linalg.solve(H, g)
    return w


def loss(a, y, beta, u, c):
    return np.mean(-log_expit(y * (u * pool(a, beta) + c)))


def check_T3():
    print("\n[T3] Theorem 3: gradient of the temperature at the best SUM rule (finite differences vs formula)")
    cases = [("C = k = 3, no near-misses", 3, 0.5, 0.5, 0.0),
             ("C = k = 3, with near-misses", 3, 0.4, 0.3, 0.3),
             ("C = 5 > k = 3, no near-misses", 5, 0.5, 0.5, 0.0),
             ("C = 5 > k = 3, with near-misses", 5, 0.4, 0.3, 0.3)]
    k, d, s, N = 3, 1.5, 1.0, 1_500_000
    for name, C, pi0, pit, pinm in cases:
        rng = np.random.default_rng(5)
        a, y, g = sample(rng, N, C, k, d, s, pi0, pit, pinm)
        u, c = best_sum(a, y)
        h = 1e-3
        fd = (loss(a, y, h, u, c) - loss(a, y, -h, u, c)) / (2 * h)
        f = u * a.mean(1) + c
        Dt = d**2 * k * (C - k) / C**2
        Dnm = d**2 * (k - 1) * (C - k + 1) / C**2
        pt = np.mean(g == 1); pnm = np.mean(g == 2)
        formula = u * (pt * Dt * np.mean(expit(-f[g == 1])) - (pnm * Dnm * np.mean(expit(f[g == 2])) if pnm > 0 else 0.0))
        if abs(formula) < 1e-9:
            report(f"{name}: dL/dbeta (formula = 0)", fd, 0.0, 2e-3, rel=False)
        else:
            report(f"{name}: dL/dbeta", fd, formula, 0.02)
        print(f"        u* = {u:.3f} (> 0: {u > 0})")
        RESULTS.append((f"{name}: u* > 0", u > 0))
    # regime (a): the SUM rule is the global minimiser over (beta, u, c)
    rng = np.random.default_rng(6)
    a, y, _ = sample(rng, 400_000, 3, 3, d, s, 0.5, 0.5, 0.0)
    u, c = best_sum(a, y)
    best = min((minimize(lambda th: loss(a, y, *th), x0, method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-8, maxiter=3000)) for x0 in
                [(1.0, u, c), (-1.0, u, c), (0.5, 0.5 * u, 0.0), (-2.0, 2 * u, c)]), key=lambda r: r.fun)
    report("regime (a): beta at the minimum over (beta, u, c)", best.x[0], 0.0, 0.03, rel=False)


# ----------------------------------------------------------------------------------------- Full layer
def check_FL():
    print("\n[FL] Remark (full attention layer, C = k): at the SUM rule only the temperature coordinate has a nonzero gradient")
    rng = np.random.default_rng(7)
    k, d, s, N = 3, 1.5, 1.0, 1_000_000
    a, y, _ = sample(rng, N, k, k, d, s, 0.4, 0.3, 0.3)
    u, c = best_sum(a, y)
    uj = np.array([0.3, -0.2, 0.1])  # arbitrary per-sensor value offsets; the bias c absorbs their mean

    def L_full(qa, b):
        logits = qa * a + b
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        f = (p * (u * a + uj)).sum(1) + c - uj.mean()
        return np.mean(-log_expit(y * f))
    h = 1e-3
    gq = (L_full(h, np.zeros(k)) - L_full(-h, np.zeros(k))) / (2 * h)
    gb = [(L_full(0, h * np.eye(k)[j]) - L_full(0, -h * np.eye(k)[j])) / (2 * h) for j in range(k)]
    fd_beta = (loss(a, y, h, u, c) - loss(a, y, -h, u, c)) / (2 * h)
    report("dL/dq_a = -dL/dbeta", gq, -fd_beta, 0.02)
    report("max_j |dL/db_j| (should be ~0)", float(np.max(np.abs(gb))), 0.0, 5e-4, rel=False)


# ----------------------------------------------------------------------------------------- Theorem 4
def kappa_formula(k, d, s, n_quad=200):
    m = k * d**2 / (2 * s**2)
    x, wq = np.polynomial.hermite_e.hermegauss(n_quad); wq = wq / wq.sum()
    zb = m + np.sqrt(2 * m) * x                      # y f0 under the base law: N(m, 2m)
    Eb_l2 = np.sum(wq * expit(zb) * expit(-zb))
    znm = (k - 2) / k * m + np.sqrt(2 * m) * x       # f0 at a near-miss: N((k-2)m/k, 2m)
    Enm_sig = np.sum(wq * expit(znm))
    return d * Enm_sig / (2 * k * s**2 * Eb_l2), m


def check_T4():
    print("\n[T4] Theorem 4: slope kappa of the optimal temperature, by direct minimisation at small rho")
    for k, d, s in [(3, 1.0, 1.0), (3, 1.5, 1.0)]:
        kap, m = kappa_formula(k, d, s)
        rng = np.random.default_rng(8)
        N = 600_000
        z0 = s * rng.normal(size=(N, k)); zt = d + s * rng.normal(size=(N, k))
        znm = np.tile(np.full(k, d), (N, 1)) + s * rng.normal(size=(N, k))
        znm[np.arange(N), np.arange(N) % k] -= d     # equal mix of the k near-miss types
        u0, c0 = k * d / s**2, -k * d**2 / (2 * s**2)

        def L(th, rho):
            b, u, c = th
            Lb = 0.5 * np.mean(-log_expit(-(u * pool(z0, b) + c))) + 0.5 * np.mean(-log_expit(u * pool(zt, b) + c))
            Ln = np.mean(-log_expit(-(u * pool(znm, b) + c)))
            return (1 - rho) * Lb + rho * Ln
        sol = {}
        for rho in [0.0, 0.005, 0.01]:
            r = minimize(L, x0=np.array([0.0, u0, c0]), args=(rho,), method="BFGS", options=dict(gtol=1e-10))
            sol[rho] = r.x[0]
        s1 = (sol[0.005] - sol[0.0]) / 0.005
        s2 = (sol[0.01] - sol[0.0]) / 0.01
        rich = 2 * s1 - s2      # Richardson extrapolation of the slope to rho = 0
        report(f"k = {k}, delta/sigma = {d/s}: slope of beta*(rho) at 0 (Richardson)", rich, kap, 0.03)
        print(f"        beta*(0) = {sol[0.0]:+.4f} (should be ~0); one-sided slopes {s1:.4f} (rho=0.005), {s2:.4f} (rho=0.01)")
        lb = d / (8 * k * s**2) * np.exp(m / 4)
        RESULTS.append((f"k={k}, d/s={d/s}: kappa >= lower bound", kap >= lb))
        print(f"  {'PASS' if kap >= lb else 'FAIL'}  kappa = {kap:.4f} >= lower bound {lb:.4f}")
    kap, _ = kappa_formula(3, 1e-3, 1.0)
    report("small-signal limit kappa -> delta/(k sigma^2)", kap, 1e-3 / 3, 0.01)
    # the closed form equals the general expression Delta E[sig] / (u0 E[l''] Var S^2)
    k, d, s = 4, 1.2, 0.9
    kap, m = kappa_formula(k, d, s)
    Ds = d**2 * (k - 1) / k**2; u0 = k * d / s**2; VarS2 = 2 * (k - 1) * s**4 / k**2
    x, wq = np.polynomial.hermite_e.hermegauss(200); wq = wq / wq.sum()
    Eb = np.sum(wq * expit(m + np.sqrt(2 * m) * x) * expit(-(m + np.sqrt(2 * m) * x)))
    En = np.sum(wq * expit((k - 2) / k * m + np.sqrt(2 * m) * x))
    report("Delta E[sig] / (u0 E[l''] Var S^2) = delta E[sig] / (2 k sigma^2 E[l''])", Ds * En / (u0 * Eb * VarS2), kap, 1e-10)


def check_HA():
    print("\n[HA] Remark (no noise): the temperature gradient is negative for every beta, so beta grows without bound")
    k, d, u, c, pinm = 3, 1.0, 4.0, -2.0, 0.3
    betas = np.linspace(-5, 30, 400)
    m = d * (k - 1) * np.exp(-betas * d) / ((k - 1) * np.exp(-betas * d) + 1)
    mp = np.gradient(m, betas)
    Lp = pinm * expit(u * m + c) * u * mp
    ok = np.all(Lp < 0)
    RESULTS.append(("noise-free: L'(beta) < 0 on [-5, 30]", ok))
    print(f"  {'PASS' if ok else 'FAIL'}  L'(beta) < 0 everywhere on the grid (max {Lp.max():.2e})")


# ----------------------------------------------------------------------------------------- Proposition 1(c)
def check_P1():
    print("\n[P1] Proposition 1(c): the intersection-union learner fails with probability <= eta at the upper bound")
    rng = np.random.default_rng(9)
    k, eta, alpha, gamma, sig, D = 3, 0.1, 0.01, 0.05, 1.0, 3.0
    za = norm.ppf(1 - alpha)
    sj = sig * ((D - za) - norm.ppf(norm.cdf(D - za) - gamma))
    n = int(np.ceil(8 * sig**2 / sj**2 * np.log(2 * k / eta)))
    nu = np.zeros(k)
    fails = 0; R = 4000
    for _ in range(R):
        xbar = nu + sig * rng.normal(size=k) / np.sqrt(n)
        tn = np.sqrt(2 * np.log(2 * k / eta) / n)
        cthr = xbar + sig * za + sig * tn
        valid = np.all(cthr >= nu + sig * za)
        close = np.all(cthr - (nu + sig * za) <= sj)
        fails += not (valid and close)
    ok = fails / R <= eta
    RESULTS.append(("IU learner failure rate <= eta", ok))
    lower = 2 * sig**2 / sj**2 * np.log(1 / (4 * eta))
    print(f"  {'PASS' if ok else 'FAIL'}  n = {n} per type (lower bound {lower:.0f}): failure rate {fails/R:.4f} <= eta = {eta}")


# ----------------------------------------------------------------------------------------- min-degree
def check_MD():
    print("\n[MD] Proposition (min-degree interpolator with near-misses): sign q is the AND on all 2^k points")
    from itertools import product
    for k in [2, 3, 4, 5]:
        ok = True
        for x in product([-1, 1], repeat=k):
            s = sum(x); q = ((s + 1) ** 2 - k**2 - 1) / (2 * k)
            want = 1 if s == k else -1
            ok &= (q == 1 if want == 1 else q <= -1)
        # q interpolates the training points
        ok &= abs(((k + 1) ** 2 - k**2 - 1) / (2 * k) - 1) < 1e-12 and abs(((k - 1) ** 2 - k**2 - 1) / (2 * k) + 1) < 1e-12 \
            and abs(((1 - k) ** 2 - k**2 - 1) / (2 * k) + 1) < 1e-12
        RESULTS.append((f"min-degree q, k = {k}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}  k = {k}")


CHECKS = dict(L1=check_L1, L2=check_L2, T1=check_T1, T2=check_T2, T3=check_T3, FL=check_FL, T4=check_T4,
              HA=check_HA, P1=check_P1, MD=check_MD)

if __name__ == "__main__":
    for key in (sys.argv[1:] or CHECKS):
        CHECKS[key]()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
