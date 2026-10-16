"""
shortcut_lib.py -- shared pieces for the "fault detectors learn the wrong rule" study.

* window_stats          6 stats x C channels (mean, min, max, std, slope, last-8 mean)
* build_construct       splice a fault deviation D into a normal window W, per test type
* and_rule_score        the "true" narrow rule: all signature channels deviate together
* DETECTORS / fit_detectors / flag_rates   models + 1%-false-alarm calibration
* subset_cv_accuracy    identifiability check (how well do channel subsets separate train data)

Every window array is (N, T, C). A construct is   X = W + G * D
where W is a normal window, D = (fault - normal) deviation, and G (T, C) says where/how much
of the deviation is applied. On real data W and D come from time-aligned fault-free and
faulty simulations; on the toy data W is AR(1) noise and D is a constant step.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from lstm_numpy import NumpyLSTMClassifier
from ci_numpy import NumpyChannelIndependentClassifier

STAT_NAMES = ["mean", "min", "max", "std", "slope", "last8"]
FPR = 0.01


# --------------------------------------------------------------------- features
def baseline_subtract(X, n=8):
    return (X - X[:, :n].mean(axis=1, keepdims=True)).astype(np.float32)


def window_stats(X):
    """(N,T,C) -> (N, 6*C), stat-major blocks: feature j = stat_idx*C + channel."""
    N, T, C = X.shape
    t = np.arange(T) - (T - 1) / 2.0
    t = t / (t ** 2).sum() if T > 1 else np.zeros(T)
    slope = np.einsum("ntc,t->nc", X, t)
    f = [X.mean(1), X.min(1), X.max(1), X.std(1), slope, X[:, -8:].mean(1)]
    return np.concatenate(f, axis=1).astype(np.float32)


def stat_columns(channels):
    """Indices of window_stats features belonging to a channel subset."""
    return lambda C: [s * C + c for s in range(len(STAT_NAMES)) for c in channels]


# ------------------------------------------------------------------ constructs
TEST_KINDS = ["CAP", "P1", "P2", "BIG1", "ASYNC", "ASYNC2", "PAIRSTAG", "UP", "OFF"]


def build_construct(kind, W, D, onsets, sig, rng, nonsig="normal", off_pool=None,
                    big=3.0, dip_len=6, keep=None, channel=None, dscale=1.0):
    """
    kind     : CAP | P<m> (m of k sig channels) | ALLBUT1 | BIG1 | ASYNC | ASYNC2 | PAIRSTAG
               | UP | OFF | SINGLE (only `channel` deviates) | NORMAL
               ASYNC/ASYNC2 are FULLY staggered (no two signature channels shifted at once).
               PAIRSTAG is the Helly boundary case: one pair of signature channels is never
               shifted at the same time, every other pair overlaps (all channels shifted for the
               same duration). Theory: only PAIRSTAG-type negatives teach "simultaneous".
    W, D     : (N,T,C) normal windows and fault deviations (already standardized)
    onsets   : (N,) onset index per window; deviation applied for t >= onset
    sig      : list of signature channel indices (length k)
    nonsig   : "normal" -> non-signature channels keep normal values (clean test)
               "real"   -> non-signature channels also carry their real deviation
    off_pool : channels OFF may move the deviation onto (default: all non-signature)
    keep     : for ALLBUT1 / PAIRSTAG, which signature position / pair per window (cycles if None)
    dscale   : multiply the fault deviation D (signal-strength sweep; theory: detectability paradox)
    """
    if dscale != 1.0:
        D = D * np.float32(dscale)
    N, T, C = W.shape
    k = len(sig)
    sig = np.asarray(sig)
    nons = np.setdiff1d(np.arange(C), sig)
    X = W.copy()
    if kind == "NORMAL":
        return X
    for n in range(N):
        o = int(onsets[n])
        G = np.zeros((T, C), np.float32)
        if nonsig == "real" and kind != "OFF":
            G[o:, nons] = 1.0
        if kind == "CAP":
            G[o:, sig] = 1.0
            if nonsig == "real":
                G[o:, :] = 1.0
        elif kind.startswith("P") and kind[1:].isdigit():
            m = int(kind[1:])
            G[o:, rng.choice(sig, size=min(m, k), replace=False)] = 1.0
        elif kind == "ALLBUT1":
            j = (n % k) if keep is None else int(keep[n])
            G[o:, np.delete(sig, j)] = 1.0
        elif kind == "SINGLE":
            G[o:, channel] = 1.0
        elif kind == "BIG1":
            G[o:, rng.choice(sig)] = big
        elif kind == "UP":
            G[o:, sig] = -1.0
        elif kind == "ASYNC":
            bounds = np.linspace(o, T, k + 1).astype(int)
            for seg, c in enumerate(rng.permutation(sig)):
                G[bounds[seg]:bounds[seg + 1], c] = 1.0
        elif kind == "PAIRSTAG":
            pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
            a, b = pairs[(n if keep is None else int(keep[n])) % len(pairs)]
            h = (T - o) // 2
            G[o:o + h, sig[a]] = 1.0
            G[o + h:o + 2 * h, sig[b]] = 1.0
            for j in range(k):
                if j not in (a, b):
                    G[o + h // 2:o + h // 2 + h, sig[j]] = 1.0
        elif kind == "ASYNC2":
            # short dips at separate random times, never more than one channel at once
            free = np.ones(T, bool)
            free[:o] = False
            for c in sig:
                starts = [s for s in range(o, T - dip_len + 1) if free[s:s + dip_len].all()]
                if not starts:
                    break
                s = rng.choice(starts)
                G[s:s + dip_len, c] = 1.0
                free[max(0, s - 2):s + dip_len + 2] = False
        elif kind == "OFF":
            pool = np.asarray(off_pool if off_pool is not None else nons)
            tgt = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for c_src, c_dst in zip(sig, tgt):
                X[n, o:, c_dst] += D[n, o:, c_src]
            continue
        else:
            raise ValueError(kind)
        X[n] += G * D[n]
    return X


# ------------------------------------------------------- theory predictions
from itertools import combinations
from statistics import NormalDist

_N = NormalDist()


def _probit(rate, n=None):
    eps = 0.5 / n if n else 1e-4
    return _N.inv_cdf(min(max(float(rate), eps), 1 - eps))


def predict_from_detection(cap_rate, r, alpha=FPR, n=None):
    """Theory, Cor. 'power-specificity curve': a detector that is (a monotone function of) the
    additive statistic flags a fault reproducing a share r of the target's signal with
    probability Phi(r * (Phi^-1(beta) + z_alpha) - z_alpha), beta = its detection rate."""
    z = _N.inv_cdf(1 - alpha)
    return _N.cdf(r * (_probit(cap_rate, n) + z) - z)


def predict_partial_additive(cap_rate, single_rates, m, alpha=FPR, n=None):
    """Same law with unequal channels: channel j's share r_j is read off its single-channel flag
    rate, r_j = (Phi^-1(single_j) + z) / D, and the m-of-k rate is averaged over channel subsets.
    If the model is additive across channels this predicts P_m from CAP and the singles alone."""
    z = _N.inv_cdf(1 - alpha)
    D = _probit(cap_rate, n) + z
    if D <= 0:
        return float("nan")
    r = [(_probit(s, n) + z) / D for s in single_rates]
    vals = [_N.cdf(D * sum(r[j] for j in sub) - z) for sub in combinations(range(len(r)), m)]
    return float(np.mean(vals))


# ------------------------------------------------------------------ true rule
def and_rule_score(X, sig, signs, win=8, start=8, scale=None):
    """The narrow mechanism: all signature channels shifted in their fault direction during
    the same stretch of time.  score = max_tau min_c mean_{t in [tau, tau+win)} s_c x_c(t) / scale_c
    (X is baseline-subtracted. A partial match is capped by its un-shifted channel; a
    staggered shift never has all channels up in the same local window.)"""
    Z = X[:, :, sig] * np.asarray(signs)[None, None, :]
    if scale is not None:
        Z = Z / np.asarray(scale)[None, None, :]
    T = Z.shape[1]
    cs = np.concatenate([np.zeros_like(Z[:, :1]), np.cumsum(Z, axis=1)], axis=1)
    taus = np.arange(start, T - win + 1)
    means = (cs[:, taus + win] - cs[:, taus]) / win
    return means.min(axis=2).max(axis=1)


# ------------------------------------------------------------------ detectors
def _raw12(X):
    return X[:, ::4].reshape(len(X), -1)


DETECTORS = {
    # name: (input kind, factory)
    "LogReg (stats)": ("stats", lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
    "Random forest (stats)": ("stats", lambda s: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=s)),
    "Grad. boosting (stats)": ("stats", lambda s: GradientBoostingClassifier(random_state=s)),
    "MLP (raw series)": ("raw12", lambda s: MLPClassifier(
        hidden_layer_sizes=(128, 64), solver="adam", max_iter=300, early_stopping=True,
        random_state=s)),
    "LSTM (raw series)": ("raw48", lambda s: NumpyLSTMClassifier(hidden=32, epochs=30, seed=s)),
    # per-channel shared encoder + linear head: additive across channels by construction
    "Channel-indep. (raw series)": ("raw48", lambda s: NumpyChannelIndependentClassifier(
        hidden=32, epochs=30, seed=s)),
    # same, but each channel's evidence is bounded (tanh): can represent the AND (theory, Prop. cap)
    "Capped CI (raw series)": ("raw48", lambda s: NumpyChannelIndependentClassifier(
        hidden=32, epochs=30, seed=s, capped=True)),
}


def _inputs(kind, X):
    if kind == "stats":
        return window_stats(X)
    if kind == "raw12":
        return _raw12(X)
    return X


def _score(model, Z):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(Z), float)
    return model.predict_proba(Z)[:, 1]


class Fitted:
    def __init__(self, name, kind, model, thr):
        self.name, self.kind, self.model, self.thr = name, kind, model, thr

    def score(self, X):
        return _score(self.model, _inputs(self.kind, X))

    def flag(self, X):
        return self.score(X) > self.thr


class AndRule:
    """The narrow mechanism itself, calibrated to the same 1% false-alarm budget.

    On real HVAC data the signature channels also move together for normal reasons
    (schedule start/stop moves fan speed, power and airflow at once). So the rule is applied
    to *residuals*: each signature channel minus a ridge prediction of it from the
    non-signature channels at the same time step (fit on normal windows). The AND then asks
    "did all signature channels move together, beyond what the rest of the system explains"."""
    name = "AND rule (true)"

    def __init__(self, sig, signs, X_val_normal, residualize=True, lam=1.0, win=8, start=8):
        self.sig, self.signs = list(sig), np.asarray(signs)
        self.win, self.start = win, start
        C = X_val_normal.shape[2]
        self.nons = [c for c in range(C) if c not in self.sig]
        half = len(X_val_normal) // 2
        fit, cal = (X_val_normal[:half], X_val_normal[half:]) if residualize else (None, X_val_normal)
        self.B = None
        if residualize and self.nons:
            A = fit[:, :, self.nons].reshape(-1, len(self.nons)).astype(np.float64)
            A = np.hstack([A, np.ones((len(A), 1))])
            Y = fit[:, :, self.sig].reshape(-1, len(self.sig)).astype(np.float64)
            self.B = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
        # per-channel residual noise level, so "moved" means the same thing on every channel
        ref = fit if fit is not None else cal
        self.scale = np.maximum(self.residual(ref)[:, :, self.sig].std(axis=(0, 1)), 1e-6)
        self.thr = np.quantile(self.score(cal), 1 - FPR)

    def residual(self, X):
        if self.B is None:
            return X
        N, T, _ = X.shape
        A = X[:, :, self.nons].reshape(-1, len(self.nons))
        pred = A @ self.B[:-1] + self.B[-1]
        R = X.copy()
        R[:, :, self.sig] = X[:, :, self.sig] - pred.reshape(N, T, -1).astype(X.dtype)
        return R

    def score(self, X):
        return and_rule_score(self.residual(X), self.sig, self.signs, win=self.win,
                              start=self.start, scale=self.scale)

    def flag(self, X):
        return self.score(X) > self.thr


def fit_detectors(X_tr, y_tr, X_val_norm, names=None, seed=0, log=print):
    out = {}
    for name in (names or DETECTORS):
        kind, fac = DETECTORS[name]
        m = fac(seed)
        m.fit(_inputs(kind, X_tr), y_tr)
        thr = np.quantile(_score(m, _inputs(kind, X_val_norm)), 1 - FPR)
        out[name] = Fitted(name, kind, m, thr)
        log(f"    fitted {name}")
    return out


def flag_rates(detectors, tests):
    """{detector: {test: fraction flagged}}; tests = {name: X}."""
    return {d.name: {t: float(d.flag(X).mean()) if len(X) else float("nan")
                     for t, X in tests.items()} for d in detectors}


def subset_cv_accuracy(X, y, C, subsets, seed=0):
    """5-fold CV accuracy of LogReg on window stats restricted to channel subsets."""
    F = window_stats(X)
    res = {}
    for label, chans in subsets.items():
        cols = stat_columns(chans)(C)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        res[label] = float(cross_val_score(clf, F[:, cols], y, cv=5).mean())
    return res


def gb_stat_importance(fitted, C, sig):
    """Gradient-boosting importance summed per statistic over signature channels (slide 16)."""
    imp = fitted.model.feature_importances_
    return {s: float(sum(imp[i * C + c] for c in sig)) for i, s in enumerate(STAT_NAMES)}


def logreg_channel_weight_share(fitted, C, sig):
    coef = np.abs(fitted.model[-1].coef_.ravel())
    per_ch = np.array([coef[[s * C + c for s in range(len(STAT_NAMES))]].sum() for c in range(C)])
    per_ch = per_ch / per_ch.sum()
    return {int(c): float(per_ch[c]) for c in sig}


def fmt_table(rates, cols, digits=2):
    w = max(len(r) for r in rates) + 2
    head = "Model".ljust(w) + "".join(c.rjust(9) for c in cols)
    lines = [head, "-" * len(head)]
    for r, row in rates.items():
        lines.append(r.ljust(w) + "".join(
            (f"{row.get(c, float('nan')):.{digits}f}").rjust(9) for c in cols))
    return "\n".join(lines)


def md_table(rates, cols, digits=2):
    lines = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for r, row in rates.items():
        lines.append(f"| {r} | " + " | ".join(
            f"{row.get(c, float('nan')):.{digits}f}" for c in cols) + " |")
    return "\n".join(lines)
