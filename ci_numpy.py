"""
ci_numpy.py -- channel-independent (CI) detector, trained by gradient descent (Adam), NumPy only.

Same structure as channel-independent time-series models (PatchTST-style encoders, the
per-channel embeddings of MOMENT / TimesFM followed by a linear head):

    z_c   = relu( x_c W1 + b1 )        shared encoder applied to each channel's series x_c (T -> H)
    score = sum_c  A_c . z_c  + b0      linear head over the concatenated channel embeddings

so  score(x) = sum_c g_c(x_c): there is NO cross-channel interaction. By construction this
model can only represent additive rules across channels -- it cannot express "all k signature
channels moved together" except as a thresholded sum. It is the empirical counterpart of the
additive class in the theory (partial-match false-alarm law, small margin with targeted
negatives).

capped=True gives the CAPPED variant (theory: Prop. "capped evidence is necessary"):

    u_c   = A_c . z_c + d_c
    score = beta * sum_c tanh(u_c) + b0

Each channel now contributes at most |beta| to the score, so k-1 loud channels can never
outweigh one silent channel when b0 sits between -(k)beta and -(k-1)beta: the model CAN
represent the intersection-union (AND) rule, while staying channel-independent. The uncapped
model cannot (one unbounded channel can always carry the score).

Run `python ci_numpy.py` for a finite-difference gradient check.
"""
import numpy as np


def _sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class NumpyChannelIndependentClassifier:
    def __init__(self, hidden=32, lr=3e-3, epochs=30, batch_size=64, clip=5.0, val_frac=0.15,
                 patience=6, seed=0, dtype=np.float32, verbose=False, capped=False):
        self.H, self.lr, self.epochs, self.batch_size = hidden, lr, epochs, batch_size
        self.clip, self.val_frac, self.patience, self.seed = clip, val_frac, patience, seed
        self.dtype, self.verbose, self.capped = dtype, verbose, capped

    def _init(self, T, C):
        rng = np.random.default_rng(self.seed)
        H = self.H
        self.p = {"W1": rng.normal(0, np.sqrt(2.0 / T), (T, H)), "b1": np.zeros(H),
                  "A": rng.normal(0, 1.0 / np.sqrt(C * H), (C, H)), "b0": np.zeros(1)}
        if self.capped:
            # same A as the uncapped model, so tanh starts in its linear regime (the capped model
            # starts as the additive one, up to the scale beta) and saturates only where it helps
            self.p["d"] = np.zeros(C)
            self.p["beta"] = np.full(1, 3.0)
        self.p = {k: v.astype(self.dtype) for k, v in self.p.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def _forward(self, X):
        p = self.p
        Zpre = np.einsum("ntc,th->nch", X, p["W1"]) + p["b1"]
        Z = np.maximum(Zpre, 0)
        if self.capped:
            U = np.tanh(np.einsum("nch,ch->nc", Z, p["A"]) + p["d"])
            s = p["beta"][0] * U.sum(1) + p["b0"][0]
            return s, Zpre, (Z, U)
        s = np.einsum("nch,ch->n", Z, p["A"]) + p["b0"][0]
        return s, Zpre, Z

    def _loss_grad(self, X, y):
        s, Zpre, Z = self._forward(X)
        pr = _sigmoid(s)
        loss = -np.mean(y * np.log(pr + 1e-7) + (1 - y) * np.log(1 - pr + 1e-7))
        ds = ((pr - y) / len(X)).astype(self.dtype)
        if self.capped:
            Z, U = Z
            g = {"b0": np.array([ds.sum()], self.dtype),
                 "beta": np.array([(ds[:, None] * U).sum()], self.dtype)}
            du = ds[:, None] * self.p["beta"][0] * (1 - U * U)          # (n, C)
            g["A"] = np.einsum("nc,nch->ch", du, Z)
            g["d"] = du.sum(0)
            dZpre = du[:, :, None] * self.p["A"][None] * (Zpre > 0)
        else:
            g = {"A": np.einsum("n,nch->ch", ds, Z), "b0": np.array([ds.sum()], self.dtype)}
            dZpre = ds[:, None, None] * self.p["A"][None] * (Zpre > 0)
        g["W1"] = np.einsum("ntc,nch->th", X, dZpre)
        g["b1"] = dZpre.sum(axis=(0, 1))
        return loss, g

    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
        sc = min(1.0, self.clip / (norm + 1e-12))
        for k in self.p:
            gk = g[k] * sc
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(self.dtype)

    def fit(self, X, y):
        X = np.asarray(X, self.dtype)
        y = np.asarray(y, self.dtype)
        rng = np.random.default_rng(self.seed + 1)
        self._init(X.shape[1], X.shape[2])
        idx = rng.permutation(len(X))
        nv = int(len(X) * self.val_frac)
        va, tr = idx[:nv], idx[nv:]
        best, best_p, bad = np.inf, None, 0
        for ep in range(self.epochs):
            perm = rng.permutation(tr)
            for s in range(0, len(perm), self.batch_size):
                b = perm[s:s + self.batch_size]
                _, g = self._loss_grad(X[b], y[b])
                self._adam(g)
            if nv:
                pr = _sigmoid(self.decision_function(X[va]))
                lv = float(-np.mean(y[va] * np.log(pr + 1e-7) + (1 - y[va]) * np.log(1 - pr + 1e-7)))
                if self.verbose:
                    print(f"    ci ep{ep:02d} val {lv:.4f}")
                if lv < best - 1e-4:
                    best, bad, best_p = lv, 0, {k: v.copy() for k, v in self.p.items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    def decision_function(self, X, batch=4096):
        X = np.asarray(X, self.dtype)
        return np.concatenate([self._forward(X[s:s + batch])[0] for s in range(0, len(X), batch)]) \
            if len(X) else np.zeros(0)

    def predict_proba(self, X):
        pr = _sigmoid(self.decision_function(X))
        return np.stack([1 - pr, pr], 1)


def gradcheck(seed=0, tol=1e-5, capped=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 9, 4))
    y = (rng.random(6) > 0.5).astype(float)
    m = NumpyChannelIndependentClassifier(hidden=5, dtype=np.float64, capped=capped)
    m._init(9, 4)
    for k in m.p:
        m.p[k] += rng.normal(scale=0.3, size=m.p[k].shape)
    _, g = m._loss_grad(X, y)
    worst = 0.0
    for k, P in m.p.items():
        flat = P.reshape(-1)
        for j in rng.choice(flat.size, size=min(10, flat.size), replace=False):
            o = flat[j]
            flat[j] = o + 1e-6
            lp, _ = m._loss_grad(X, y)
            flat[j] = o - 1e-6
            lm, _ = m._loss_grad(X, y)
            flat[j] = o
            num = (lp - lm) / 2e-6
            ana = g[k].reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst, worst < tol


if __name__ == "__main__":
    ok_all = True
    for capped in (False, True):
        err, ok = gradcheck(capped=capped)
        ok_all &= ok
        print(f"{'capped' if capped else 'uncapped'} CI model gradient check: max relative error "
              f"{err:.2e} -> {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok_all else 1)
