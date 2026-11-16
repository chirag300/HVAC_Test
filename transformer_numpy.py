"""
transformer_numpy.py -- two transformer fault detectors, trained by gradient descent (Adam),
NumPy only (forward pass, backprop and Adam written out; finite-difference gradient checks).

1. NumpyChannelTokenTransformer  ("full attention", iTransformer-style)
   Every sensor is ONE token. Its 48-step window is embedded by a small per-channel MLP plus a
   learned sensor embedding, one self-attention layer mixes the sensor tokens, and a learned
   query pools them with softmax attention:

       e_c  = relu(x_c W1 + b1) W2 + b2 + P_c            token of sensor c        (d)
       h    = e + softmax(e WQ (e WK)^T / sqrt d) e WV WO  self-attention, residual (C x d)
       pi   = softmax( (h Wk) q / sqrt d )               pooling attention over sensors (C)
       score = w . sum_c pi_c h_c + b0

   The pooling attention is the object of the theory: uniform pi is the sum rule; pi that
   concentrates on the sensor with the least fault evidence is the AND (intersection-union)
   rule. `attention(X)` returns pi so its focus can be measured.

2. NumpyPatchTSTClassifier  (channel-independent, PatchTST-style)
   Each sensor's series is processed ALONE by a shared patch transformer, and the per-sensor
   outputs are added by a linear head, so score(X) = sum_c g_c(X_c): additive across sensors by
   construction (theory: Theorem 4(d) / finite-strength look-alikes).

       optional RevIN: x_c <- (x_c - mean(x_c)) / std(x_c)  (instance normalization, no params)
       patches of length 8, stride 8 -> 6 tokens;  t = patch Wp + bp + pos
       h = t + self-attention(t);   z_c = mean_tokens relu(h Wf + bf)
       score = sum_c A_c . z_c + b0

   With RevIN each sensor's evidence is scale-free, hence bounded under ever larger shifts:
   the theory predicts RevIN removes the "one loud sensor" failure of additive models.

Both follow the API of the other NumPy models: fit(X, y), decision_function(X), predict_proba.
X is (N, T, C) float, y in {0, 1}. Run `python transformer_numpy.py` for the gradient checks.
"""
import numpy as np


def _sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(S, axis=-1):
    S = S - S.max(axis=axis, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------- self-attention block
def attn_forward(E, WQ, WK, WV, WO):
    """E: (B, L, d). Returns H = E + softmax(Q K^T / sqrt d) V WO and a cache."""
    d = E.shape[-1]
    Q, K, V = E @ WQ, E @ WK, E @ WV
    A = _softmax(np.einsum("bld,bmd->blm", Q, K) / np.sqrt(d))
    O = np.einsum("blm,bmd->bld", A, V)
    H = E + O @ WO
    return H, (E, Q, K, V, A, O)


def attn_backward(dH, cache, WQ, WK, WV, WO):
    E, Q, K, V, A, O = cache
    d = E.shape[-1]
    g = {}
    g["WO"] = np.einsum("bld,ble->de", O, dH)
    dO = dH @ WO.T
    dA = np.einsum("bld,bmd->blm", dO, V)
    dV = np.einsum("blm,bld->bmd", A, dO)
    dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(d)
    dQ = np.einsum("blm,bmd->bld", dS, K)
    dK = np.einsum("blm,bld->bmd", dS, Q)
    g["WQ"] = np.einsum("bld,ble->de", E, dQ)
    g["WK"] = np.einsum("bld,ble->de", E, dK)
    g["WV"] = np.einsum("bld,ble->de", E, dV)
    dE = dH + dQ @ WQ.T + dK @ WK.T + dV @ WV.T
    return dE, g


# --------------------------------------------------------------- shared training loop
class _AdamTrainer:
    """Adam + gradient clipping + early stopping on an internal validation split."""

    def __init__(self, lr=3e-3, epochs=30, batch_size=64, clip=5.0, weight_decay=0.0,
                 val_frac=0.15, patience=6, seed=0, dtype=np.float32, verbose=False):
        self.lr, self.epochs, self.batch_size, self.clip = lr, epochs, batch_size, clip
        self.wd, self.val_frac, self.patience, self.seed = weight_decay, val_frac, patience, seed
        self.dtype, self.verbose = dtype, verbose

    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        if self.wd:
            g = {k: g[k] + self.wd * self.p[k] for k in g}
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
        sc = min(1.0, self.clip / (norm + 1e-12))
        for k in self.p:
            gk = g[k] * sc
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(self.dtype)

    def _bce(self, X, y):
        pr = _sigmoid(self.decision_function(X))
        return float(-np.mean(y * np.log(pr + 1e-7) + (1 - y) * np.log(1 - pr + 1e-7)))

    def _loss_grad(self, X, y):
        s, cache = self._forward(X)
        pr = _sigmoid(s)
        loss = -np.mean(y * np.log(pr + 1e-7) + (1 - y) * np.log(1 - pr + 1e-7))
        ds = ((pr - y) / len(X)).astype(X.dtype)
        return loss, self._backward(ds, cache)

    def fit(self, X, y):
        X = np.asarray(X, self.dtype)
        y = np.asarray(y, self.dtype)
        rng = np.random.default_rng(self.seed + 1)
        self._init(X.shape[1], X.shape[2])
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
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
                lv = self._bce(X[va], y[va])
                if self.verbose:
                    print(f"    {type(self).__name__} ep{ep:02d} val {lv:.4f}")
                if lv < best - 1e-4:
                    best, bad, best_p = lv, 0, {k: v.copy() for k, v in self.p.items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    def decision_function(self, X, batch=2048):
        X = np.asarray(X, self.dtype)
        if not len(X):
            return np.zeros(0)
        return np.concatenate([self._forward(X[s:s + batch])[0] for s in range(0, len(X), batch)])

    def predict_proba(self, X):
        pr = _sigmoid(self.decision_function(X))
        return np.stack([1 - pr, pr], 1)


# --------------------------------------------------------------- 1. channel tokens
class NumpyChannelTokenTransformer(_AdamTrainer):
    def __init__(self, d=32, hidden=32, self_attn=True, **kw):
        super().__init__(**kw)
        self.d, self.H, self.self_attn = d, hidden, self_attn

    def _init(self, T, C):
        rng = np.random.default_rng(self.seed)
        d, H = self.d, self.H
        p = {"W1": rng.normal(0, np.sqrt(2.0 / T), (T, H)), "b1": np.zeros(H),
             "W2": rng.normal(0, np.sqrt(1.0 / H), (H, d)), "b2": np.zeros(d),
             "P": rng.normal(0, 0.1, (C, d)),
             "Wk": rng.normal(0, np.sqrt(1.0 / d), (d, d)), "q": rng.normal(0, 0.1, d),
             "w": rng.normal(0, np.sqrt(1.0 / d), d), "b0": np.zeros(1)}
        if self.self_attn:
            for n in ("WQ", "WK", "WV"):
                p[n] = rng.normal(0, np.sqrt(1.0 / d), (d, d))
            p["WO"] = rng.normal(0, 0.5 * np.sqrt(1.0 / d), (d, d))
        self.p = {k: v.astype(self.dtype) for k, v in p.items()}

    def _forward(self, X):
        p = self.p
        Xc = np.transpose(X, (0, 2, 1))                          # (N, C, T): one token per sensor
        Hpre = Xc @ p["W1"] + p["b1"]
        Z1 = np.maximum(Hpre, 0)
        E = Z1 @ p["W2"] + p["b2"] + p["P"]
        if self.self_attn:
            Hs, acache = attn_forward(E, p["WQ"], p["WK"], p["WV"], p["WO"])
        else:
            Hs, acache = E, None
        Kp = Hs @ p["Wk"]
        s = Kp @ p["q"] / np.sqrt(self.d)                        # (N, C)
        pi = _softmax(s)
        pooled = np.einsum("nc,ncd->nd", pi, Hs)
        score = pooled @ p["w"] + p["b0"][0]
        return score, (Xc, Hpre, Z1, E, acache, Hs, Kp, pi, pooled)

    def _backward(self, ds, cache):
        p = self.p
        Xc, Hpre, Z1, E, acache, Hs, Kp, pi, pooled = cache
        g = {"w": pooled.T @ ds, "b0": np.array([ds.sum()], ds.dtype)}
        dpooled = ds[:, None] * p["w"][None]                      # (N, d)
        dpi = np.einsum("nd,ncd->nc", dpooled, Hs)
        dHs = pi[:, :, None] * dpooled[:, None, :]
        dsl = pi * (dpi - (dpi * pi).sum(1, keepdims=True)) / np.sqrt(self.d)
        g["q"] = np.einsum("nc,ncd->d", dsl, Kp)
        dKp = dsl[:, :, None] * p["q"][None, None, :]
        g["Wk"] = np.einsum("ncd,nce->de", Hs, dKp)
        dHs = dHs + dKp @ p["Wk"].T
        if self.self_attn:
            dE, ga = attn_backward(dHs, acache, p["WQ"], p["WK"], p["WV"], p["WO"])
            g.update(ga)
        else:
            dE = dHs
        g["P"] = dE.sum(0)
        g["b2"] = dE.sum((0, 1))
        g["W2"] = np.einsum("nch,ncd->hd", Z1, dE)
        dHpre = (dE @ p["W2"].T) * (Hpre > 0)
        g["b1"] = dHpre.sum((0, 1))
        g["W1"] = np.einsum("nct,nch->th", Xc, dHpre)
        return g

    def attention(self, X, batch=2048):
        """Pooling attention weights over sensors, (N, C)."""
        X = np.asarray(X, self.dtype)
        return np.concatenate([self._forward(X[s:s + batch])[1][7] for s in range(0, len(X), batch)])


# --------------------------------------------------------------- 2. PatchTST (channel-independent)
class NumpyPatchTSTClassifier(_AdamTrainer):
    def __init__(self, d=32, patch=8, stride=8, revin=False, **kw):
        super().__init__(**kw)
        self.d, self.patch, self.stride, self.revin = d, patch, stride, revin

    def _patches(self, X):
        """(N, T, C) -> (N*C, n_patches, patch), with optional per-series instance normalization."""
        N, T, C = X.shape
        S = np.transpose(X, (0, 2, 1)).reshape(N * C, T)
        if self.revin:
            S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-5)
        starts = range(0, T - self.patch + 1, self.stride)
        return np.stack([S[:, s:s + self.patch] for s in starts], 1)

    def _init(self, T, C):
        rng = np.random.default_rng(self.seed)
        d, P = self.d, self.patch
        npatch = (T - P) // self.stride + 1
        self.C = C
        p = {"Wp": rng.normal(0, np.sqrt(1.0 / P), (P, d)), "bp": np.zeros(d),
             "pos": rng.normal(0, 0.1, (npatch, d)),
             "WQ": rng.normal(0, np.sqrt(1.0 / d), (d, d)), "WK": rng.normal(0, np.sqrt(1.0 / d), (d, d)),
             "WV": rng.normal(0, np.sqrt(1.0 / d), (d, d)), "WO": rng.normal(0, 0.5 * np.sqrt(1.0 / d), (d, d)),
             "Wf": rng.normal(0, np.sqrt(2.0 / d), (d, d)), "bf": np.zeros(d),
             "A": rng.normal(0, 1.0 / np.sqrt(C * d), (C, d)), "b0": np.zeros(1)}
        self.p = {k: v.astype(self.dtype) for k, v in p.items()}

    def _forward(self, X):
        p = self.p
        N, T, C = X.shape
        Pt = self._patches(X)                                      # (N*C, np, P)
        E = Pt @ p["Wp"] + p["bp"] + p["pos"]
        Hs, acache = attn_forward(E, p["WQ"], p["WK"], p["WV"], p["WO"])
        G = Hs @ p["Wf"] + p["bf"]
        R = np.maximum(G, 0)
        z = R.mean(1).reshape(N, C, self.d)                        # per-sensor representation
        score = np.einsum("ncd,cd->n", z, p["A"]) + p["b0"][0]     # additive over sensors
        return score, (Pt, acache, Hs, G, z)

    def _backward(self, ds, cache):
        p = self.p
        Pt, acache, Hs, G, z = cache
        N, C, d = z.shape
        npatch = Hs.shape[1]
        g = {"A": np.einsum("n,ncd->cd", ds, z), "b0": np.array([ds.sum()], ds.dtype)}
        dz = (ds[:, None, None] * p["A"][None]).reshape(N * C, d)
        dG = np.repeat(dz[:, None, :] / npatch, npatch, axis=1) * (G > 0)
        g["Wf"] = np.einsum("bld,ble->de", Hs, dG)
        g["bf"] = dG.sum((0, 1))
        dHs = dG @ p["Wf"].T
        dE, ga = attn_backward(dHs, acache, p["WQ"], p["WK"], p["WV"], p["WO"])
        g.update(ga)
        g["pos"] = dE.sum(0)
        g["bp"] = dE.sum((0, 1))
        g["Wp"] = np.einsum("blp,bld->pd", Pt, dE)
        return g


# --------------------------------------------------------------- gradient checks
def gradcheck(model, T=16, C=4, seed=0, tol=1e-5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(5, T, C))
    y = (rng.random(5) > 0.5).astype(float)
    model.dtype = np.float64
    model._init(T, C)
    for k in model.p:
        model.p[k] = model.p[k] + rng.normal(scale=0.3, size=model.p[k].shape)
    _, g = model._loss_grad(X, y)
    worst = 0.0
    for k, P in model.p.items():
        flat = P.reshape(-1)
        for j in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            o = flat[j]
            flat[j] = o + 1e-6
            lp, _ = model._loss_grad(X, y)
            flat[j] = o - 1e-6
            lm, _ = model._loss_grad(X, y)
            flat[j] = o
            num = (lp - lm) / 2e-6
            ana = g[k].reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst, worst < tol


if __name__ == "__main__":
    ok_all = True
    for name, m in [("channel-token transformer", NumpyChannelTokenTransformer(d=6, hidden=5)),
                    ("channel-token, pooling only", NumpyChannelTokenTransformer(d=6, hidden=5, self_attn=False)),
                    ("PatchTST (no RevIN)", NumpyPatchTSTClassifier(d=6, patch=4, stride=4)),
                    ("PatchTST (RevIN)", NumpyPatchTSTClassifier(d=6, patch=4, stride=4, revin=True))]:
        err, ok = gradcheck(m)
        ok_all &= ok
        print(f"{name} gradient check: max relative error {err:.2e} -> {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok_all else 1)
