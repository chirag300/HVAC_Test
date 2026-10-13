"""
lstm_numpy.py -- dependency-free LSTM binary classifier trained by gradient descent (Adam).

No PyTorch: forward pass, full backprop-through-time and Adam are written in NumPy so the
code runs on Beefinator as-is.

    model = NumpyLSTMClassifier(hidden=32, epochs=30)
    model.fit(X, y)                 # X: (N, T, C) float, y: (N,) in {0,1}
    s = model.decision_function(X)  # logits (higher = more "fault")

Architecture: 1-layer LSTM (PyTorch gate order i, f, g, o), logit = w . h_T + b.
Loss: mean binary cross-entropy with logits. Gradient-norm clipping, early stopping on an
internal validation split (best weights restored).

Run `python lstm_numpy.py` to execute a finite-difference gradient check.
"""
import numpy as np


def _sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class NumpyLSTMClassifier:
    def __init__(self, hidden=32, lr=3e-3, epochs=30, batch_size=64, clip=5.0,
                 weight_decay=0.0, val_frac=0.15, patience=6, seed=0,
                 dtype=np.float32, verbose=False):
        self.H = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip = clip
        self.wd = weight_decay
        self.val_frac = val_frac
        self.patience = patience
        self.seed = seed
        self.dtype = dtype
        self.verbose = verbose
        self.history = []

    # ------------------------------------------------------------------ params
    def _init(self, C):
        rng = np.random.default_rng(self.seed)
        H = self.H
        k = 1.0 / np.sqrt(H)
        self.p = {
            "Wx": rng.uniform(-k, k, (C, 4 * H)),
            "Wh": rng.uniform(-k, k, (H, 4 * H)),
            "b": np.zeros(4 * H),
            "w": rng.uniform(-k, k, H),
            "c0": np.zeros(1),
        }
        self.p["b"][H:2 * H] = 1.0  # forget-gate bias
        self.p = {n: v.astype(self.dtype) for n, v in self.p.items()}
        self.m = {n: np.zeros_like(v) for n, v in self.p.items()}
        self.v = {n: np.zeros_like(v) for n, v in self.p.items()}
        self.t = 0

    # ----------------------------------------------------------------- forward
    def _forward(self, X, keep=True):
        p, H = self.p, self.H
        B, T, _ = X.shape
        h = np.zeros((B, H), self.dtype)
        c = np.zeros((B, H), self.dtype)
        cache = []
        XW = X @ p["Wx"] + p["b"]  # (B,T,4H) precomputed input projection
        for t in range(T):
            a = XW[:, t] + h @ p["Wh"]
            i = _sigmoid(a[:, :H])
            f = _sigmoid(a[:, H:2 * H])
            g = np.tanh(a[:, 2 * H:3 * H])
            o = _sigmoid(a[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            if keep:
                cache.append((h, c, i, f, g, o, tc))
            h, c = h_new, c_new
        logit = h @ p["w"] + p["c0"][0]
        return logit, h, cache

    def _loss_grad(self, X, y):
        p, H = self.p, self.H
        B, T, _ = X.shape
        logit, hT, cache = self._forward(X)
        prob = _sigmoid(logit)
        eps = 1e-7
        loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
        g = {n: np.zeros_like(v) for n, v in p.items()}
        dlogit = ((prob - y) / B).astype(self.dtype)
        g["w"] = hT.T @ dlogit
        g["c0"] = np.array([dlogit.sum()], self.dtype)
        dh = np.outer(dlogit, p["w"]).astype(self.dtype)
        dc = np.zeros((B, H), self.dtype)
        dA = np.empty((B, T, 4 * H), self.dtype)
        for t in range(T - 1, -1, -1):
            h_prev, c_prev, i, f, gg, o, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc * tc)
            di = dc * gg
            dg = dc * i
            df = dc * c_prev
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 dg * (1 - gg * gg), do * o * (1 - o)], axis=1)
            dA[:, t] = da
            g["Wh"] += h_prev.T @ da
            dh = da @ p["Wh"].T
            dc = dc * f
        g["Wx"] = np.einsum("btc,btk->ck", X, dA)
        g["b"] = dA.sum(axis=(0, 1))
        return loss, g

    # ------------------------------------------------------------------- train
    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
        scale = min(1.0, self.clip / (norm + 1e-12))
        for n in self.p:
            gn = g[n] * scale + self.wd * self.p[n]
            self.m[n] = b1 * self.m[n] + (1 - b1) * gn
            self.v[n] = b2 * self.v[n] + (1 - b2) * gn * gn
            mh = self.m[n] / (1 - b1 ** self.t)
            vh = self.v[n] / (1 - b2 ** self.t)
            self.p[n] -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(self.dtype)

    def fit(self, X, y):
        X = np.asarray(X, self.dtype)
        y = np.asarray(y, self.dtype)
        rng = np.random.default_rng(self.seed + 1)
        self._init(X.shape[2])
        idx = rng.permutation(len(X))
        n_val = int(len(X) * self.val_frac) if self.val_frac > 0 else 0
        va, tr = idx[:n_val], idx[n_val:]
        best, best_p, bad = np.inf, None, 0
        for ep in range(self.epochs):
            perm = rng.permutation(tr)
            tot = 0.0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s:s + self.batch_size]
                loss, g = self._loss_grad(X[b], y[b])
                self._adam(g)
                tot += loss * len(b)
            tr_loss = tot / len(tr)
            if n_val:
                lv = self._bce(X[va], y[va])
                self.history.append((ep, tr_loss, lv))
                if self.verbose:
                    print(f"    lstm ep{ep:02d} train {tr_loss:.4f} val {lv:.4f}")
                if lv < best - 1e-4:
                    best, bad = lv, 0
                    best_p = {n: v.copy() for n, v in self.p.items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    def _bce(self, X, y):
        prob = _sigmoid(self.decision_function(X))
        return float(-np.mean(y * np.log(prob + 1e-7) + (1 - y) * np.log(1 - prob + 1e-7)))

    def decision_function(self, X, batch=2048):
        X = np.asarray(X, self.dtype)
        out = [self._forward(X[s:s + batch], keep=False)[0] for s in range(0, len(X), batch)]
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.stack([1 - p, p], axis=1)


def gradcheck(seed=0, tol=1e-5):
    """Central finite differences vs analytic BPTT gradients (float64)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(5, 7, 4))
    y = (rng.random(5) > 0.5).astype(float)
    m = NumpyLSTMClassifier(hidden=6, dtype=np.float64)
    m._init(4)
    for n in m.p:
        m.p[n] += rng.normal(scale=0.3, size=m.p[n].shape)
    _, g = m._loss_grad(X, y)
    worst = 0.0
    for n, P in m.p.items():
        flat = P.reshape(-1)
        for j in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + 1e-6
            lp, _ = m._loss_grad(X, y)
            flat[j] = old - 1e-6
            lm, _ = m._loss_grad(X, y)
            flat[j] = old
            num = (lp - lm) / 2e-6
            ana = g[n].reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst, worst < tol


if __name__ == "__main__":
    err, ok = gradcheck()
    print(f"LSTM gradient check: max relative error {err:.2e} -> {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
