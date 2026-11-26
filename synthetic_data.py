"""
synthetic_data.py -- the synthetic sensor source used in the CS 886 report.

20 sensors, 48 time steps, AR(1) noise (phi = 0.7, unit variance), a slow common load on sensors 0-5.
Fault 1 lowers sensors 0, 1, 2 by delta = 2.5 noise units from a random onset in [10, 25]; the
per-window baseline (mean of the first 8 steps) is subtracted. Look-alikes are built from the same
step change by shortcut_lib.build_construct (P1, P2, BIG1, ASYNC, PAIRSTAG, ...).
"""
import numpy as np

import shortcut_lib as L


class ToySource:
    def __init__(self, C=20, T=48, phi=0.7, delta=2.5, sig=(0, 1, 2), crosstalk=(0, 6),
                 load_phi=0.97, load_w=0.6, onset=(10, 25), nonsig="normal"):
        self.name = "synthetic"
        self.C, self.T, self.phi, self.delta = C, T, phi, delta
        self.sig = list(sig)
        self.signs = np.array([-1] * len(sig))
        self.channel_names = [f"ch{c}" for c in range(C)]
        self.ct = range(*crosstalk)
        self.load_phi, self.load_w = load_phi, load_w
        self.onset = onset
        self.off_pool = [c for c in range(C) if c not in sig and c not in self.ct]

    def _ar1(self, rng, n, C, phi):
        x = np.empty((n, self.T, C), np.float32)
        x[:, 0] = rng.normal(size=(n, C))
        e = rng.normal(scale=np.sqrt(1 - phi ** 2), size=(n, self.T, C))
        for t in range(1, self.T):
            x[:, t] = phi * x[:, t - 1] + e[:, t]
        return x

    def _raw(self, n, rng):
        W = self._ar1(rng, n, self.C, self.phi)
        load = self._ar1(rng, n, 1, self.load_phi)[:, :, 0]
        for c in self.ct:
            W[:, :, c] += self.load_w * load
        D = np.zeros_like(W)
        D[:, :, self.sig] = -self.delta
        onsets = rng.integers(self.onset[0], self.onset[1] + 1, size=n)
        return W, D, onsets

    def normal(self, split, n, rng):
        W, _, _ = self._raw(n, rng)
        return L.baseline_subtract(W)

    train_negatives = lambda self, n, rng: self.normal("train", n, rng)

    def construct(self, split, kind, n, rng, **kw):
        W, D, onsets = self._raw(n, rng)
        X = L.build_construct(kind, W, D, onsets, self.sig, rng, nonsig="normal",
                              off_pool=self.off_pool, **kw)
        return L.baseline_subtract(X)

    def unknown_faults(self, split, n, rng):
        """Other faults for the synthetic source: an equal mix of the overlap constructs."""
        ov = {"P1": 1, "P2": 2, "BIG1": 1, "ASYNC": 3, "ASYNC2": 3, "UP": 0, "OFF": 0}
        for kind, o in ov.items():
            yield f"toy:{kind}", self.construct(split, kind, n, rng), np.full(n, o), "other"
