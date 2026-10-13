"""
run_toy.py -- re-run the toy study from the slides with the same code path used on real data.

Purpose: (1) check that the NumPy LSTM and the shared test battery reproduce the slide-12
numbers before trusting them on LBNL data, (2) give a fast end-to-end smoke test.

Toy (slide 7): 20 channels, 48 steps, AR(1) phi=0.7 unit variance, shared slow "load" factor
on channels 0-5, capacitor fault = channels 0,1,2 drop by delta=2.5 together from a random
onset t in [10,25] to the end, baseline = mean of first 8 steps subtracted, 4000 windows 50/50.

    python run_toy.py                 # full battery (~5-10 min on a laptop CPU)
    python run_toy.py --quick         # smaller sweep, fewer LSTM epochs
"""
import argparse

import numpy as np

import shortcut_lib as L
from experiments import run_suite


class ToySource:
    def __init__(self, C=20, T=48, phi=0.7, delta=2.5, sig=(0, 1, 2), crosstalk=(0, 6),
                 load_phi=0.97, load_w=0.6, onset=(10, 25), nonsig="normal"):
        self.name = "toy (slide 7 replica)"
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
        """Toy 'fleet': other faults = equal mix of the overlap constructs."""
        ov = {"P1": 1, "P2": 2, "BIG1": 1, "ASYNC": 3, "ASYNC2": 3, "UP": 0, "OFF": 0}
        for kind, o in ov.items():
            yield f"toy:{kind}", self.construct(split, kind, n, rng), np.full(n, o), "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = dict(n_train=4000, n_val=2000, n_test=2000, n_other=1000, seed=a.seed, force=a.force,
               models=list(L.DETECTORS), sweep_sizes=[500, 2000, 8000, 20000],
               sweep_prev=[0.1, 0.5, 0.9], nonsig="normal")
    if a.quick:
        cfg.update(n_test=1000, n_other=500, sweep_sizes=[500, 2000], sweep_prev=[0.1, 0.5])
        L.DETECTORS["LSTM (raw series)"] = ("raw48", lambda s: L.NumpyLSTMClassifier(
            hidden=32, epochs=12, seed=s))
    run_suite(ToySource(), a.out, cfg)
    print("\nSlide 12 reference (LSTM): CAP 0.98  P1 0.20  P2 0.73  BIG1 0.91  ASYNC 0.11  "
          "ASYNC2 0.04  UP 0.00  OFF 0.01")


if __name__ == "__main__":
    main()
