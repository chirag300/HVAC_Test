"""
real_data.py -- turn an LBNL-style FDD dataset (one CSV per fault x severity, plus fault-free
CSV(s) simulated under the same weather/schedule) into the toy-study test battery.

Why this works on LBNL data: each faulty CSV is the same building/year as its fault-free
CSV, only with the fault imposed. So for any time window we have both the normal signals W
and the fault deviation D = faulty - normal, aligned sample-for-sample. That lets us:

  * build "onset" windows exactly like the toy: normal before a random onset, faulty after
  * find the target fault's signature channels (largest standardized deviation)
  * splice: keep the real deviation on only some signature channels (partial match), stagger
    it in time (ASYNC), amplify one channel (BIG1), flip it (UP), move it to unrelated
    channels (OFF) -- all from real fault physics, not synthetic steps
  * treat every OTHER fault file as an "unknown fault" never seen in training, and measure
    how often it is flagged as the target, bucketed by how many signature channels it moves

Schema-agnostic: the timestamp column, numeric channels and fault-free file(s) are detected
automatically. Check the detection with  `python run_real.py --data ... --inspect`.
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import shortcut_lib as L

NORMAL_RE = r"fault[\s_\-]*free|faultfree|baseline|no[\s_\-]*fault|\bnormal\b|\bff\b"
DROP_COLS_RE = r"fault|label|ground[\s_\-]*truth|\bclass\b|unnamed"
# setpoints are controller targets, not measurements; a fault should not change them
SETPOINT_RE = r"spt$|stpt$|setpoint|_sp_?set"
TIME_HINTS = ("datetime", "timestamp", "date", "time")


def _tokens(s):
    return set(t for t in re.split(r"[^a-z0-9]+", s.lower()) if t)


def default_family(stem):
    """Fault family = file stem with severity numbers removed (override with --family-regex)."""
    s = stem.lower()
    s = re.sub(r"\d+(\.\d+)?\s*(%|pct|percent|deg[cf]?|f|c)?", " ", s)
    s = re.sub(r"[^a-z]+", " ", s).strip()
    return s or stem


def _to_numeric(col):
    if col.dtype == bool:
        return col.astype(float)
    if col.dtype == object:
        low = col.astype(str).str.strip().str.lower()
        mapped = low.map({"true": 1.0, "false": 0.0, "on": 1.0, "off": 0.0})
        num = pd.to_numeric(col, errors="coerce")
        return num.where(num.notna(), mapped)
    return pd.to_numeric(col, errors="coerce")


def load_csv(path, resample, base_freq, cache_dir, log=print):
    st = Path(path).stat()
    key = hashlib.md5(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime}|{resample}|{base_freq}|v2".encode()).hexdigest()[:16]
    cpath = Path(cache_dir) / f"{key}.pkl"
    if cpath.exists():
        return pd.read_pickle(cpath)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    tcol = None
    for c in df.columns:
        if any(h in c.lower() for h in TIME_HINTS):
            tcol = c
            break
    idx = None
    if tcol is None:  # try the first column
        tcol = df.columns[0]
    parsed = pd.to_datetime(df[tcol], errors="coerce")
    if parsed.notna().mean() > 0.9:
        idx = parsed
        df = df.drop(columns=[tcol])
    else:
        log(f"    ! {Path(path).name}: no parseable timestamp, assuming {base_freq} spacing")
        idx = pd.Series(pd.date_range("2001-01-01", periods=len(df), freq=base_freq))
    out = pd.DataFrame({c: _to_numeric(df[c]) for c in df.columns})
    out.index = pd.DatetimeIndex(idx.values)
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated()].sort_index()
    out = out.loc[:, out.notna().mean() > 0.8]
    if resample:
        out = out.resample(resample).mean().interpolate(limit=2)
    out = out.astype(np.float32)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cpath)
    return out


class LBNLSource:
    def __init__(self, data_dir, target="auto", k=3, resample="15min", base_freq="1min", T=48,
                 onset=(10, 25), min_dev=0.5, dedup_corr=0.98, nonsig="normal",
                 normal_regex=NORMAL_RE, drop_cols_regex=DROP_COLS_RE, include_regex=None,
                 exclude_regex=None, family_regex=None, block="7D", split=(0.6, 0.2, 0.2),
                 cache_dir="data/cache", min_target_windows=500, seed=0, log=print,
                 setpoint_regex=SETPOINT_RE, mismatch_frac=0.8):
        self.log = log
        self.T, self.onset, self.min_dev = T, onset, min_dev
        self.nonsig = nonsig
        self.k_req = k
        self.dedup_corr = dedup_corr
        self.family_regex = family_regex
        self.min_target_windows = min_target_windows
        rng = np.random.default_rng(seed)
        root = Path(data_dir)

        # ---------------------------------------------------------- discover files
        files = [p for p in sorted(root.rglob("*.csv"))
                 if "__MACOSX" not in p.parts and not p.name.startswith("._")]
        if include_regex:
            files = [p for p in files if re.search(include_regex, str(p.relative_to(root)), re.I)]
        if exclude_regex:
            files = [p for p in files if not re.search(exclude_regex, str(p.relative_to(root)), re.I)]
        if not files:
            raise SystemExit(f"No CSV files under {root}. Run download.py first.")
        names = {}
        for p in files:
            n = p.stem if list(f.stem for f in files).count(p.stem) == 1 else \
                str(p.relative_to(root).with_suffix("")).replace("/", "__")
            names[n] = p
        is_norm = {n: bool(re.search(normal_regex, n, re.I)) for n in names}
        normals = [n for n in names if is_norm[n]]
        faults = [n for n in names if not is_norm[n]]
        if not normals:
            raise SystemExit("No fault-free file detected. Pass --normal-regex matching its name. "
                             f"Files: {list(names)[:20]}")
        log(f"  {len(normals)} fault-free file(s): {normals}")
        log(f"  {len(faults)} faulty file(s)")

        # ------------------------------------------------------------- load + align
        raw = {}
        for i, (n, p) in enumerate(names.items()):
            log(f"  loading [{i + 1}/{len(names)}] {p.name}")
            raw[n] = load_csv(p, resample, base_freq, cache_dir, log)
        cols = set.intersection(*(set(df.columns) for df in raw.values()))
        dropped_label = sorted(c for c in cols if re.search(drop_cols_regex, c, re.I) or
                               (setpoint_regex and re.search(setpoint_regex, c, re.I)))
        cols = sorted(c for c in cols if c not in dropped_label)
        if dropped_label:
            log(f"  excluded label-like / setpoint columns: {dropped_label}")

        self.pair = {}
        for f in faults:  # pair each fault with the most similar fault-free file
            ft = _tokens(f)
            self.pair[f] = max(normals, key=lambda nn: (len(ft & _tokens(nn)) / max(1, len(ft | _tokens(nn))),
                                                        len(raw[f].index.intersection(raw[nn].index))))

        # standardization from fault-free data in the train blocks (set below)
        self.block_split = {}
        Zraw = {nn: raw[nn][cols].to_numpy(np.float32) for nn in normals}
        for nn in normals:
            idx = raw[nn].index
            # sampling step: the resample rule, else the data's own median spacing
            step = pd.Timedelta(resample) if resample else \
                (pd.Series(idx).diff().median() if len(idx) > 1 else pd.Timedelta(base_freq))
            bsteps = max(self.T + 1, int(pd.Timedelta(block) / step))
            bid = np.arange(len(idx)) // bsteps
            ub = np.unique(bid)
            perm = rng.permutation(ub)
            n_tr, n_va = int(len(ub) * split[0]), int(len(ub) * split[1])
            code = np.empty(ub.max() + 1, np.int8)
            code[perm[:n_tr]] = 0
            code[perm[n_tr:n_tr + n_va]] = 1
            code[perm[n_tr + n_va:]] = 2
            self.block_split[nn] = (bid, code[bid])
        train_rows = np.concatenate([Zraw[nn][self.block_split[nn][1] == 0] for nn in normals])
        mu = np.nanmean(train_rows, 0)
        sd = np.nanstd(train_rows, 0)
        keep = sd > 1e-6 * (1 + np.abs(mu))
        # near-constant in fault-free data (>=95% of rows at one value): any change in a
        # fault file becomes a gigantic z-score that swamps everything else
        med = np.nanmedian(train_rows, 0)
        at_med = (np.abs(train_rows - med) <= 1e-6 * (1 + np.abs(med))).mean(0)
        keep &= at_med < 0.95
        const = [c for c, kk in zip(cols, keep) if not kk]
        if const:
            log(f"  dropped {len(const)} channel(s) constant in fault-free data: {const}")
        self.channel_names = [c for c, kk in zip(cols, keep) if kk]
        self.C = len(self.channel_names)
        mu, sd = mu[keep], sd[keep]
        log(f"  {self.C} channels used")

        self.Zn = {nn: ((Zraw[nn][:, keep] - mu) / sd).astype(np.float32) for nn in normals}
        self.D = {}
        for f in faults:
            nn = self.pair[f]
            zf = raw[f][cols].reindex(raw[nn].index).to_numpy(np.float32)[:, keep]
            cover = np.isfinite(zf).all(1).mean()
            if cover < 0.5:
                log(f"  ! {f}: only {cover:.0%} timestamp overlap with {nn}; skipped")
                continue
            self.D[f] = ((zf - mu) / sd - self.Zn[nn]).astype(np.float32)
        del raw, Zraw
        self.faults = list(self.D)
        self.normals = normals
        self.family = {f: self._family(f) for f in self.faults}

        # normal-data correlation (for dedup and OFF pool)
        tr = np.concatenate([self.Zn[nn][self.block_split[nn][1] == 0] for nn in normals])
        tr = tr[np.isfinite(tr).all(1)]
        self.corr = np.nan_to_num(np.corrcoef(tr.T)) if len(tr) > 10 else np.eye(self.C)
        self._starts_cache = {}

        # --------------------------------------------- fault profiles and target
        self.profiles = {f: self._profile(f) for f in self.faults}
        self._drop_baseline_mismatch(mismatch_frac)
        self.ranking = self._rank_targets()
        if target == "auto":
            ok = [r for r in self.ranking if r["est_train_windows"] >= min_target_windows]
            if not ok:
                raise SystemExit("No fault has enough qualifying windows; lower --min-dev or --k.")
            target = ok[0]["fault"]
        else:
            hits = [f for f in self.faults if target.lower() == f.lower()] or \
                   [f for f in self.faults if target.lower() in f.lower()]
            if not hits:
                raise SystemExit(f"--target {target!r} matches no fault file")
            target = hits[0]
        self.target = target
        self.sig, self.signs = self._signature(target)
        self.name = f"{root.name}: target={target} ({nonsig} non-signature channels)"
        tn = self.pair[target]
        self.off_pool = [c for c in range(self.C) if c not in self.sig and
                         np.abs(self.corr[c, self.sig]).max() < 0.5] or \
                        [c for c in range(self.C) if c not in self.sig]
        self.sig_info = {
            "target_file": target, "paired_fault_free": tn,
            "signature": {self.channel_names[c]: round(float(self.profiles[target]["d"][c]), 3)
                          for c in self.sig},
            "next_strongest": {self.channel_names[c]: round(float(self.profiles[target]["d"][c]), 3)
                               for c in np.argsort(-np.abs(self.profiles[target]["d"]))[:8]
                               if c not in self.sig},
        }
        log(f"  target: {target}; signature: {self.sig_info['signature']}")
        if target in self._fallback:
            log("  ! signature is a FALLBACK: fewer than 2 channels passed --min-dev after skipping "
                "near-duplicates (|corr| > --dedup-corr). It includes weak or duplicate channels. "
                "Consider --min-dev lower, --dedup-corr 1.0, or another --target.")
        self.sig_info["strict_signature"] = target not in self._fallback

    # ----------------------------------------------------------------- helpers
    def _drop_baseline_mismatch(self, frac_thr):
        """A channel that moves the same way in (almost) every fault family is a difference
        between the fault-free run and the faulty runs (different setpoint, control version,
        period), not a fault effect. Left in, it lets every model separate 'any fault file'
        from 'fault-free' through that channel alone. Drop such channels."""
        fams = defaultdict(list)
        for f in self.faults:
            fams[self.family[f]].append(self.profiles[f]["d"])
        if len(fams) < 3:
            return
        M = np.array([np.median(v, 0) for v in fams.values()])  # (families, C)
        dom = np.sign(np.median(M, 0))
        frac = ((np.abs(M) >= self.min_dev) & (np.sign(M) == dom)).mean(0)
        bad = np.flatnonzero(frac >= frac_thr)
        if len(bad) == 0:
            return
        self.log(f"  ! dropped {len(bad)} channel(s) that differ between the fault-free file and "
                 f"{frac_thr:.0%}+ of fault families (baseline mismatch, not fault effects): "
                 + ", ".join(f"{self.channel_names[c]} ({frac[c]:.0%})" for c in bad))
        keep = np.setdiff1d(np.arange(self.C), bad)
        self.channel_names = [self.channel_names[c] for c in keep]
        self.C = len(keep)
        self.Zn = {k: v[:, keep] for k, v in self.Zn.items()}
        self.D = {k: v[:, keep] for k, v in self.D.items()}
        self.corr = self.corr[np.ix_(keep, keep)]
        self.profiles = {f: self._profile(f) for f in self.faults}

    def _family(self, f):
        if self.family_regex:
            m = re.search(self.family_regex, f)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return default_family(f)

    def _profile(self, f):
        nn = self.pair[f]
        tr = self.block_split[nn][1] == 0
        D = self.D[f][tr]
        D = D[np.isfinite(D).all(1)]
        if len(D) == 0:
            return {"d": np.zeros(self.C), "active_frac": 0.0}
        act = np.abs(D).max(1) >= self.min_dev  # any channel visibly moved
        d = D[act].mean(0) if act.any() else np.zeros(self.C)
        return {"d": d, "active_frac": float(act.mean())}

    def _signature(self, f, k=None):
        d = self.profiles[f]["d"]
        k = k or self.k_req
        sel = []
        for c in np.argsort(-np.abs(d)):
            if len(sel) == k:
                break
            if abs(d[c]) < self.min_dev:
                break
            if sel and np.abs(self.corr[c, sel]).max() > self.dedup_corr:
                continue
            sel.append(int(c))
        strict = len(sel) >= 2
        if not strict:  # fall back to top channels even if weak or near-duplicate
            sel = [int(c) for c in np.argsort(-np.abs(d))[:max(2, k)]]
        self._fallback = getattr(self, "_fallback", set())
        (self._fallback.discard if strict else self._fallback.add)(f)
        return sel, np.sign(d[sel]).astype(int)

    def _rank_targets(self):
        rows = []
        rng = np.random.default_rng(123)
        for f in self.faults:
            sig, signs = self._signature(f)
            pairs = self._qualifying(f, sig, signs, 0, 400, rng, max_rounds=3, return_rate=True)
            frac, n_starts = pairs
            partial = 0
            fam = self.family[f]
            others = defaultdict(list)
            for g in self.faults:
                if self.family[g] == fam:
                    continue
                dg = self.profiles[g]["d"]
                ov = int(sum(s * dg[c] >= self.min_dev for c, s in zip(sig, signs)))
                others[self.family[g]].append(ov)
            for fam_g, ovs in others.items():
                if any(0 < o < len(sig) for o in ovs):
                    partial += 1
            rows.append({"fault": f, "family": fam, "k": len(sig),
                         "signature": [self.channel_names[c] for c in sig],
                         "qualify_rate": round(frac, 3),
                         "est_train_windows": int(frac * n_starts * (self.onset[1] - self.onset[0] + 1)),
                         "families_with_partial_overlap": partial,
                         "strict_signature": f not in self._fallback,
                         "active_frac": round(self.profiles[f]["active_frac"], 3)})
        rows.sort(key=lambda r: (-(r["est_train_windows"] >= self.min_target_windows), -r["strict_signature"],
                                 -(r["k"] == self.k_req),
                                 -r["families_with_partial_overlap"], -r["qualify_rate"]))
        return rows

    def _starts(self, nn, split, f=None):
        key = (nn, split, f)
        if key not in self._starts_cache:
            bid, sp = self.block_split[nn]
            T = self.T
            n = len(bid) - T + 1
            if n <= 0:
                self._starts_cache[key] = np.zeros(0, int)
                return self._starts_cache[key]
            ok = (bid[:n] == bid[T - 1:]) & (sp[:n] == split)
            for A in [self.Zn[nn]] + ([self.D[f]] if f else []):
                bad = np.concatenate([[0], np.cumsum(~np.isfinite(A).all(1))])
                ok &= (bad[T:] - bad[:-T]) == 0
            self._starts_cache[key] = np.flatnonzero(ok)
        return self._starts_cache[key]

    def _post_means(self, A, s, o):
        """mean of A[s+o : s+T] per column, A (Tn, m)."""
        cs = np.concatenate([np.zeros((1, A.shape[1]), np.float64), np.cumsum(np.nan_to_num(A), 0)])
        return (cs[s + self.T] - cs[s + o]) / (self.T - o)[:, None]

    def _qualifying(self, f, sig, signs, split, n, rng, max_rounds=40, return_rate=False):
        """Sample (start, onset) pairs where every signature channel deviates >= min_dev
        in its fault direction after onset (i.e. the target fault is clearly present)."""
        nn = self.pair[f]
        starts = self._starts(nn, split, f)
        if len(starts) == 0:
            return (0.0, 0) if return_rate else (np.zeros(0, int), np.zeros(0, int))
        S = self.D[f][:, sig] * np.asarray(signs)[None, :]
        got_s, got_o, tried, hit = [], [], 0, 0
        for _ in range(max_rounds):
            m = max(4 * n, 2000)
            s = rng.choice(starts, m)
            o = rng.integers(self.onset[0], self.onset[1] + 1, m)
            q = (self._post_means(S, s, o) >= self.min_dev).all(1)
            tried += m
            hit += int(q.sum())
            got_s.append(s[q])
            got_o.append(o[q])
            if sum(len(x) for x in got_s) >= n:
                break
        if return_rate:
            return hit / max(1, tried), len(starts)
        s, o = np.concatenate(got_s)[:n], np.concatenate(got_o)[:n]
        if len(s) < n:
            self.log(f"    ! only {len(s)}/{n} qualifying windows for {f} in split {split}")
        return s, o

    def _windows(self, A, s):
        return A[s[:, None] + np.arange(self.T)[None, :]]

    # ------------------------------------------------------------ source API
    SPLIT = {"train": 0, "val": 1, "test": 2}

    def normal(self, split, n, rng):
        sp = self.SPLIT[split]
        pools = [(nn, self._starts(nn, sp)) for nn in self.normals]
        pools = [(nn, st) for nn, st in pools if len(st)]
        w = np.array([len(st) for _, st in pools], float)
        pick = rng.choice(len(pools), n, p=w / w.sum())
        out = []
        for i, (nn, st) in enumerate(pools):
            m = int((pick == i).sum())
            if m:
                out.append(self._windows(self.Zn[nn], rng.choice(st, m)))
        X = np.concatenate(out)
        return L.baseline_subtract(X[rng.permutation(len(X))])

    def train_negatives(self, n, rng):
        """Half matched (same timestamps as target-fault windows, fault removed), half random."""
        n_m = n // 2
        s, _ = self._qualifying(self.target, self.sig, self.signs, 0, n_m, rng)
        Xm = self._windows(self.Zn[self.pair[self.target]], s)
        Xr = self.normal("train", n - len(s), rng)
        return np.concatenate([L.baseline_subtract(Xm), Xr])

    def construct(self, split, kind, n, rng, **kw):
        f = self.target
        s, o = self._qualifying(f, self.sig, self.signs, self.SPLIT[split], n, rng)
        W = self._windows(self.Zn[self.pair[f]], s)
        D = self._windows(self.D[f], s)
        X = L.build_construct(kind, W, D, o, self.sig, rng, nonsig=self.nonsig,
                              off_pool=self.off_pool, **kw)
        return L.baseline_subtract(X)

    def unknown_faults(self, split, n, rng):
        sp = self.SPLIT[split]
        tfam = self.family[self.target]
        for f in self.faults:
            if f == self.target:
                continue
            nn = self.pair[f]
            starts = self._starts(nn, sp, f)
            if len(starts) == 0:
                continue
            m = max(20 * n, 5000)
            s = rng.choice(starts, m)
            o = rng.integers(self.onset[0], self.onset[1] + 1, m)
            post = self._post_means(self.D[f], s, o)
            active = np.abs(post).max(1) >= self.min_dev  # this fault visibly present
            s, o, post = s[active][:n], o[active][:n], post[active][:n]
            if len(s) == 0:
                continue
            overlap = (post[:, self.sig] * self.signs[None, :] >= self.min_dev).sum(1)
            W = self._windows(self.Zn[nn], s)
            D = self._windows(self.D[f], s)
            X = W.copy()
            for i in range(len(s)):
                X[i, o[i]:] += D[i, o[i]:]
            group = "same_family" if self.family[f] == tfam else "other"
            yield f, L.baseline_subtract(X), overlap, group

    # ------------------------------------------------------------ inspection
    def describe(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prof = pd.DataFrame({f: self.profiles[f]["d"] for f in self.faults},
                            index=self.channel_names).T
        prof.insert(0, "family", [self.family[f] for f in prof.index])
        prof.insert(1, "paired_fault_free", [self.pair[f] for f in prof.index])
        prof.insert(2, "active_frac", [self.profiles[f]["active_frac"] for f in prof.index])
        prof.round(3).to_csv(out / "fault_profiles_sd_units.csv")
        pd.DataFrame(self.ranking).to_csv(out / "target_ranking.csv", index=False)
        ov = {f: {self.channel_names[c]: round(float(s * self.profiles[f]["d"][c]), 2)
                  for c, s in zip(self.sig, self.signs)} for f in self.faults}
        with open(out / "dataset_summary.json", "w") as fh:
            json.dump({"channels": self.channel_names, "fault_free": self.normals,
                       "pairing": self.pair, "families": self.family, "target": self.sig_info,
                       "signed_deviation_on_target_signature": ov}, fh, indent=1)
        lines = ["Top target candidates (more partial-overlap families = closer to the slide's "
                 "'other faults look like part of the capacitor pattern' assumption):"]
        for r in self.ranking[:15]:
            fb = "" if r["strict_signature"] else "  [fallback signature]"
            lines.append(f"  {r['fault'][:45]:45s} k={r['k']} partial-overlap families="
                         f"{r['families_with_partial_overlap']:2d} qualify={r['qualify_rate']:.3f} "
                         f"~{r['est_train_windows']} windows  sig={r['signature']}{fb}")
        return "\n".join(lines)
