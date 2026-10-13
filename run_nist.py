"""
run_nist.py -- tabular version of the battery for the NIST residential heat-pump FDD data
(fdd_heat_pump_cooling_1416seer_longshort_alldata_2019.xlsx, 14 and 16 SEER units, cooling
mode, single faults imposed at several levels: e.g. refrigerant under/overcharge, condenser /
evaporator airflow reduction, liquid-line restriction, non-condensables, compressor valve leak).

Each row is treated as one sample (no time axis), so the timing tests (ASYNC, ASYNC2) and the
LSTM do not apply; the channel tests do: partial match (P1..P{k-1}), one big channel (BIG1),
direction (UP), unrelated channels (OFF), other real faults by signature overlap, more data /
rebalancing, and targeted all-but-one negatives. Gradient-descent model: the MLP (Adam).

  # 1. find the sheet, the fault-label column and the operating-condition columns
  python run_nist.py --xlsx data/raw/nist/fdd_heat_pump_cooling_1416seer_longshort_alldata_2019.xlsx --inspect

  # 2. run (example: label column 'Fault', fault-free label matching 'none|ff|free',
  #    pairing each faulty test with the nearest fault-free test in outdoor/indoor temperature)
  python run_nist.py --xlsx <file> --sheet <name> --fault-col Fault --normal-regex "none|^ff$|free" \
      --cond-cols "T_outdoor,T_indoor_db,T_indoor_wb" --target auto --out results/nist

If the file turns out to hold time series per test (one block of rows per test), pass
--group-col <test id column> so train/val/test are split by test, not by row.
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import shortcut_lib as L
from experiments import run_suite

# keep labels like "None" / "NA" as text (pandas would turn them into NaN by default)
NA_KW = dict(keep_default_na=False, na_values=["", "#N/A", "NaN", "nan", "-", "N/A"])
ID_LIKE = r"(^|_)(id|index|run|test|time|date|file|no\.?|number)($|_)"


def inspect(path):
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        df = xl.parse(sh, nrows=2000, **NA_KW)
        print(f"\n=== sheet {sh!r}: {df.shape[0]}+ rows x {df.shape[1]} cols")
        for c in df.columns:
            col = df[c]
            num = pd.to_numeric(col, errors="coerce").notna().mean()
            nun = col.nunique(dropna=True)
            kind = "numeric" if num > 0.8 else "text"
            ex = ", ".join(map(str, col.dropna().unique()[:6]))
            flag = "  <-- label candidate" if (kind == "text" and 1 < nun <= 60) or \
                re.search("fault|level|type|case", str(c), re.I) else ""
            print(f"  {str(c)[:40]:40s} {kind:7s} unique={nun:5d}  e.g. {ex[:70]}{flag}")


class NISTSource:
    test_kinds = ["BIG1", "UP", "OFF"]
    and_kwargs = {"win": 1, "start": 0}

    def __init__(self, path, sheet, fault_col, normal_regex, target="auto", level_col=None,
                 cond_cols=(), group_col=None, k=3, min_dev=0.5, dedup_corr=0.98,
                 drop_regex=ID_LIKE, seed=0, log=print):
        rng = np.random.default_rng(seed)
        df = pd.read_excel(path, sheet_name=sheet, **NA_KW)
        df.columns = [str(c).strip() for c in df.columns]
        lab = df[fault_col].astype(str).str.strip()
        if level_col:
            lab = lab + " @" + df[level_col].astype(str)
        is_norm = df[fault_col].astype(str).str.contains(normal_regex, case=False, regex=True).to_numpy()
        if not is_norm.any():
            raise SystemExit(f"no fault-free rows match {normal_regex!r} in {fault_col}: "
                             f"{df[fault_col].astype(str).unique()[:20]}")
        excl = {fault_col, level_col, group_col, *cond_cols} - {None}
        num = {}
        for c in df.columns:
            if c in excl or re.search(drop_regex, c, re.I):
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().mean() > 0.8:
                num[c] = v
        F = pd.DataFrame(num)
        F = F.fillna(F[is_norm].median())
        cond = df[list(cond_cols)].apply(pd.to_numeric, errors="coerce") if cond_cols else None

        # split by group (test) if given, else by row
        groups = df[group_col].astype(str).to_numpy() if group_col else np.arange(len(df)).astype(str)
        ug = np.unique(groups)
        perm = rng.permutation(ug)
        code = {g: (0 if i < 0.6 * len(ug) else 1 if i < 0.8 * len(ug) else 2) for i, g in enumerate(perm)}
        self.split_of = np.array([code[g] for g in groups])

        tr_norm = is_norm & (self.split_of == 0)
        mu, sd = F[tr_norm].mean(), F[tr_norm].std()
        keep = sd > 1e-9
        self.channel_names = list(F.columns[keep])
        self.C = len(self.channel_names)
        Z = ((F.loc[:, keep] - mu[keep]) / sd[keep]).to_numpy(np.float32)
        log(f"  {len(df)} rows, {is_norm.sum()} fault-free, {self.C} feature channels")
        self.Z, self.is_norm, self.lab = Z, is_norm, lab.to_numpy()
        self.family = {v: df[fault_col].astype(str).str.strip()[lab == v].iloc[0] for v in np.unique(self.lab)}

        # matched fault-free reference per row (nearest in operating conditions, same split)
        self.ref = np.full(len(df), -1)
        if cond is not None:
            cz = ((cond - cond[is_norm].mean()) / cond[is_norm].std()).fillna(0).to_numpy()
            for sp in range(3):
                nidx = np.flatnonzero(is_norm & (self.split_of == sp))
                fidx = np.flatnonzero(~is_norm & (self.split_of == sp))
                if len(nidx) == 0:
                    continue
                d = ((cz[fidx, None, :] - cz[None, nidx, :]) ** 2).sum(-1)
                self.ref[fidx] = nidx[d.argmin(1)]
        else:
            log("  ! no --cond-cols: deviations are measured against the mean fault-free row "
                "(weaker; operating-condition effects leak into the signature)")
        self.min_dev, self.k_req, self.dedup = min_dev, k, dedup_corr
        nz = Z[tr_norm]
        self.corr = np.nan_to_num(np.corrcoef(nz.T)) if len(nz) > 3 else np.eye(self.C)
        self.faults = [v for v in np.unique(self.lab[~is_norm])]
        self.profiles = {f: self._dev(np.flatnonzero((self.lab == f) & (self.split_of == 0))).mean(0)
                         if ((self.lab == f) & (self.split_of == 0)).any() else np.zeros(self.C)
                         for f in self.faults}
        rank = []
        for f in self.faults:
            sig, signs = self._sig(f)
            qual = self._qual_rows(f, sig, signs, None)
            partial = len({self.family[g] for g in self.faults if self.family[g] != self.family[f] and
                           0 < sum(s * self.profiles[g][c] >= min_dev for c, s in zip(sig, signs)) < len(sig)})
            rank.append((len(sig) == k, partial, len(qual), f))
        rank.sort(reverse=True)
        self.ranking = rank
        if target == "auto":
            target = rank[0][3]
        else:
            hits = [f for f in self.faults if re.search(target, f, re.I)]
            if not hits:
                raise SystemExit(f"--target {target!r} matches none of {self.faults}")
            target = max(hits, key=lambda f: np.abs(self.profiles[f]).max())
        self.target = target
        self.sig, self.signs = self._sig(target)
        self.off_pool = [c for c in range(self.C) if c not in self.sig and
                         np.abs(self.corr[c, self.sig]).max() < 0.5] or \
                        [c for c in range(self.C) if c not in self.sig]
        self.name = f"NIST heat pump: target={target}"
        self.nonsig = "normal"
        self.sig_info = {"target": target, "signature": {self.channel_names[c]: round(float(self.profiles[target][c]), 3)
                                                         for c in self.sig}}
        log(f"  target {target}: {self.sig_info['signature']}")
        log("  ranking (full k, partial-overlap families, qualifying rows, label):")
        for r in rank[:12]:
            log(f"    {r}")

    def _dev(self, idx):
        if len(idx) == 0:
            return np.zeros((0, self.C), np.float32)
        if (self.ref[idx] >= 0).all():
            return self.Z[idx] - self.Z[self.ref[idx]]
        return self.Z[idx] - self.Z[self.is_norm].mean(0)

    def _sig(self, f):
        d = self.profiles[f]
        sel = []
        for c in np.argsort(-np.abs(d)):
            if len(sel) == self.k_req or abs(d[c]) < self.min_dev:
                break
            if sel and np.abs(self.corr[c, sel]).max() > self.dedup:
                continue
            sel.append(int(c))
        if len(sel) < 2:
            sel = [int(c) for c in np.argsort(-np.abs(d))[:max(2, self.k_req)]]
        return sel, np.sign(d[sel]).astype(int)

    def _qual_rows(self, f, sig, signs, split):
        idx = np.flatnonzero((self.lab == f) & ((self.split_of == split) if split is not None else True))
        D = self._dev(idx)
        ok = (D[:, sig] * signs >= self.min_dev).all(1) if len(D) else np.zeros(0, bool)
        return idx[ok]

    SPLIT = {"train": 0, "val": 1, "test": 2}

    def _pairs(self, split, n, rng):
        rows = self._qual_rows(self.target, self.sig, self.signs, self.SPLIT[split])
        if len(rows) == 0:
            raise SystemExit(f"no qualifying {self.target} rows in split {split}; lower --min-dev")
        r = rng.choice(rows, n)
        W = (self.Z[self.ref[r]] if (self.ref[r] >= 0).all()
             else np.tile(self.Z[self.is_norm].mean(0), (n, 1)))
        return W[:, None, :].astype(np.float32), self._dev(r)[:, None, :].astype(np.float32)

    def normal(self, split, n, rng):
        idx = np.flatnonzero(self.is_norm & (self.split_of == self.SPLIT[split]))
        if len(idx) == 0:
            idx = np.flatnonzero(self.is_norm)
        return self.Z[rng.choice(idx, n)][:, None, :].copy()

    train_negatives = lambda self, n, rng: self.normal("train", n, rng)

    def construct(self, split, kind, n, rng, **kw):
        W, D = self._pairs(split, n, rng)
        return L.build_construct(kind, W, D, np.zeros(n, int), self.sig, rng, nonsig="normal",
                                 off_pool=self.off_pool, **kw)

    def unknown_faults(self, split, n, rng):
        tfam = self.family[self.target]
        for f in self.faults:
            if f == self.target:
                continue
            idx = np.flatnonzero((self.lab == f) & (self.split_of == self.SPLIT[split]))
            if len(idx) == 0:
                continue
            D = self._dev(idx)
            act = np.abs(D).max(1) >= self.min_dev
            idx, D = idx[act], D[act]
            if len(idx) == 0:
                continue
            ov = (D[:, self.sig] * self.signs >= self.min_dev).sum(1)
            yield f, self.Z[idx][:, None, :].copy(), ov, \
                ("same_family" if self.family[f] == tfam else "other")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--fault-col")
    ap.add_argument("--level-col", default=None)
    ap.add_argument("--normal-regex", default=r"none|no fault|fault[- ]?free|^ff$|baseline|normal")
    ap.add_argument("--cond-cols", default="")
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--target", default="auto")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-dev", type=float, default=0.5)
    ap.add_argument("--out", default="results/nist")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.inspect or not a.fault_col:
        inspect(a.xlsx)
        if not a.fault_col:
            print("\nRe-run with --fault-col (and ideally --cond-cols) set from the listing above.")
        return
    sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet
    src = NISTSource(a.xlsx, sheet, a.fault_col, a.normal_regex, a.target, a.level_col,
                     [c for c in a.cond_cols.split(",") if c], a.group_col, a.k, a.min_dev, seed=a.seed)
    models = [m for m in L.DETECTORS if not m.startswith("LSTM")]
    cfg = dict(n_train=a.n_train, n_val=2000, n_test=1000, n_other=500, seed=a.seed, force=a.force,
               models=models, nonsig="normal", sweep_sizes=[500, 2000, 8000], sweep_prev=[0.1, 0.5, 0.9])
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.target)[:80]
    run_suite(src, f"{a.out}/{safe}", cfg)
    print("\nNote: rows are resampled with replacement to reach the requested set sizes; "
          "check 'qualifying rows' in the ranking above for how much distinct data there is.")


if __name__ == "__main__":
    main()
