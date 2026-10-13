# When fault detectors learn the wrong rule — real-data test

Re-runs the toy study ("narrow multi-sensor fault signature vs. the broader rule models
learn") on public HVAC fault datasets, with an LSTM trained by gradient descent (Adam) as the
main model, plus LogReg / random forest / gradient boosting / MLP for comparison.

No PyTorch. The LSTM (forward, backprop-through-time, Adam) is written in NumPy
(`lstm_numpy.py`, finite-difference gradient check included), so everything runs on a plain
CPU Python with numpy, pandas, scikit-learn, matplotlib.

## Files

| File | What it does |
|---|---|
| `lstm_numpy.py` | NumPy LSTM classifier (BPTT + Adam, grad clipping, early stopping). `python lstm_numpy.py` = gradient check |
| `shortcut_lib.py` | Window stats, the splice constructs (CAP, P1..Pk-1, BIG1, ASYNC, ASYNC2, UP, OFF), the true AND rule, detectors, 1%-FPR calibration |
| `experiments.py` | The test battery + hypothesis verdicts + REPORT.md + figures (shared by toy, LBNL, NIST) |
| `run_toy.py` | Slide-7 toy replica through the same code — check you reproduce slide 12 first |
| `download.py` | Downloads + unpacks LBNL SDAHU / RTU / FCU and NIST heat-pump data into `data/raw/` |
| `real_data.py` | Schema-agnostic LBNL loader; pairs each fault CSV with its fault-free twin; builds windows |
| `run_real.py` | Runs the battery on one LBNL dataset |
| `run_nist.py` | Tabular version for the NIST heat-pump spreadsheet (no time axis → no LSTM / ASYNC) |
| `ci_numpy.py` | Channel-independent detector (shared per-channel encoder + linear head, additive across channels), Adam/backprop in NumPy, gradient check |
| `tep_to_csv.py` | Tennessee Eastman (Braatz runs) -> LBNL-style CSVs for `run_real.py` (see TEP note below) |
| `make_fake_lbnl.py` | Fake dataset in LBNL layout with a known answer, for smoke tests |
| `run_all.sh`, `smoke_test.sh` | One-shot pipelines |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python lstm_numpy.py          # gradient check -> PASS
bash smoke_test.sh            # ~5 min, fake data, exercises every step
```

## Full process

### 0. Reproduce the toy first (validates the NumPy LSTM against your slide 12)
```bash
python run_toy.py --out results/toy          # ~15-25 min CPU; --quick for ~3 min
```
Expected (LSTM row, slide 12): CAP 0.98, P1 0.20, P2 0.73, BIG1 0.91, ASYNC 0.11, UP 0.00, OFF 0.01.

### 1. Download
```bash
python download.py sdahu fcu rtu nist
```
LBNL puts its zips behind a short registration form. If a link returns the form instead of a
zip, the script prints the exact file name and folder to save it to from your browser; re-run
the same command and it unpacks. Sizes: SDAHU ≈ 590 MB, RTU ≈ 290 MB, FCU ≈ 520 MB.

### 2. Inspect before running (do not skip)
```bash
python run_real.py --data data/raw/sdahu --out results/sdahu --inspect
```
Prints and saves to `results/sdahu/inspect/`:
* which CSV(s) were detected as fault-free and how every fault file was paired with one
* channels used (label/ground-truth columns and channels constant in fault-free data are dropped)
* `fault_profiles_sd_units.csv` — every fault's mean deviation per channel (fault-free SD units)
* `target_ranking.csv` — candidate targets, ranked by (enough windows, full k-channel signature,
  number of other fault families that partially overlap its signature)

Check: the fault-free file is right (`--normal-regex` if not), severities of the same fault
share a family (`--family-regex` if not), and the chosen signature is physically sensible.
If almost nothing qualifies, lower `--min-dev` (default 0.5 SD).

The "capacitor-like" target the hypothesis needs is a fault whose signature is several
channels moving together AND whose partial pattern is shared by other faults. The auto pick
optimizes exactly that; you can force one with `--target <substring of file name>`.

### 3. Run
```bash
python run_real.py --data data/raw/sdahu --out results/sdahu            # auto target
python run_real.py --data data/raw/fcu   --out results/fcu
python run_real.py --data data/raw/rtu   --out results/rtu --target <name>
```
Runtime: ~20–60 min per dataset on a laptop CPU (the LSTM sweep dominates). Every step is
cached as JSON and the sweep is resumable cell by cell; `--force` recomputes.
`--steps main` runs just the main table.

### 4. NIST heat pump (tabular)
```bash
python run_nist.py --xlsx data/raw/nist/fdd_heat_pump_cooling_1416seer_longshort_alldata_2019.xlsx --inspect
python run_nist.py --xlsx <file> --sheet <sheet> --fault-col "<label col>" [--level-col "<severity col>"] \
    --cond-cols "<outdoor T>,<indoor T>" --target <fault regex> --out results/nist
```
`--inspect` lists every sheet's columns and flags likely label columns; `--cond-cols` pairs
each faulty test with the nearest fault-free test in operating conditions.
Your memory notes say NIST is reserved for zero-shot evaluation; if that still holds, use only
`--inspect` now and keep the run for later.

## How the toy maps onto real data

LBNL simulated datasets are paired: every faulty CSV is the same building, weather and
schedule as the fault-free CSV, with one fault imposed. So for any window we know the normal
signals **W** and the fault's exact effect **D = faulty − normal**. Every test is a splice
`X = W + G·D` (G says which channels / time steps carry the real deviation):

| Toy (slide 9) | Real-data version |
|---|---|
| Capacitor: ch 0,1,2 drop together from random onset | Target fault's real deviation on its top-k signature channels, from a random onset (t = 10–25 of 48) |
| P1, P2 | Real deviation kept on only 1 or k−1 signature channels |
| BIG1 | One signature channel's real deviation ×3 |
| ASYNC / ASYNC2 | Each channel's real deviation in its own time segment / short separate dips |
| UP | Deviation sign flipped |
| OFF | Deviation moved onto k unrelated (|corr| < 0.5) channels |
| "Other faults exist, unlabeled" | Every other fault file, never seen in training, bucketed by how many signature channels it actually moves |
| Fleet 89/1/10 | Same mix using real other-fault windows |

Signature = the k channels with the largest mean standardized deviation while the fault is
visibly present (near-duplicate sensors, |corr| > 0.98, are skipped). A training-positive
window must have **every** signature channel moved ≥ `--min-dev` after onset, i.e. the narrow
AND really holds in the data the model is trained on.

Two variants are run (`--nonsig both`):
* **clean** — only the signature channels carry the deviation: the slide's mechanism on real signals
* **real** — training positives are full faulty windows (every channel deviates). Harder to
  interpret (partial constructs still carry real deviations elsewhere) but closer to deployment

Other choices: windows of 48 × 15 min (12 h); per-window baseline = mean of first 8 steps;
train/val/test split by whole weeks (no overlapping-window leakage); half of the training
negatives are the same timestamps as the positives with the fault removed (so the model
cannot key on season/time of day); thresholds at 1% false alarms on held-out normal windows.

**True AND rule.** On HVAC data the signature channels also move together for normal reasons
(fan start/stop moves speed, power and airflow at once). The reference rule therefore works on
residuals: each signature channel minus a ridge prediction from the non-signature channels,
scaled by its own residual noise, and requires all of them up in the same 8-step stretch.
It is a noise-floor reference: with slow, autocorrelated residuals it can pass some partial
matches by chance, so H2 is judged as a gap (model − rule), not against zero.

## Targeted-negative variants and the channel-independent model

Each run trains three fixes, all with the same number of extra label-0 windows (n_train/2):

| Step | Negatives added | Tests |
|---|---|---|
| `fix` | all-but-one: k−1 signature channels move | channel identifiability (H5) |
| `fixtime` | staggered: all k move, but at different times (ASYNC, ASYNC2) | time identifiability (H5t) |
| `fixboth` | half of each | both (H5b) |

`Channel-indep. (raw series)` is additive across channels by construction, like
channel-independent time-series models with a linear head. H8 checks the theory's prediction
that it stays fooled by partial matches after the channel fix while the LSTM does not.

## Tennessee Eastman (TEP)

`python tep_to_csv.py` downloads the classic Braatz runs (one fault-free and 21 faulty test
runs, 960 samples at 3 min, fault from sample 160) and writes LBNL-style CSVs.
`run_real.py --data data/raw/tep --resample none --block 4h --inspect` works and is useful:
it shows that IDV4 (reactor cooling-water step) is effectively a single-channel fault
(`XMV_10` only) and that IDV11/IDV14 are variance faults, so 4/11/14 is not a conjunction
test; IDV5 (condenser cooling water, `XMV_11` + `XMEAS_11` + `XMEAS_35`) is the closer analogue.
The full battery needs more than one 48-hour run per fault (controllers compensate IDV5 within
hours, leaving too few test windows), so use the multi-run TEP dataset (Rieth et al. 2017,
Harvard Dataverse) for real results. The fault-free and faulty Braatz runs use different
random seeds, so their deviations are not exact counterfactuals.

## Hypothesis verdicts (top of each REPORT.md)

| Check | Slide claim | Criterion |
|---|---|---|
| H1 | One channel almost separates training data (slide 15 table) | best single-channel CV acc ≥ 0.90 and within 0.08 of all channels |
| H2 | Unseen partial faults fool the model (slide 12) | LSTM flags ≥ 30% of P{k−1}, and ≥ 0.20 more than the AND rule |
| H2b | Every model family fails the same way | all models flag ≥ 30% of P{k−1} |
| H3 | The true rule scores lower on the training task | AND-rule CAP recall < mean learned recall |
| H4 | More data / balancing does not help (slide 15) | LSTM P{k−1} false alarms ≥ 0.30 across the sweep |
| H5 | Targeted negatives help (slide 18) | P{k−1} false alarms halve, CAP recall drops ≤ 0.05 |
| H6 (per model) | Real unknown faults that overlap the signature get flagged | partial-overlap flag rate ≥ 0.20, ≥ 0.15 above and ≥ 2× the no-overlap rate |
| H5t / H5b | Staggered negatives fix simultaneity; both kinds fix both | ASYNC/ASYNC2 (and P{k−1}) halve, CAP drops ≤ 0.05 |
| H8 | Additive (channel-independent) models stay fooled after the channel fix | channel-indep. P{k−1} ≥ LSTM's + 0.10 after `fix` |
| H7 (per model) | Most fleet false alarms come from unknown faults (slide 13) | ≥ 50% in the 89/1/10 mix; INCONCLUSIVE if unknown faults are flagged < 5% (share of tiny numbers) |

H6 and H7 are the new evidence the toy could not give: they use real, different faults rather
than constructs. A result that is NOT SUPPORTED is still informative — e.g. if the target's
channels are so tightly coupled physically that no real fault moves only some of them, the
"key assumption" on slide 9 fails for that dataset, which is worth reporting.

## Caveats
* Simulated data (EnergyPlus/Modelica, HVACSIM+) has little sensor noise, so detection is
  easier than in the field; the mechanism, not the size, is what transfers (slide 10 #8).
* LBNL CSVs may carry a per-row ground-truth column; it is excluded from inputs. Windows only
  count as faulty when the signature is visibly present, so seasonal faults are handled.
* The RTU set has 7 fault-free cases; each fault file is paired with the most similar one by
  file name and timestamp overlap — check the pairing in `inspect/dataset_summary.json`.
* NIST rows are resampled with replacement to reach set sizes; small fault classes mean the
  sweep beyond the number of distinct rows adds little.
