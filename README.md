# Sum Instead of AND — CS 886 course project

Code, reference results and the paper for the final project of **CS 886: Learning Theory for Modern AI**
(University of Waterloo, Fall 2026):

> Chirag Seth. *Sum Instead of AND: Why Fault Detectors Flag Look-Alikes, and How Near-Misses Teach
> Attention the Conjunction.* CS 886 final project report, 2026.
> [`paper/Seth_CS886_report_source.pdf`](paper/Seth_CS886_report_source.pdf)

A fault detector is usually trained to separate normal operation from one target fault (Fault 1), but
in service it also meets *look-alikes*: other conditions that shift only part of Fault 1's sensors. The
report shows that the most powerful detector for the training problem adds the sensors' evidence (a SUM),
while a detector that must stay quiet on every look-alike has to require all of them (an AND), and no
weighted sum can replace it. For softmax attention pooling over sensor tokens, the inverse temperature
β moves the pooled score from an OR through the SUM to the AND. *Near-misses* (look-alikes that miss
exactly one sensor) push β toward the AND, and under noise training settles on a soft AND. Transformer
detectors trained on synthetic data and on the LBNL fan-coil benchmark follow these predictions.

This repository reproduces every number, table and figure of the report. Everything is written in
NumPy (models, backpropagation and Adam included), so it runs on a plain CPU without a deep-learning
framework.

## Repository layout

```
.
├── README.md                   this file
├── requirements.txt            numpy, pandas, scikit-learn, matplotlib, scipy
├── reproduce.sh                runs every step below in order
│
├── transformer_numpy.py        the detectors: sensor-token transformer and attention pool
│                               (manual backprop + Adam); `python transformer_numpy.py` runs gradient checks
├── run_transformer.py          the experiments of Section 4 and Appendix F (battery, soft-AND sweep,
│                               look-alike strength sweep); writes JSON, REPORT_transformer.md and figures
├── shortcut_lib.py             shared library: window construction X = W + G*D for every look-alike type,
│                               the AND-rule oracle, detector registry, calibration to 1% false alarms
├── synthetic_data.py           the synthetic source (20 sensors, 48 steps, AR(1) noise, Fault 1 = 3 sensors)
├── real_data.py                LBNL loader: pairs each faulty run with the fault-free run, finds the
│                               signature, builds windows by counterfactual splicing
├── ci_numpy.py                 additional per-sensor baseline models trained alongside the transformers
├── download.py                 fetches the LBNL data into data/raw/
├── verify_all.py               numerical checks of every closed-form result (Theorems 1-4, Lemmas 1-2)
├── theory/
│   └── attention_checks.py     checks of Theorems 3-4 through the implicit-function slope
│
├── reference_results/
│   ├── transformer_toy/        synthetic data, 3 seeds: REPORT_transformer.md, battery/softand/strength
│   │                           JSON, fig_law.png, fig_softand.png, fig_strength.png
│   └── transformer_fcu/        LBNL fan-coil unit, Fault 1 = FCU_FanOutletBlockage, 3 seeds (same files)
│
└── paper/
    ├── Seth_CS886_report_source.pdf            the report (ICML 2026 template, 3 pages + appendix)
    ├── Seth_CS886_report.pdf                   an earlier build of the report
    ├── Seth_CS886_near_miss_propositions.pdf   companion note: which near-misses, and how many
    ├── Seth_CS886_slides.pdf                   presentation slides
    └── latex/                                  LaTeX sources of the report, the note and the slides
```

`data/` and `results/` are created when the scripts run and are not tracked.

## Setup

Python 3.9 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python transformer_numpy.py        # gradient checks of the models -> PASS
```

## Reproducing the report

Run `bash reproduce.sh`, or the steps one at a time:

| Step | Command | Time (CPU) | Reproduces |
|---|---|---|---|
| 1 | `python verify_all.py` | ~6 min | numerical check of every closed-form result: Theorems 1–4, Lemmas 1–2, Remark 1 (53 checks, all pass) |
| 2 | `python theory/attention_checks.py` | ~5 min | Theorem 3(a–c) and the slope κ of Theorem 4 via Monte Carlo Hessians |
| 3 | `python run_transformer.py --out results/transformer_toy --seeds 0,1,2` | ~1.5 h | synthetic columns of Table 2; Tables 3–5 and Figure 4 (synthetic panel) in Appendix F |
| 4 | `python download.py fcu` | download | LBNL fan-coil data into `data/raw/fcu/` |
| 5 | `python run_transformer.py --data data/raw/fcu --out results/transformer_fcu --target FCU_FanOutletBlockage --seeds 0,1,2 --rhos 0,0.1,0.2,0.4 --snr-scales 0.5,1` | ~2 h | fan-coil columns of Table 2; Table 6 and Figure 4 (fan-coil panel) in Appendix F |

Times are for two CPU cores. A ten-minute sanity run of the whole experiment pipeline:

```bash
python run_transformer.py --out /tmp/quick --seeds 0 --quick
```

Every experiment cell is cached as JSON (`battery.json`, `softand.json`, `strength.json`), so an
interrupted run resumes where it stopped; `--steps ""` only rebuilds `REPORT_transformer.md` and the
figures from the caches. Compare your outputs with `reference_results/`.

### Where each result is in the output

`run_transformer.py` has three steps (`--steps battery,softand,strength`):

| Step | What it does | Report |
|---|---|---|
| `battery` | trains every model on each training set and measures alarm rates on every window type; adds the false-alarm law of Theorem 1(ii) and, for the attention pool, the learned inverse temperature β̂ and the left-out ratio | Table 2; Tables 3, 4 and 6 |
| `softand` | trains the attention pool with near-miss share ρ (`--rhos`) at fault scales `--snr-scales` | Figure 4 (`fig_softand.png`) |
| `strength` | tests the plainly trained models on look-alikes whose shift is multiplied by M (`--strengths`) | Table 5 (`fig_strength.png`) |

Window types and training sets as they are named in the output files:

| Output name | In the report |
|---|---|
| `NORMAL` | normal windows (thresholds are set for 1% false alarms on them) |
| `CAP` | Fault 1 (detection rate) |
| `P2` / `P1` | two-of-three / one-of-three look-alikes |
| `BIG1` | one loud sensor (shift ×3) |
| `AND rule (true)` | the AND-rule oracle |
| `plain` | training on normal and Fault 1 windows |
| `+NM` | plus the three channel near-misses (2,000 windows) |

The report uses the `Transformer (sensor tokens)` and `Attention pool (sensor tokens)` rows and the
`plain` and `+NM` training sets. The output also contains timing look-alikes (`ASYNC`, `PAIRSTAG`,
`CAPH`) and the `+NM+PS` and `+NM+PS+H` training sets, which are analysed in the companion note, and
additional baseline models that the report does not use.

## Data

The fan-coil experiments use the public LBNL fault dataset for a fan-coil unit (Granderson et al., 2023,
*A labeled dataset for building HVAC systems operating in faulted and fault-free states*, Scientific
Data 10:342): one year of simulated operation without faults and with each fault, under the same
building and weather. `python download.py fcu` fetches it. LBNL sometimes asks for a short registration
form; in that case the script prints the exact file and folder to use, and re-running it unpacks the
file. The runs are paired, so the effect of a fault is `ΔX = X_fault − X_normal`, and every window is
built by counterfactual splicing `X = W + M ⊙ ΔX`. A near-miss of type j uses Fault 1's mask with
sensor j removed. The synthetic source needs no data.

## Building the paper

```bash
cd paper/latex/report && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ../near_miss_propositions && pdflatex props && bibtex props && pdflatex props && pdflatex props
cd ../slides && pdflatex sum_instead_of_and_talk && pdflatex sum_instead_of_and_talk
```

## Citation

```bibtex
@misc{seth2026sumand,
  author       = {Seth, Chirag},
  title        = {Sum Instead of {AND}: Why Fault Detectors Flag Look-Alikes, and How Near-Misses
                  Teach Attention the Conjunction},
  howpublished = {CS 886 (Learning Theory for Modern AI) final project report, University of Waterloo},
  year         = {2026}
}
```
