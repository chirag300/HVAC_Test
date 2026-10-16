# Theory: why detectors learn the broad rule, and what fixes it

| File | What it is |
|---|---|
| `theory.pdf` / `theory.tex` | Statements and complete proofs (Sections 3–7), the recipe (Section 8), numerical checks (Section 9), how this revises the earlier ICLR draft's Lemmas 1–4 (Section 10), novelty vs prior work (Section 11), open problems (Section 12) |
| `theory_checks.py` | Numerical checks C1–C9, one per theorem. `python theory/theory_checks.py` runs all (~15 min CPU), `python theory/theory_checks.py C3 C6` runs a subset; results merge into `checks_out/results.json`, log in `checks_out/log.txt` |
| `make_figures.py` | Rebuilds `figs/` from `checks_out/results.json` |

Rebuild the PDF:
```bash
python theory/make_figures.py
cd theory && latexmk -pdf theory.tex
```

## Map from theorem to check and to the pipeline experiment

| Result | Numerical check | Pipeline step / verdict on real data |
|---|---|---|
| Thm 3.1 training-optimal detector is additive, n- and prevalence-invariant | C1 | `sweep` / H4 |
| Thm 3.3, Cor 3.4 partial-overlap law, detectability paradox | C1, C2, C8 | "Theory check" table / HA; `scale` / HP |
| Thm 4.1, Prop 4.2 composite null, capped evidence | C3 | `Capped CI` model / H9, H8 |
| Thm 5.2–5.4 near-miss position, k× norm cost | C4 | — |
| Prop 5.6 necessary near-miss fraction p*(λ) | C9, C5 | `nmsweep` (+ `--nm-wd`) / HR |
| Lemma 6.1, 6.2 window statistics are timing- and sign-blind | C6a | ASYNC / PAIRSTAG / UP columns |
| Lemma 6.3 AR(1) effective noise | C7 | — |
| Thm 7.1, 7.2 teaching set: k channel + C(k,2) pair-staggered near-misses | C6b | `fixhelly` vs `fixtime` / H5h |
| Prop 7.4 examples per near-miss type | — | `nmsweep` |
