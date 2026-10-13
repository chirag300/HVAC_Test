# Shortcut-learning test: toy (slide 7 replica)

Signature (k=3): ch0 (-), ch1 (-), ch2 (-)
Train windows: 4000 (50/50); thresholds at 1% false alarms on held-out normal; nonsig mode: normal

## Hypothesis checks

| Check | Verdict | Evidence |
|---|---|---|
| H1 one signature channel nearly separates the training data | **SUPPORTED** | best single-channel CV acc 0.916 vs all channels 0.982 |
| H2 LSTM flags partial-signature faults (P2) | **SUPPORTED** | LSTM P2 flagged 0.58; true AND rule 0.25; LSTM CAP recall 0.96 |
| H2b every model family is fooled the same way | **SUPPORTED** | 5/5 models flag >=30% of P2: LogReg (stats), Random forest (stats), Grad. boosting (stats), MLP (raw series), LSTM (raw series) |
| H3 the true rule scores lower on the training task | **SUPPORTED** | AND rule CAP recall 0.93 vs learned mean 0.96 |
| H4 more data / rebalancing does not remove partial false alarms (LSTM) | **SUPPORTED** | LSTM P2 false-alarm rate across sweep: min 0.36, max 0.76 |
| H5 targeted all-but-one negatives fix it (LSTM) | **NOT SUPPORTED** | P2: 0.58 -> 0.15; CAP recall change -0.12 |
| H6 real unknown faults that partially overlap the signature are flagged | **SUPPORTED** | LSTM flags 0.45 of real other-fault windows that move 1..2 signature channels vs 0.00 for those moving none (n=2000); overlap 1: 0.39 (n=2000), overlap 2: 0.57 (n=1000) |
| H7 most fleet false alarms come from unknown faults (LSTM) | **SUPPORTED** | 63% of false alarms in an 89/1/10 fleet mix |

## Flag rates (fraction flagged as the target fault)

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.02 | 0.98 | 0.24 | 0.77 | 1.00 | 0.49 | 0.18 | 0.00 | 0.02 |
| Random forest (stats) | 0.02 | 0.96 | 0.13 | 0.71 | 0.22 | 0.47 | 0.21 | 0.00 | 0.02 |
| Grad. boosting (stats) | 0.02 | 0.98 | 0.16 | 0.81 | 0.33 | 0.58 | 0.27 | 0.00 | 0.02 |
| MLP (raw series) | 0.02 | 0.93 | 0.14 | 0.58 | 0.81 | 0.15 | 0.07 | 0.00 | 0.01 |
| LSTM (raw series) | 0.01 | 0.96 | 0.13 | 0.58 | 0.65 | 0.15 | 0.06 | 0.00 | 0.01 |
| AND rule (true) | 0.02 | 0.93 | 0.06 | 0.25 | 0.07 | 0.12 | 0.07 | 0.00 | 0.02 |

## Which single channel carries the rule (only one signature channel deviates)

| Model | ONLY:ch0 | ONLY:ch1 | ONLY:ch2 |
|---|---:|---:|---:|
| LogReg (stats) | 0.26 | 0.28 | 0.24 |
| Random forest (stats) | 0.15 | 0.14 | 0.16 |
| Grad. boosting (stats) | 0.19 | 0.16 | 0.17 |
| MLP (raw series) | 0.15 | 0.14 | 0.14 |
| LSTM (raw series) | 0.16 | 0.12 | 0.16 |
| AND rule (true) | 0.07 | 0.08 | 0.08 |

GB importance by statistic (summed over signature channels): mean=0.088, min=0.092, max=0.000, std=0.191, slope=0.586, last8=0.003

LogReg |weight| share on signature channels: ch0=0.11, ch1=0.13, ch2=0.11

## Identifiability: CV accuracy with channel subsets

| Channels used | CV acc |
|---|---:|
| ch0 only | 0.906 |
| ch1 only | 0.916 |
| ch2 only | 0.909 |
| all sig but ch0 | 0.953 |
| all sig but ch1 | 0.951 |
| all sig but ch2 | 0.950 |
| all signature channels | 0.966 |
| all 20 channels | 0.982 |

## Real unknown faults (never seen in training)

| Fault file | group | n | mean overlap | full match | LogReg (stats) | Random forest (stats) | Grad. boosting (stats) | MLP (raw series) | LSTM (raw series) | AND rule (true) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toy:ASYNC | other | 1000 | 3.00 | 1.00 | 0.47 | 0.47 | 0.59 | 0.13 | 0.14 | 0.13 |
| toy:ASYNC2 | other | 1000 | 3.00 | 1.00 | 0.19 | 0.21 | 0.28 | 0.05 | 0.06 | 0.08 |
| toy:P2 | other | 1000 | 2.00 | 0.00 | 0.77 | 0.71 | 0.80 | 0.56 | 0.57 | 0.26 |
| toy:P1 | other | 1000 | 1.00 | 0.00 | 0.26 | 0.15 | 0.19 | 0.15 | 0.14 | 0.07 |
| toy:BIG1 | other | 1000 | 1.00 | 0.00 | 1.00 | 0.23 | 0.32 | 0.80 | 0.64 | 0.08 |
| toy:UP | other | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| toy:OFF | other | 1000 | 0.00 | 0.00 | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.02 |

Flag rate of other faults by number of signature channels they move in the fault direction:

| Model | overlap 0 | overlap 1 | overlap 2 | overlap 3 |
|---|---:|---:|---:|---:|
| LogReg (stats) | 0.01 (n=2000) | 0.63 (n=2000) | 0.77 (n=1000) | 0.33 (n=2000) |
| Random forest (stats) | 0.01 (n=2000) | 0.19 (n=2000) | 0.71 (n=1000) | 0.34 (n=2000) |
| Grad. boosting (stats) | 0.01 (n=2000) | 0.25 (n=2000) | 0.80 (n=1000) | 0.43 (n=2000) |
| MLP (raw series) | 0.00 (n=2000) | 0.48 (n=2000) | 0.56 (n=1000) | 0.09 (n=2000) |
| LSTM (raw series) | 0.00 (n=2000) | 0.39 (n=2000) | 0.57 (n=1000) | 0.10 (n=2000) |
| AND rule (true) | 0.01 (n=2000) | 0.08 (n=2000) | 0.26 (n=1000) | 0.10 (n=2000) |

Share of fleet false alarms caused by unknown faults (89% normal / 1% target / 10% other):

- LogReg (stats): 73%
- Random forest (stats): 62%
- Grad. boosting (stats): 67%
- MLP (raw series): 64%
- LSTM (raw series): 63%
- AND rule (true): 30%

## More data / balancing (partial-signature false-alarm rate)

| Train size | Prevalence | LogReg (stats) | Grad. boosting (stats) | MLP (raw series) | LSTM (raw series) |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.1 | 0.68 | 0.54 | 0.35 | 0.49 |
| 500 | 0.5 | 0.63 | 0.36 | 0.42 | 0.42 |
| 500 | 0.9 | 0.69 | 0.17 | 0.35 | 0.36 |
| 2000 | 0.1 | 0.75 | 0.76 | 0.53 | 0.58 |
| 2000 | 0.5 | 0.69 | 0.67 | 0.52 | 0.50 |
| 2000 | 0.9 | 0.67 | 0.18 | 0.55 | 0.46 |
| 8000 | 0.1 | 0.76 | 0.78 | 0.57 | 0.55 |
| 8000 | 0.5 | 0.78 | 0.70 | 0.63 | 0.69 |
| 8000 | 0.9 | 0.76 | 0.34 | 0.60 | 0.47 |
| 20000 | 0.1 | 0.86 | 0.85 | 0.60 | 0.67 |
| 20000 | 0.5 | 0.82 | 0.79 | 0.61 | 0.76 |
| 20000 | 0.9 | 0.80 | 0.56 | 0.64 | 0.68 |

## Targeted all-but-one negatives added to training

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.02 | 0.96 | 0.16 | 0.64 | 0.88 | 0.18 | 0.09 | 0.00 | 0.01 |
| Random forest (stats) | 0.01 | 0.87 | 0.03 | 0.17 | 0.04 | 0.22 | 0.09 | 0.00 | 0.01 |
| Grad. boosting (stats) | 0.01 | 0.92 | 0.03 | 0.21 | 0.02 | 0.37 | 0.17 | 0.00 | 0.01 |
| MLP (raw series) | 0.01 | 0.87 | 0.06 | 0.28 | 0.02 | 0.10 | 0.05 | 0.00 | 0.01 |
| LSTM (raw series) | 0.01 | 0.84 | 0.03 | 0.15 | 0.00 | 0.03 | 0.02 | 0.00 | 0.01 |

Other-fault flag rate by overlap after the fix:

- LogReg (stats): ov0=0.01, ov1=0.52, ov2=0.62, ov3=0.14
- Random forest (stats): ov0=0.01, ov1=0.03, ov2=0.15, ov3=0.15
- Grad. boosting (stats): ov0=0.01, ov1=0.03, ov2=0.20, ov3=0.27
- MLP (raw series): ov0=0.00, ov1=0.03, ov2=0.25, ov3=0.08
- LSTM (raw series): ov0=0.00, ov1=0.02, ov2=0.14, ov3=0.03
