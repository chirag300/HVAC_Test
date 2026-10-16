# Shortcut-learning test: toy (slide 7 replica)

Signature (k=3): ch0 (-), ch1 (-), ch2 (-)
Train windows: 4000 (50/50); thresholds at 1% false alarms on held-out normal; nonsig mode: normal

## Hypothesis checks

| Check | Verdict | Evidence |
|---|---|---|
| H1 one signature channel nearly separates the training data | **SUPPORTED** | best single-channel CV acc 0.916 vs all channels 0.982 |
| H2 LSTM flags partial-signature faults (P2) | **SUPPORTED** | LSTM P2 flagged 0.58; true AND rule 0.25; LSTM CAP recall 0.96 |
| H2b every model family is fooled the same way | **SUPPORTED** | 7/7 models flag >=30% of P2: LogReg (stats), Random forest (stats), Grad. boosting (stats), MLP (raw series), LSTM (raw series), Channel-indep. (raw series), Capped CI (raw series) |
| H3 the true rule scores lower on the training task | **SUPPORTED** | AND rule CAP recall 0.93 vs learned mean 0.95 |
| H5 targeted all-but-one negatives fix it (LSTM) | **NOT SUPPORTED** | P2: 0.58 -> 0.15; CAP recall change -0.12 |
| H5t staggered (time) negatives fix simultaneity (LSTM) | **SUPPORTED** | ASYNC/ASYNC2 mean 0.11 -> 0.03; P2 0.58 -> 0.44 (time negatives alone should not fix it); CAP change +0.00 |
| H5b channel + time negatives fix both (LSTM) | **NOT SUPPORTED** | mean of P2, ASYNC, ASYNC2: 0.27 -> 0.10; CAP change -0.08 |
| H8 additive (channel-independent) model stays fooled after the fix | **SUPPORTED** | after all-but-one negatives: channel-indep. P2 0.72 -> 0.35 vs LSTM 0.58 -> 0.15 |
| HA partial-match rate follows the additive law (LSTM) | **SUPPORTED** | P2 measured 0.58; predicted from CAP + single-channel rates 0.58; from CAP alone (equal channels) 0.64 |
| H5h pair-staggered negatives teach simultaneity, fully staggered do not (LSTM) | **NOT SUPPORTED** | PAIRSTAG flagged: before 0.30; + fully staggered negatives 0.17 (their own ASYNC/ASYNC2 0.03); + pair-staggered negatives 0.06 |
| H9 capped per-channel evidence can be fixed, uncapped cannot | **NOT SUPPORTED** | after all-but-one negatives P2: capped 0.37 -> 0.22 (CAP 0.90 -> 0.85); uncapped 0.72 -> 0.35; BIG1 without any fix: capped 0.08 vs uncapped 0.99 |
| HP stronger target fault -> more partial false alarms (LSTM) | **SUPPORTED** | scale x0.25: CAP 0.09 / P2 0.05 (law 0.05), x0.5: CAP 0.27 / P2 0.10 (law 0.12), x0.75: CAP 0.74 / P2 0.34 (law 0.37), x1: CAP 0.96 / P2 0.59 (law 0.66), x1.5: CAP 1.00 / P2 0.94 (law 0.89) |
| HR more near-miss negatives -> fewer partial false alarms (LSTM) | **SUPPORTED** | P2 by near-misses per type: 0:0.58, 3:0.51, 10:0.58, 30:0.47, 100:0.35, 300:0.18, 1000:0.16; halved (CAP kept) at >1000 per type |
| HR more near-miss negatives -> fewer partial false alarms (Channel-indep.) | **SUPPORTED** | P2 by near-misses per type: 0:0.73, 3:0.71, 10:0.72, 30:0.65, 100:0.59, 300:0.42, 1000:0.32; halved (CAP kept) at 1000 per type |
| HR more near-miss negatives -> fewer partial false alarms (Capped CI) | **NOT SUPPORTED** | P2 by near-misses per type: 0:0.48, 3:0.44, 10:0.40, 30:0.42, 100:0.32, 300:0.27, 1000:0.26; halved (CAP kept) at >1000 per type |
| HR more near-miss negatives -> fewer partial false alarms (LSTM wd=0.001) | **SUPPORTED** | P2 by near-misses per type: 0:0.53, 3:0.48, 10:0.49, 30:0.38, 100:0.31, 300:0.18, 1000:0.14; halved (CAP kept) at >1000 per type |
| H6 real unknown faults with partial overlap are flagged (LogReg) | **SUPPORTED** | partial overlap 0.68 vs none 0.01 (n=2000); ov1 0.64 (n=2000), ov2 0.78 (n=1000) |
| H7 most fleet false alarms come from unknown faults (LogReg) | **SUPPORTED** | 73% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.39 on average |
| H6 real unknown faults with partial overlap are flagged (Random forest) | **SUPPORTED** | partial overlap 0.37 vs none 0.01 (n=2000); ov1 0.20 (n=2000), ov2 0.71 (n=1000) |
| H7 most fleet false alarms come from unknown faults (Random forest) | **SUPPORTED** | 62% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.26 on average |
| H6 real unknown faults with partial overlap are flagged (Grad. boosting) | **SUPPORTED** | partial overlap 0.45 vs none 0.01 (n=2000); ov1 0.27 (n=2000), ov2 0.80 (n=1000) |
| H7 most fleet false alarms come from unknown faults (Grad. boosting) | **SUPPORTED** | 67% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.32 on average |
| H6 real unknown faults with partial overlap are flagged (MLP) | **SUPPORTED** | partial overlap 0.52 vs none 0.00 (n=2000); ov1 0.49 (n=2000), ov2 0.57 (n=1000) |
| H7 most fleet false alarms come from unknown faults (MLP) | **SUPPORTED** | 65% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.25 on average |
| H6 real unknown faults with partial overlap are flagged (LSTM) | **SUPPORTED** | partial overlap 0.47 vs none 0.00 (n=2000); ov1 0.40 (n=2000), ov2 0.60 (n=1000) |
| H7 most fleet false alarms come from unknown faults (LSTM) | **SUPPORTED** | 63% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.23 on average |
| H6 real unknown faults with partial overlap are flagged (Channel-indep.) | **SUPPORTED** | partial overlap 0.63 vs none 0.00 (n=2000); ov1 0.59 (n=2000), ov2 0.71 (n=1000) |
| H7 most fleet false alarms come from unknown faults (Channel-indep.) | **SUPPORTED** | 76% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.31 on average |
| H6 real unknown faults with partial overlap are flagged (Capped CI) | **NOT SUPPORTED** | partial overlap 0.18 vs none 0.00 (n=2000); ov1 0.09 (n=2000), ov2 0.35 (n=1000) |
| H7 most fleet false alarms come from unknown faults (Capped CI) | **NOT SUPPORTED** | 43% of false alarms in an 89/1/10 fleet mix; unknown faults flagged 0.10 on average |

## Flag rates (fraction flagged as the target fault)

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | PAIRSTAG | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.02 | 0.98 | 0.24 | 0.77 | 1.00 | 0.49 | 0.18 | 0.73 | 0.00 | 0.02 |
| Random forest (stats) | 0.02 | 0.96 | 0.13 | 0.71 | 0.22 | 0.47 | 0.21 | 0.66 | 0.00 | 0.02 |
| Grad. boosting (stats) | 0.02 | 0.98 | 0.16 | 0.81 | 0.33 | 0.58 | 0.27 | 0.79 | 0.00 | 0.02 |
| MLP (raw series) | 0.02 | 0.93 | 0.14 | 0.58 | 0.81 | 0.15 | 0.07 | 0.38 | 0.00 | 0.01 |
| LSTM (raw series) | 0.01 | 0.96 | 0.13 | 0.58 | 0.65 | 0.15 | 0.06 | 0.30 | 0.00 | 0.01 |
| Channel-indep. (raw series) | 0.01 | 0.96 | 0.20 | 0.72 | 0.99 | 0.19 | 0.06 | 0.47 | 0.00 | 0.00 |
| Capped CI (raw series) | 0.01 | 0.90 | 0.06 | 0.37 | 0.08 | 0.10 | 0.04 | 0.23 | 0.00 | 0.02 |
| AND rule (true) | 0.02 | 0.93 | 0.06 | 0.25 | 0.07 | 0.12 | 0.07 | 0.40 | 0.00 | 0.02 |

## Which single channel carries the rule (only one signature channel deviates)

| Model | ONLY:ch0 | ONLY:ch1 | ONLY:ch2 |
|---|---:|---:|---:|
| LogReg (stats) | 0.25 | 0.27 | 0.23 |
| Random forest (stats) | 0.14 | 0.15 | 0.15 |
| Grad. boosting (stats) | 0.18 | 0.17 | 0.18 |
| MLP (raw series) | 0.15 | 0.15 | 0.15 |
| LSTM (raw series) | 0.14 | 0.13 | 0.16 |
| Channel-indep. (raw series) | 0.22 | 0.20 | 0.19 |
| Capped CI (raw series) | 0.06 | 0.07 | 0.07 |
| AND rule (true) | 0.08 | 0.08 | 0.07 |

GB importance by statistic (summed over signature channels): mean=0.088, min=0.092, max=0.000, std=0.191, slope=0.586, last8=0.003

LogReg |weight| share on signature channels: ch0=0.11, ch1=0.13, ch2=0.11

## Theory check: partial-match rates predicted from the detection rate

Law (theory/theory.pdf, Cor. 'power-specificity curve'): P_m = Phi(r (Phi^-1(CAP) + z) - z), z = Phi^-1(0.99). 'CAP only' assumes equal channels, r = m/k; 'CAP+singles' reads each channel's share r_j from its single-channel flag rate and averages over channel subsets. Close agreement = the model behaves like an additive (sum-of-evidence) detector.

| Model | CAP | P1 | P1 pred (CAP only) | P1 pred (CAP+singles) | P2 | P2 pred (CAP only) | P2 pred (CAP+singles) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.98 | 0.24 | 0.19 | 0.25 | 0.77 | 0.71 | 0.84 |
| Random forest (stats) | 0.96 | 0.13 | 0.17 | 0.14 | 0.71 | 0.66 | 0.58 |
| Grad. boosting (stats) | 0.98 | 0.16 | 0.19 | 0.18 | 0.81 | 0.71 | 0.68 |
| MLP (raw series) | 0.93 | 0.14 | 0.15 | 0.15 | 0.58 | 0.58 | 0.59 |
| LSTM (raw series) | 0.96 | 0.13 | 0.16 | 0.14 | 0.58 | 0.64 | 0.58 |
| Channel-indep. (raw series) | 0.96 | 0.20 | 0.17 | 0.20 | 0.72 | 0.65 | 0.75 |
| Capped CI (raw series) | 0.90 | 0.06 | 0.13 | 0.07 | 0.37 | 0.53 | 0.26 |
| AND rule (true) | 0.93 | 0.06 | 0.14 | 0.08 | 0.25 | 0.57 | 0.29 |

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

| Fault file | group | n | mean overlap | full match | LogReg (stats) | Random forest (stats) | Grad. boosting (stats) | MLP (raw series) | LSTM (raw series) | Channel-indep. (raw series) | Capped CI (raw series) | AND rule (true) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toy:ASYNC | other | 1000 | 3.00 | 1.00 | 0.48 | 0.47 | 0.60 | 0.15 | 0.15 | 0.21 | 0.10 | 0.13 |
| toy:ASYNC2 | other | 1000 | 3.00 | 1.00 | 0.17 | 0.19 | 0.26 | 0.06 | 0.05 | 0.07 | 0.03 | 0.07 |
| toy:P2 | other | 1000 | 2.00 | 0.00 | 0.78 | 0.71 | 0.80 | 0.57 | 0.60 | 0.71 | 0.35 | 0.27 |
| toy:P1 | other | 1000 | 1.00 | 0.00 | 0.27 | 0.14 | 0.18 | 0.15 | 0.16 | 0.20 | 0.07 | 0.07 |
| toy:BIG1 | other | 1000 | 1.00 | 0.00 | 1.00 | 0.26 | 0.36 | 0.83 | 0.65 | 0.99 | 0.10 | 0.09 |
| toy:UP | other | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| toy:OFF | other | 1000 | 0.00 | 0.00 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.00 | 0.01 | 0.02 |

Flag rate of other faults by number of signature channels they move in the fault direction:

| Model | overlap 0 | overlap 1 | overlap 2 | overlap 3 |
|---|---:|---:|---:|---:|
| LogReg (stats) | 0.01 (n=2000) | 0.64 (n=2000) | 0.78 (n=1000) | 0.32 (n=2000) |
| Random forest (stats) | 0.01 (n=2000) | 0.20 (n=2000) | 0.71 (n=1000) | 0.33 (n=2000) |
| Grad. boosting (stats) | 0.01 (n=2000) | 0.27 (n=2000) | 0.80 (n=1000) | 0.43 (n=2000) |
| MLP (raw series) | 0.00 (n=2000) | 0.49 (n=2000) | 0.57 (n=1000) | 0.10 (n=2000) |
| LSTM (raw series) | 0.00 (n=2000) | 0.40 (n=2000) | 0.60 (n=1000) | 0.10 (n=2000) |
| Channel-indep. (raw series) | 0.00 (n=2000) | 0.59 (n=2000) | 0.71 (n=1000) | 0.14 (n=2000) |
| Capped CI (raw series) | 0.00 (n=2000) | 0.09 (n=2000) | 0.35 (n=1000) | 0.07 (n=2000) |
| AND rule (true) | 0.01 (n=2000) | 0.08 (n=2000) | 0.27 (n=1000) | 0.10 (n=2000) |

Share of fleet false alarms caused by unknown faults (89% normal / 1% target / 10% other):

- LogReg (stats): 73%
- Random forest (stats): 62%
- Grad. boosting (stats): 67%
- MLP (raw series): 65%
- LSTM (raw series): 63%
- Channel-indep. (raw series): 76%
- Capped CI (raw series): 43%
- AND rule (true): 30%

## Targeted all-but-one negatives added to training

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | PAIRSTAG | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.02 | 0.96 | 0.16 | 0.64 | 0.88 | 0.18 | 0.09 | 0.44 | 0.00 | 0.01 |
| Random forest (stats) | 0.01 | 0.87 | 0.03 | 0.17 | 0.04 | 0.22 | 0.09 | 0.39 | 0.00 | 0.01 |
| Grad. boosting (stats) | 0.01 | 0.92 | 0.03 | 0.21 | 0.02 | 0.37 | 0.17 | 0.59 | 0.00 | 0.01 |
| MLP (raw series) | 0.01 | 0.87 | 0.06 | 0.28 | 0.02 | 0.10 | 0.05 | 0.30 | 0.00 | 0.00 |
| LSTM (raw series) | 0.01 | 0.84 | 0.03 | 0.15 | 0.00 | 0.03 | 0.02 | 0.08 | 0.00 | 0.01 |
| Channel-indep. (raw series) | 0.01 | 0.93 | 0.08 | 0.35 | 0.31 | 0.09 | 0.03 | 0.26 | 0.00 | 0.01 |
| Capped CI (raw series) | 0.01 | 0.85 | 0.04 | 0.22 | 0.05 | 0.08 | 0.03 | 0.21 | 0.00 | 0.00 |

Other-fault flag rate by overlap after the fix:

- LogReg (stats): ov0=0.01, ov1=0.52, ov2=0.62, ov3=0.14
- Random forest (stats): ov0=0.01, ov1=0.03, ov2=0.15, ov3=0.15
- Grad. boosting (stats): ov0=0.01, ov1=0.03, ov2=0.20, ov3=0.27
- MLP (raw series): ov0=0.00, ov1=0.03, ov2=0.25, ov3=0.08
- LSTM (raw series): ov0=0.00, ov1=0.02, ov2=0.14, ov3=0.03
- Channel-indep. (raw series): ov0=0.00, ov1=0.19, ov2=0.34, ov3=0.07
- Capped CI (raw series): ov0=0.00, ov1=0.04, ov2=0.19, ov3=0.06

## Targeted fully staggered (time) negatives added to training

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | PAIRSTAG | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.01 | 0.94 | 0.17 | 0.62 | 0.95 | 0.03 | 0.01 | 0.15 | 0.00 | 0.01 |
| Random forest (stats) | 0.01 | 0.89 | 0.06 | 0.44 | 0.10 | 0.08 | 0.03 | 0.17 | 0.00 | 0.01 |
| Grad. boosting (stats) | 0.01 | 0.92 | 0.07 | 0.47 | 0.11 | 0.06 | 0.02 | 0.15 | 0.00 | 0.01 |
| MLP (raw series) | 0.01 | 0.90 | 0.10 | 0.49 | 0.68 | 0.08 | 0.05 | 0.27 | 0.00 | 0.00 |
| LSTM (raw series) | 0.01 | 0.96 | 0.06 | 0.44 | 0.13 | 0.03 | 0.02 | 0.17 | 0.00 | 0.01 |
| Channel-indep. (raw series) | 0.01 | 0.94 | 0.16 | 0.63 | 0.93 | 0.03 | 0.02 | 0.12 | 0.00 | 0.01 |
| Capped CI (raw series) | 0.01 | 0.90 | 0.08 | 0.50 | 0.15 | 0.02 | 0.02 | 0.08 | 0.00 | 0.01 |

Other-fault flag rate by overlap:

- LogReg (stats): ov0=0.01, ov1=0.55, ov2=0.62, ov3=0.02
- Random forest (stats): ov0=0.01, ov1=0.08, ov2=0.44, ov3=0.06
- Grad. boosting (stats): ov0=0.00, ov1=0.09, ov2=0.48, ov3=0.04
- MLP (raw series): ov0=0.00, ov1=0.39, ov2=0.48, ov3=0.07
- LSTM (raw series): ov0=0.01, ov1=0.09, ov2=0.45, ov3=0.03
- Channel-indep. (raw series): ov0=0.00, ov1=0.55, ov2=0.63, ov3=0.02
- Capped CI (raw series): ov0=0.00, ov1=0.12, ov2=0.49, ov3=0.02

## Targeted all-but-one + fully staggered negatives added to training

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | PAIRSTAG | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.02 | 0.95 | 0.15 | 0.62 | 0.86 | 0.09 | 0.04 | 0.28 | 0.00 | 0.01 |
| Random forest (stats) | 0.01 | 0.88 | 0.03 | 0.19 | 0.05 | 0.11 | 0.05 | 0.21 | 0.00 | 0.01 |
| Grad. boosting (stats) | 0.01 | 0.91 | 0.04 | 0.20 | 0.04 | 0.09 | 0.04 | 0.20 | 0.00 | 0.01 |
| MLP (raw series) | 0.01 | 0.89 | 0.08 | 0.36 | 0.11 | 0.10 | 0.05 | 0.29 | 0.00 | 0.00 |
| LSTM (raw series) | 0.01 | 0.88 | 0.05 | 0.24 | 0.10 | 0.04 | 0.02 | 0.13 | 0.00 | 0.01 |
| Channel-indep. (raw series) | 0.01 | 0.91 | 0.07 | 0.34 | 0.31 | 0.05 | 0.03 | 0.18 | 0.00 | 0.01 |
| Capped CI (raw series) | 0.01 | 0.88 | 0.05 | 0.27 | 0.06 | 0.07 | 0.03 | 0.21 | 0.00 | 0.01 |

Other-fault flag rate by overlap:

- LogReg (stats): ov0=0.01, ov1=0.50, ov2=0.61, ov3=0.06
- Random forest (stats): ov0=0.00, ov1=0.04, ov2=0.20, ov3=0.08
- Grad. boosting (stats): ov0=0.01, ov1=0.03, ov2=0.19, ov3=0.06
- MLP (raw series): ov0=0.00, ov1=0.09, ov2=0.36, ov3=0.06
- LSTM (raw series): ov0=0.00, ov1=0.07, ov2=0.24, ov3=0.03
- Channel-indep. (raw series): ov0=0.01, ov1=0.19, ov2=0.32, ov3=0.04
- Capped CI (raw series): ov0=0.01, ov1=0.04, ov2=0.29, ov3=0.05

## Targeted all-but-one + pair-staggered (Helly) negatives added to training

| Model | NORMAL | CAP | P1 | P2 | BIG1 | ASYNC | ASYNC2 | PAIRSTAG | UP | OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (stats) | 0.02 | 0.94 | 0.15 | 0.57 | 0.75 | 0.06 | 0.03 | 0.10 | 0.00 | 0.01 |
| Random forest (stats) | 0.01 | 0.87 | 0.03 | 0.20 | 0.05 | 0.09 | 0.05 | 0.12 | 0.00 | 0.01 |
| Grad. boosting (stats) | 0.01 | 0.90 | 0.03 | 0.20 | 0.04 | 0.09 | 0.04 | 0.10 | 0.00 | 0.01 |
| MLP (raw series) | 0.01 | 0.88 | 0.07 | 0.34 | 0.08 | 0.10 | 0.05 | 0.11 | 0.00 | 0.00 |
| LSTM (raw series) | 0.02 | 0.90 | 0.06 | 0.26 | 0.06 | 0.04 | 0.03 | 0.06 | 0.00 | 0.02 |
| Channel-indep. (raw series) | 0.01 | 0.93 | 0.10 | 0.40 | 0.37 | 0.06 | 0.03 | 0.05 | 0.00 | 0.01 |
| Capped CI (raw series) | 0.01 | 0.88 | 0.06 | 0.32 | 0.08 | 0.06 | 0.03 | 0.06 | 0.00 | 0.01 |

Other-fault flag rate by overlap:

- LogReg (stats): ov0=0.01, ov1=0.46, ov2=0.59, ov3=0.04
- Random forest (stats): ov0=0.01, ov1=0.04, ov2=0.17, ov3=0.08
- Grad. boosting (stats): ov0=0.01, ov1=0.04, ov2=0.19, ov3=0.07
- MLP (raw series): ov0=0.00, ov1=0.06, ov2=0.32, ov3=0.08
- LSTM (raw series): ov0=0.01, ov1=0.06, ov2=0.26, ov3=0.04
- Channel-indep. (raw series): ov0=0.00, ov1=0.23, ov2=0.39, ov3=0.05
- Capped CI (raw series): ov0=0.00, ov1=0.06, ov2=0.31, ov3=0.05

## Recipe curve: near-miss (all-but-one) negatives per type

Same normal-vs-fault training set; only the number of near-miss negatives changes. Cells: P2 flagged / CAP flagged.

| Near-misses per type | MLP (raw series) | LSTM (raw series) | Channel-indep. (raw series) | Capped CI (raw series) | LSTM wd=0.001 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.60 / 0.94 | 0.58 / 0.95 | 0.73 / 0.97 | 0.48 / 0.92 | 0.53 / 0.93 |
| 3 | 0.55 / 0.92 | 0.51 / 0.93 | 0.71 / 0.97 | 0.44 / 0.90 | 0.48 / 0.92 |
| 10 | 0.54 / 0.92 | 0.58 / 0.95 | 0.72 / 0.97 | 0.40 / 0.91 | 0.49 / 0.92 |
| 30 | 0.49 / 0.91 | 0.47 / 0.94 | 0.65 / 0.96 | 0.42 / 0.89 | 0.38 / 0.92 |
| 100 | 0.48 / 0.92 | 0.35 / 0.91 | 0.59 / 0.97 | 0.32 / 0.88 | 0.31 / 0.91 |
| 300 | 0.33 / 0.89 | 0.18 / 0.85 | 0.42 / 0.94 | 0.27 / 0.86 | 0.18 / 0.84 |
| 1000 | 0.19 / 0.85 | 0.16 / 0.86 | 0.32 / 0.92 | 0.26 / 0.88 | 0.14 / 0.82 |

## Signal-strength sweep (fault deviation x s, train and test)

Cells: CAP / P2 / P2 predicted from CAP by the law (equal channels).

| Scale | LogReg (stats) | MLP (raw series) | LSTM (raw series) | Channel-indep. (raw series) | Capped CI (raw series) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.11 / 0.06 / 0.06 | 0.09 / 0.05 / 0.05 | 0.09 / 0.05 / 0.05 | 0.09 / 0.06 / 0.05 | 0.06 / 0.04 / 0.03 |
| 0.5 | 0.43 / 0.18 / 0.18 | 0.39 / 0.16 / 0.17 | 0.27 / 0.10 / 0.12 | 0.42 / 0.18 / 0.18 | 0.28 / 0.11 / 0.12 |
| 0.75 | 0.83 / 0.47 / 0.44 | 0.73 / 0.33 / 0.36 | 0.74 / 0.34 / 0.37 | 0.81 / 0.41 / 0.42 | 0.69 / 0.26 / 0.33 |
| 1 | 0.98 / 0.76 / 0.71 | 0.94 / 0.56 / 0.61 | 0.96 / 0.59 / 0.66 | 0.97 / 0.72 / 0.69 | 0.93 / 0.48 / 0.58 |
| 1.5 | 1.00 / 1.00 / 0.96 | 1.00 / 0.86 / 0.90 | 1.00 / 0.94 / 0.89 | 1.00 / 0.97 / 0.92 | 1.00 / 0.92 / 0.89 |
