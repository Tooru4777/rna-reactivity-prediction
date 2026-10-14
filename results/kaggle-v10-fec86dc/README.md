# Kaggle Version 10 — validated P100 run

This directory is the compact evidence package for the reportable RNA
reactivity experiment. Files were downloaded from Kaggle kernel Version 10
after it reached `COMPLETE`. CSV and JSON values plus the exact-length histogram
are archived without modification. During review, `training_curves.png` and
`error_by_length.png` were reproducibly re-rendered from the archived CSVs to
show cross-seed standard-deviation bands/bars and sample counts. `README.md` and
`artifact_checksums.sha256` are repository-level metadata added during review.

## Provenance

- Run date: 2026-08-11
- Kernel: [`tooru4777/rna-reactivity-grouped-ablation`](https://www.kaggle.com/code/tooru4777/rna-reactivity-grouped-ablation)
- Training commit: `fec86dc0b2fd23e2e5433b6c65380a9baad1a9f5`
- Cohort SHA-256: `94fdb9d6195604243b5fbb290b7c1d38ceb3fee810f959fc3e06514783ae366a`
- Hardware: one Tesla P100-PCIE-16GB
- Software: Python 3.12.13, PyTorch 2.7.1 + CUDA 11.8, ViennaRNA features enabled
- Source: competition-mounted `OLD/train_data.csv`; `OLD` is Kaggle's
  directory name and does not indicate generated data

The kernel sampled 1,000 quality-eligible unique sequences with sample seed 42
and retained 1,820 experimental profiles. Validation seeds 42, 123, and 2026
each used row-random and exact-sequence-grouped 70/15/15 splits for four model
variants.

## Headline results

| Model | Grouped CV MAE | Held-out test MAE |
|---|---:|---:|
| CNN only | 0.2249 ± 0.0052 | 0.2221 ± 0.0037 |
| CNN + Bi-LSTM | 0.2175 ± 0.0059 | 0.2137 ± 0.0042 |
| CNN + LSTM + Transformer | 0.2268 ± 0.0048 | 0.2228 ± 0.0040 |
| Full model (+ ViennaRNA 7d) | **0.1947 ± 0.0055** | **0.1919 ± 0.0026** |

Model selection used the lowest mean grouped CV MAE. Values are mean ± sample
standard deviation across three seeds. Each seed defines a different held-out
partition, so the test result is a repeated-holdout estimate, not an external
test or Kaggle leaderboard score.

Adding three ViennaRNA MFE channels improved grouped CV MAE from 0.2268 to
0.1947 within the same Transformer architecture: 0.0320 absolute (14.1%
relative). This is a controlled feature comparison, not a comparison against
the CNN + Bi-LSTM model. In this validated run, the ViennaRNA model was also the
grouped-CV-selected model.

## Leakage and cohort audit

All grouped train/CV/test exact-sequence overlap counts were zero. Row-random
splits contained repeated sequences across partitions. The grouped-minus-random
CV differences were +0.0044 for CNN only, +0.0072 for CNN + Bi-LSTM, +0.0061
for the Transformer, and +0.0004 for the ViennaRNA model. These deltas are
descriptive because partition composition changes along with leakage removal.

The exact-length artifact reports these non-zero unique-sequence counts:

| Length (nt) | Unique sequences | Experiment profiles |
|---:|---:|---:|
| 115 | 15 | 147 |
| 155 | 3 | 18 |
| 170 | 54 | 106 |
| 177 | 915 | 1,523 |
| 206 | 13 | 26 |

All other integer lengths from 115 through 206 have zero unique sequences.
This highly concentrated distribution is real for the sampled cohort and is a
reason not to make strong length-generalization claims.

## Independent validation

The repository validator and review checks confirmed:

- 24 expected seed/split/model evaluations and 15 epochs per evaluation;
- zero exact-sequence overlap for every grouped partition pair;
- grouped partitions of 700/150/150 unique sequences for every seed;
- exact agreement between per-seed and aggregate statistics;
- held-out error summaries reconciling to test MAE within `7.5e-9`;
- 24 original checkpoints load successfully, contain finite tensors, and have
  the expected parameter counts; and
- no traceback, out-of-memory event, synthetic-data fallback, or incomplete
  training output in the downloaded kernel log.

Run the compact archive check with:

```bash
python experiments/validate_archive.py results/kaggle-v10-fec86dc
```

## Limitations and contents

Exact-sequence grouping does not separate near-identical mutants or RNA library
families. The short held-out length strata contain very few sequences. CUDA
attention also emitted a nondeterministic-algorithm warning under
`warn_only=True`, so fixed seeds do not guarantee bitwise-identical reruns.

The archive includes JSON provenance and audits; CSV epoch, score, leakage,
ViennaRNA, exact-length, and error tables; and three PNG figures. Competition
data, full logs, checkpoints, and profile-level error rows remain on Kaggle and
are intentionally excluded from Git.
