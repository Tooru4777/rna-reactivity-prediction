# Kaggle Version 9 — validated P100 run

This directory is the compact evidence package for the reportable RNA
reactivity experiment. Files were downloaded from Kaggle kernel Version 9 after
it reached `COMPLETE` and were archived without changing their values.

## Provenance

- Run date: 2026-08-07
- Kernel: [`tooru4777/rna-reactivity-grouped-ablation`](https://www.kaggle.com/code/tooru4777/rna-reactivity-grouped-ablation)
- Training commit: `bff599f5bfa59ccf6a2d202a35231a76384b843a`
- Cohort SHA-256: `94fdb9d6195604243b5fbb290b7c1d38ceb3fee810f959fc3e06514783ae366a`
- Hardware: one Tesla P100-PCIE-16GB
- Software: Python 3.12.13, PyTorch 2.7.1 + CUDA 11.8
- Source: the competition-mounted `OLD/train_data.csv`; `OLD` is Kaggle's
  directory name and does not indicate generated data

The kernel selected 1,000 quality-eligible unique sequences with sample seed 42
and retained 1,820 associated experimental profiles. Validation seeds were 42,
123, and 2026; each seed used row-random and exact-sequence-grouped 70/15/15
splits for four model variants.

## Headline results

| Model | Grouped CV MAE | Held-out test MAE |
|---|---:|---:|
| CNN only | 0.2249 ± 0.0052 | 0.2221 ± 0.0037 |
| CNN + Bi-LSTM | 0.2175 ± 0.0059 | 0.2137 ± 0.0042 |
| CNN + LSTM + Transformer | 0.2268 ± 0.0048 | 0.2228 ± 0.0040 |
| Full model (+ ViennaRNA 7d) | **0.1947 ± 0.0055** | **0.1919 ± 0.0026** |

Model selection used the lowest mean grouped CV MAE. Values are mean ± sample
standard deviation across the three seeds. Each seed has a different held-out
partition, so the test value is a repeated-holdout estimate rather than an
external or Kaggle leaderboard score.

Within the same Transformer architecture, adding three ViennaRNA MFE channels
improved grouped CV MAE by 0.0320 on average and held-out test MAE by 0.0310.
The feature comparison adds 960 input-projection parameters (2,282,306 versus
2,283,266 total parameters).

## Leakage audit

Row-random splitting placed repeated exact sequences across partitions. Across
the three seeds, train/CV overlap ranged from 153 to 158 sequences and
train/test overlap from 151 to 161. All nine pairwise overlap checks across the
three grouped splits were zero.

| Model | Random CV MAE | Grouped CV MAE | Grouped − random |
|---|---:|---:|---:|
| CNN only | 0.2205 | 0.2249 | +0.0044 |
| CNN + Bi-LSTM | 0.2104 | 0.2175 | +0.0072 |
| CNN + LSTM + Transformer | 0.2207 | 0.2268 | +0.0061 |
| Full model (+ ViennaRNA 7d) | 0.1943 | 0.1947 | +0.0004 |

The overlap establishes contamination in the row-random protocol. Score deltas
are descriptive rather than causal because partition composition also changes.

## Independent validation

The archive validator confirmed:

- 24 expected seed/split/model held-out evaluations;
- 24 non-empty checkpoints in the original Kaggle output (not archived here);
- zero exact-sequence overlap for every grouped partition pair;
- complete grouped partitions of 700/150/150 unique sequences for every seed;
- exact agreement between per-seed results and aggregate mean/standard deviation;
- agreement between held-out MAE and saved error numerators/denominators within
  `7.5e-9`; and
- no traceback, out-of-memory failure, generated-data fallback, or incomplete
  artifact in the original kernel output.

Run the repository check with:

```bash
python experiments/validate_archive.py results/kaggle-v9-bff599f
```

## Data-quality and interpretation caveats

- The cohort has no exact duplicate rows but includes 129 repeated
  sequence/experiment keys. They are reported rather than silently removed;
  exact-sequence grouping keeps them within one partition.
- Of 178,937 measured targets, 27,187 are below zero and 20,063 are above one.
  Source values are retained, then predictions and targets are clipped to
  `[0, 1]` for the reported competition metric.
- The 101–150 nt held-out length bin contains only 1–5 unique grouped-test
  sequences per seed, whereas the 151–206 nt bin contains 145–149. The length
  plot is therefore descriptive and does not support a strong subgroup claim.
- Grouping is based on exact sequence identity and does not remove similarity
  between near-identical mutants or library families.
- CUDA attention emitted a deterministic-algorithm warning with
  `warn_only=True`; fixed seeds do not guarantee bitwise-identical reruns.

## Contents

- JSON: run provenance, environment, data quality, split audit, model selection
- CSV: per-epoch, per-seed, aggregate, leakage, ViennaRNA, and error summaries
- PNG: mean validation curves and selected-model length-stratified error

Competition data, full logs, checkpoints, and profile-level error rows remain on
Kaggle and are intentionally excluded from Git.
