# V17: current-data CV development evidence

This archive contains 28 unmodified, checksum-verified compact artifacts from
Kaggle Version 17, training commit `9d904ff15892e7453dd5ccfba8366657c7a3a473`.
It is non-reportable development evidence, not final-test or leaderboard evidence.
No source measurements, profile-level outputs, logs or model weights are included.

Protocol: 1,000 current SN-filtered unique RNA sequences, three training seeds
(42, 123, 2026), six variants, three split strategies, 20 epochs, T4 x2.
The clustered partition is fixed across training seeds; standard deviations
therefore quantify training-seed variation, not uncertainty across new cohorts.

Mean clustered CV nucleotide-weighted MAE:

| Variant | MAE |
|---|---:|
| Train-only assay mean | 0.2933 |
| CNN | 0.2316 |
| CNN + Bi-LSTM | 0.2235 |
| Transformer sequence-only | 0.2318 |
| Transformer real ViennaRNA | 0.2086 |
| Transformer zero channels | 0.2314 |
| Transformer shuffled ViennaRNA | 0.2331 |

All 54 CV evaluations beat the mean baseline; none triggered the 95% clipping
or near-constant warning. Real-vs-zero/shuffled paired cluster-bootstrap
intervals are positive for each training seed, but remain exploratory CV
comparisons. Two missed-edge repairs were required before the similarity audit
passed. Twenty-three checkpoints were selected at the final epoch, motivating
a convergence experiment rather than declaring all models converged.

No dedicated test evaluations were performed. However, random/grouped runs
used the same full cohort, so the clustered test membership must not be reused
as an independent final test. A future final cohort must exclude every sequence
used in development and qualifying similarity neighbours.

Validate with:

```bash
python experiments/validate_development_archive.py results/kaggle-v17-9d904ff
```
