# Leakage-Safe RNA Reactivity Prediction

A reproducible PyTorch ablation study for per-nucleotide RNA chemical
reactivity prediction using the
[Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)
dataset. The project compares CNN, Bi-LSTM, Transformer, and ViennaRNA-derived
secondary-structure features while auditing exact-sequence leakage in ordinary
row-random validation.

The reportable experiment uses real competition measurements only. Competition
data and model checkpoints are not redistributed.

## Archived V10 benchmark

> **Historical, not the final headline result.** V10 used Kaggle's real
> `OLD/train_data.csv` snapshot. Kaggle now provides a corrected current
> `train_data.csv` with changed error fields and a small number of changed
> `SN_filter` values. The models did not consume error fields, but they did use
> `SN_filter`; therefore V10 remains an auditable benchmark and is not presented
> as the final current-data estimate. New runs fail closed if only `OLD` is
> available.

Kaggle Version 10 trained four controlled model variants on a quality-filtered
cohort of 1,000 unique RNA sequences (1,820 experimental profiles). Three
70/15/15 repeated holdouts used seeds 42, 123, and 2026. Model selection used
mean sequence-grouped CV MAE; each selected checkpoint was then evaluated on
that seed's held-out test partition.

| Model | Parameters | Grouped CV MAE | Held-out test MAE |
|---|---:|---:|---:|
| CNN only | 43,074 | 0.2249 ± 0.0052 | 0.2221 ± 0.0037 |
| CNN + Bi-LSTM | 702,786 | 0.2175 ± 0.0059 | 0.2137 ± 0.0042 |
| CNN + LSTM + Transformer | 2,282,306 | 0.2268 ± 0.0048 | 0.2228 ± 0.0040 |
| Full model (+ ViennaRNA 7d) | 2,283,266 | **0.1947 ± 0.0055** | **0.1919 ± 0.0026** |

Values are mean ± sample standard deviation over three seeds. The full model
was selected from grouped CV only. Its test value is an internal repeated-
holdout estimate, not a Kaggle leaderboard score or external validation result.

The same Transformer architecture improved from 0.2268 to 0.1947 grouped CV
MAE when three ViennaRNA MFE structure channels were added: an observed absolute
improvement of 0.0320 (14.1% relative). On held-out partitions, the corresponding
mean difference was 0.0310. The ViennaRNA model was also the best model in this
run; this supersedes the model ranking from the earlier single-split exploratory
experiment.

The complete compact evidence package is in
[`results/kaggle-v10-fec86dc`](results/kaggle-v10-fec86dc/README.md). It records the
training commit, cohort fingerprint, environment, split audits, per-seed scores,
aggregates, exact-length counts, and error analysis. The archived headline
metrics were independently recomputed from saved error numerators and
denominators to within `7.5e-9`; a SHA-256 inventory protects every compact
CSV, JSON, and PNG artifact.

## Research question

The study asks two scoped questions:

1. How do local, recurrent, attention-based, and predicted-structure features
   compare under the same training and evaluation protocol?
2. How much does row-random validation differ from validation that keeps every
   occurrence of an exact RNA sequence in one partition?

The study measures associations within this cohort. It does not claim
state-of-the-art performance, causal biological mechanisms, or generalisation
to unrelated RNA families.

## Data and cohort

The Kaggle runner requires the corrected current competition
`train_data.csv` and explicitly rejects any path whose components contain
`OLD`. It then:

1. requires `SN_filter == 1` quality eligibility;
2. samples unique sequences with a recorded sampling seed (or uses the full
   quality-filtered cohort when `--max-sequences 0` is requested);
3. retains every eligible experimental profile for those sequences; and
4. validates sequence, experiment, and reactivity fields before training.

The archived V10 cohort contains:

- 1,820 profiles across 1,000 exact sequences;
- 864 `2A3_MaP` and 956 `DMS_MaP` profiles;
- 691 sequences with both experiment types and 309 with one;
- 178,937 measured nucleotide targets; and
- sequence lengths from 115 to 206 nt (median 177 nt).

New Kaggle runs save the complete exact-length cohort table as
`sequence_length_distribution.csv` and a 1-nt unique-sequence histogram as
`sequence_length_histogram.png`. Counts are computed after deduplicating
experiment-profile rows by exact sequence, and zero-count lengths between the
minimum and maximum remain explicit in the CSV.

There are no exact duplicate rows. The quality report records 129 repeated
sequence/experiment keys rather than silently removing them because the source
can contain distinct experimental profiles for one key. Grouping by sequence
keeps all such profiles in the same partition.

## Leakage-safe evaluation

Both split strategies use the same cohort, seeds, model configurations, and
metric:

- **Row-random:** rows are split directly; repeated sequences can cross
  partitions.
- **Exact-sequence-grouped:** unique sequence identities are assigned 70/15/15 to
  train, CV, and test, then all associated profiles follow that assignment.
- **Similarity-clustered:** MMseqs2 connected components use at least 80%
  sequence identity and at least 80% coverage of both sequences. Whole clusters
  are allocated 70/15/15 using measured-target counts, so exact sequences and
  operationally defined near neighbours cannot cross partitions.

The MMseqs2 preprocessing records its version and parameters, uses a sorted
SHA-256-labelled FASTA, one thread, single-step connected-component clustering,
and a frozen assignment checksum. Training validates the source hash, cohort
membership, canonical cluster IDs, and zero cluster overlap before fitting.

The row-random splits contained 153–158 overlapping sequences between train and
CV and 151–161 between train and test. Every grouped train/CV/test overlap count
was zero. Grouping raised mean CV MAE by 0.0044 for CNN only, 0.0072 for CNN +
Bi-LSTM, 0.0061 for the Transformer, and 0.0004 for the ViennaRNA model.

These differences show that the row-random protocol is contaminated by exact-
sequence reuse. Their magnitude is model- and split-dependent: it combines
leakage removal with ordinary partition-composition variation and should not be
reported as a universal causal leakage effect.

## Model design

```text
ACGU one-hot ──> padding-safe CNN ──> packed Bi-LSTM ──> masked Transformer ──> reactivity
     + optional ViennaRNA MFE channels: paired-left, paired-right, unpaired
```

The current ablation registry includes six variants:

1. CNN only;
2. CNN + Bi-LSTM;
3. CNN + Bi-LSTM + Transformer; and
4. the same full sequence architecture with three real ViennaRNA structure
   channels;
5. the identical 7D architecture with ViennaRNA positions deterministically
   shuffled within each RNA; and
6. the identical 7D architecture with all three structure channels zeroed.

The real, shuffled, and zero controls have identical parameter counts and, for
the same training seed, identical initialisation. The shuffle seed is fixed
independently of the training seed. This separates useful position-specific
structure information from extra input width or parameter count.

Padding is excluded at each stage: per-position LayerNorm avoids batch-statistic
contamination, CNN padded activations are zeroed, the Bi-LSTM uses packed
sequences, and the Transformer receives an explicit padding mask. Tests check
padding invariance in both evaluation and training mode.

Training minimises masked L1 loss against targets clipped to `[0, 1]` while
leaving predictions unclipped so out-of-range predictions retain gradients.
Evaluation clips both predictions and targets and reports nucleotide-weighted
MAE over measured, non-padding positions.

## Held-out evaluation and error analysis

Every checkpoint is re-evaluated on CV after epoch selection. Run modes enforce
test discipline:

- `smoke`: exercises CV and test plumbing for all requested variants but writes
  `reportable=false`;
- `development`: trains and evaluates CV only; test is not accessed; and
- `final-eval`: requires one frozen similarity manifest, selects the variant by
  mean CV MAE, then and only then evaluates that selected variant on the fixed
  test membership across training seeds.

Outputs include nucleotide-weighted MAE and macro-sequence MAE. The latter first
pools all profiles for each exact RNA, then gives every sequence equal weight.
Controlled ViennaRNA comparisons use a paired 10,000-replicate bootstrap that
resamples whole similarity clusters. Positive improvement is defined as
reference MAE minus candidate MAE.

The archive contains error summaries by:

- sequence length;
- chemical-probing experiment (`2A3_MaP` versus `DMS_MaP`); and
- ViennaRNA paired versus unpaired position.

Schema v3 runs also fit one constant reactivity value per experiment type using
training targets only, then score that trivial baseline on CV and test. It is a
sanity reference and is never included in neural-model selection.

The short 101–150 nt stratum contains few unique sequences in each grouped test
split, so the length figure is descriptive and is not used for a strong subgroup
claim.

## Reproducibility

### Local CPU checks

```bash
python -m pip install -r requirements.txt
python -m compileall -q src experiments kaggle tests
python -m pytest -q
python experiments/validate_archive.py results/kaggle-v10-fec86dc
```

### Kaggle GPU run

1. Accept the Ribonanza competition rules and attach the competition source.
2. Select one Kaggle P100 and enable Internet for dependency, MMseqs2, and repository
   access.
3. Configure `~/.kaggle/kaggle.json`.
4. Run `bash kaggle/run_gpu_ablation.sh`, or push `kaggle/` with the Kaggle CLI.

The next submitted run is deliberately a non-reportable smoke test: 256 current-
data sequences, one epoch, seed 42, all three split strategies, and four
Transformer controls (sequence-only, real ViennaRNA, shuffled ViennaRNA, and
zero structure channels). Only after this passes should a full development run
consume substantial GPU time.

The archived run used Python 3.12.13, PyTorch 2.7.1 + CUDA 11.8, ViennaRNA
features, and one Tesla P100 16 GB. New runs pin ViennaRNA 2.7.2, record exact
dependency versions, hash both the full source CSV and the selected cohort
including measured targets, and emit a train-only experiment-mean baseline.
The runner fails if measured competition data or ViennaRNA is unavailable; it
has no generated-data fallback.

## Repository layout

```text
src/                         models, data pipelines, MMseqs splitting, statistics
experiments/build_similarity_manifest.py  frozen MMseqs2 split preprocessing
experiments/run_ablation.py  repeated ablation and held-out evaluation
experiments/validate_archive.py independent compact-result validator
kaggle/                      private Kaggle GPU entrypoint and metadata
results/kaggle-v10-fec86dc/  versioned reportable artifacts
tests/                       data, model, padding, training, and archive tests
docs/                        development journal
```

## Limitations

- The archived V10 experiment samples 1,000 quality-eligible sequences rather
  than the full Ribonanza corpus; the current-data full run is not yet complete.
- Three repeated holdouts quantify seed sensitivity but do not replace an
  external test set or RNA-family-aware benchmark.
- MMseqs2 similarity clustering is an operational threshold definition, not a
  guarantee that every biologically related RNA family is separated.
- Archived V10 seeds used different held-out partitions. Schema-v4 final runs
  instead keep one frozen cluster membership across all optimisation seeds.
- Source targets outside `[0, 1]` are retained and clipped for the competition
  metric; the archive reports their counts.
- Repeated profiles give some sequences greater weight in nucleotide-weighted
  MAE, so macro-sequence MAE is reported as a secondary robustness metric.
- ViennaRNA MFE supplies one predicted secondary structure and omits ensemble
  uncertainty and pseudoknots.
- CUDA attention emitted a deterministic-algorithm warning under
  `warn_only=True`; fixed seeds improve repeatability but do not guarantee
  bitwise-identical GPU reruns.

## Historical experiments

Early single-split experiments are retained in
[`experiments/experiment_notes.md`](experiments/experiment_notes.md) as a
development record. They used a different padding implementation and evaluation
protocol and are not comparable with, or used to support, the current model
ranking.

## References

- Stanford Ribonanza RNA Folding Competition:
  https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding
- Lorenz et al. (2011), *ViennaRNA Package 2.0*, Algorithms for Molecular
  Biology 6, 26.
- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS.

## License

Code is released under the [MIT License](LICENSE). Ribonanza competition data
are not redistributed and remain subject to their original terms.
