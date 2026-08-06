# Leakage-Safe RNA Reactivity Prediction

*Note: In the context of this project, "RL" stands for **Representation Learning**, focusing on learning meaningful representations of RNA sequences, not Reinforcement Learning.*

An ablation study of CNN, Bi-LSTM, Transformer, and ViennaRNA-derived features
for per-nucleotide chemical reactivity prediction using data from the
[Stanford Ribonanza RNA Folding Competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding).
The primary contribution is a reproducible, sequence-grouped evaluation that
detects row-level leakage and separates CV model selection from held-out testing.

## Motivation

RNA molecules fold into structures that influence their biological function.
Chemical probing measurements provide nucleotide-resolution signals related to
local flexibility and accessibility. This repository focuses on:

1. **Reactivity Prediction**: Predicting per-nucleotide chemical reactivity values (2A3_MaP and DMS_MaP) that serve as proxies for RNA flexibility and accessibility.
2. **Evaluation methodology**: Preventing repeated sequences from crossing data
   partitions and measuring model/feature effects across repeated holdouts.

The repository also contains an early 3D-coordinate pipeline prototype. It
accepts real sequence/coordinate CSVs only and is not presented as a validated
scientific model.

## Architecture

### Reactivity Model

```
Input (7-dim)  →  CNN (2-layer + LayerNorm)  →  Bi-LSTM  →  Transformer Encoder  →  Output (2-dim)
  ↑                                                ↑              ↑                      ↑
ACGU + 2D         Local motif               Sequential       Self-attention         2A3 & DMS
structure         extraction               context (both     for long-range        reactivity
from ViennaRNA                             directions)       base interactions
```

**Input features (7 dimensions per nucleotide)**:
- 4 dims: One-hot encoded sequence (A, C, G, U)
- 3 dims: One-hot encoded secondary structure from ViennaRNA MFE (`(`, `)`, `.`)

**Key design choices**:
| Decision | Rationale |
|----------|-----------|
| Dual-layer CNN + per-position LayerNorm | Local motif hierarchy without mixing padding into batch statistics |
| Bi-LSTM before Transformer | Implicit positional encoding; captures sequential dependencies |
| Transformer residual connection | Stabilises early training; model can fall back to LSTM features |
| L1 Loss (MAE) over MSE | More robust to outlier reactivity values in the dataset |
| Clamp targets only (not predictions) | Avoids gradient vanishing — [see debugging story below](#debugging-gradient-vanishing) |

### Experimental 3D Prototype

A CNN + Bi-LSTM prototype maps one-hot encoded sequences to nucleotide
coordinates. It requires real coordinate labels and remains outside the main
reactivity result because geometry-aware evaluation has not been completed.

## Project Structure

```
RNA_RL_bioresearch/
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── src/                       ← Core source code (modular, importable)
│   ├── __init__.py
│   ├── model_reactivity.py    ← Reactivity model variants (ablation)
│   ├── data_pipeline_reactivity.py ← Dataset for reactivity prediction
│   ├── model_3d.py            ← CNN + Bi-LSTM for 3D coordinates
│   ├── data_pipeline_3d.py    ← Dataset for 3D coordinate data
│   ├── train_3d.py            ← 3D training pipeline with masked loss
│   └── inference_3d.py        ← Inference + 3D visualisation
│
├── kaggle/                    ← Kaggle competition code (self-contained)
│   ├── gpu_ablation.py        ← Current reproducible GPU entrypoint
│   ├── run_gpu_ablation.sh    ← Submit/check helper
│   ├── kernel-metadata.json   ← Private Kaggle kernel configuration
│
├── experiments/               ← Experiment scripts and logs
│   ├── run_ablation.py        ← Ablation study (4 model variants)
│   ├── plot_results.py        ← Training curves + attention heatmap
│   ├── experiment_notes.md    ← Hyperparameter tuning log
│   ├── ablation_results.csv   ← Per-epoch loss data (generated)
│   └── ablation_summary.csv   ← Summary table (generated)
│
├── notebooks/                 ← Exploratory analysis
│   └── 01_data_exploration.py ← EDA: distributions, reactivity patterns
│
├── results/                   ← Output visualisations (generated)
│   └── training_curves.png    ← Loss vs. epoch for all variants
│
└── docs/                      ← Development documentation
    └── learning_journal.md    ← Learning process and reflections
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Tests (Local, CPU)

```bash
python -m pytest -q
```

### Full Training (Kaggle GPU)

1. Attach the Ribonanza competition source to the private Kaggle kernel.
2. Select a single Kaggle P100 GPU for the reportable run.
3. Run `bash kaggle/run_gpu_ablation.sh`, or save a new version manually in Kaggle.
4. Download and archive the compact reports described in [`results/README.md`](results/README.md).

## Debugging: Gradient Vanishing

During training, I observed that the model's loss stopped decreasing after a few epochs. After investigation, I identified the root cause in the loss computation:

```python
# BUG: clamping predictions kills gradients outside [0, 1]
preds_clipped = torch.clamp(predictions, 0.0, 1.0)   # ← gradient = 0 when pred > 1 or pred < 0
loss = criterion(preds_clipped, targets_clipped)

# FIX: only clamp targets, keep raw predictions for gradient flow
reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
loss = criterion(predictions, reacts_clipped)          # ← full gradient preserved
```

**Why this matters**: When `torch.clamp` is applied to predictions, any predicted value outside `[0, 1]` gets a gradient of exactly zero. The model cannot learn to pull those predictions back into range, causing training to stall. The fix is to only clamp the target values, allowing the L1 loss to produce non-zero gradients for all predictions.

This distinction between training loss (unclamped predictions) and evaluation metric (clamped predictions) is a subtle but critical detail in competition ML.

## Ablation Study

The current protocol samples 1,000 quality-eligible unique RNA sequences,
retains every experimental row for those sequences, and repeats row-random and
sequence-grouped 70/15/15 splits with seeds 42, 123, and 2026. Explicit masks
keep padding out of the CNN context, packed Bi-LSTM, and Transformer attention.
CV selects the best checkpoint; that checkpoint is then evaluated once on the
seed's held-out test partition.

The metric is nucleotide-weighted clipped MAE: predictions and targets are
clipped to `[0, 1]`, while padding and missing measurements are excluded.

### Historical exploratory run

This table is retained as project history. It used one row-random split and is
not directly comparable with the current grouped, multi-seed, padding-safe run.

| Variant | Parameters | Best CV Loss (Clipped MAE) | Best Epoch | Training Time |
|---------|-----------|---------------------------|------------|---------------|
| CNN Only | 43,074 | 0.1281 | 15 | 19s |
| CNN + Bi-LSTM | 702,786 | **0.1186** | 15 | 85s |
| CNN + LSTM + Transformer | 2,282,306 | 0.1334 | 14 | 363s |
| Full Model (+ ViennaRNA 7d) | 2,283,266 | 0.1258 | 15 | 363s |

**Historical findings**:
1. **CNN + Bi-LSTM achieves the lowest CV loss** (0.1186), outperforming the more complex Transformer variants on this small dataset.
2. Adding the Transformer increased CV loss from 0.1186 to 0.1334 in that run;
   the design did not distinguish optimization difficulty from overfitting.
3. ViennaRNA features reduced the same Transformer architecture's CV loss from
   0.1334 to 0.1258, but the single split was insufficient for a general claim.
4. Transformer variants converged more slowly under the tested hyperparameters;
   no mechanism is inferred from that observation alone.

These historical scores are not used for current model selection or
generalisation claims.

*To reproduce: first run `bash kaggle/download_data.sh`, then
`python experiments/run_ablation.py --max-samples 1000` (seed=42).*

The historical ViennaRNA gain is a controlled comparison within the Transformer
architecture (0.1334 → 0.1258 clipped MAE). It does not make the ViennaRNA model
the overall winner: CNN + Bi-LSTM remains best in this historical run at 0.1186.

*See [experiments/experiment_notes.md](experiments/experiment_notes.md) for detailed hyperparameter tuning history.*

### Kaggle GPU run

After configuring `~/.kaggle/kaggle.json` and accepting the competition rules,
submit the private GPU kernel with `bash kaggle/run_gpu_ablation.sh`. The kernel
mounts the competition data and writes reproducibility metadata, grouped-split
checks, per-epoch results, a summary, learning curves, and best model weights to
its downloadable `results/` output directory. It selects unique RNA sequences,
keeps every quality-eligible experiment row for those sequences, and runs both row-random and
sequence-grouped validation so `leakage_comparison.csv` reports their observed
score difference alongside the overlap audit. Sequence sampling occurs after `SN_filter=1` eligibility is
established, and three validation seeds (42, 123, 2026) produce aggregate mean
and standard-deviation reports for model ranking, leakage, and ViennaRNA effect.
The reportable run uses a single Kaggle P100 (`sm_60`). The kernel pins a CUDA
11.8 PyTorch build compatible with that architecture. The training code remains
portable to multiple GPUs and saves unwrapped checkpoints that can be loaded on
a single GPU or CPU. The CLI
submission helper does not set a timeout because Kaggle counts queue time toward
that limit; runtime remains bounded by the finite seed/model/epoch loops.

Each run also writes held-out test scores, a machine-readable model-selection
record, per-sequence errors, error summaries by sequence length and experiment
type, paired/unpaired error summaries, and an error-by-length figure. Compact
reports and figures are archived in Git; checkpoints and competition data remain
Kaggle artifacts because of their size and source terms.

### Historical Training Curves

![Training curves showing loss vs. epoch for all four model variants](results/training_curves.png)

This figure belongs to the historical single-split run. Final figures from the
padding-safe held-out protocol will be stored in a versioned results directory.



## Learning Journey

This project is an end-to-end deep learning study applied to a real
bioinformatics dataset. Key learning milestones:

1. **Data Engineering**: Learned to handle variable-length biological sequences with padding and masking
2. **Architecture Design**: Compared local, sequential, attention-based, and domain-feature models under controlled settings
3. **Debugging ML Models**: Discovered and fixed a gradient vanishing bug caused by incorrect loss clamping
4. **Systematic Evaluation**: Designed and executed an ablation study to quantify the contribution of each component
5. **Competition ML**: Learned the importance of aligning training loss with evaluation metrics

See [docs/learning_journal.md](docs/learning_journal.md) for a more detailed reflection.

## Reproducibility

All experiments use fixed sampling and validation seeds for reproducibility:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

To reproduce the full ablation study:
```bash
pip install -r requirements.txt
python experiments/run_ablation.py --split-method both --seeds 42 123 2026
python experiments/plot_results.py    # Generate visualisations
```

## Future Directions

- [ ] Scale beyond the 1,000-sequence quality-filtered cohort and report the compute/accuracy trade-off
- [ ] Explore **Graph Neural Networks** (GNN) to directly model base-pairing interactions, following Joshi et al. (2023) on RNA structure prediction with GNNs
- [ ] Investigate **SE(3)-equivariant architectures** for physically consistent 3D predictions, inspired by Townshend et al. (2021) geometric deep learning for RNA
- [ ] **Multi-task learning**: jointly predict reactivity and secondary structure, testing the hypothesis that 2D structure as an auxiliary task provides useful inductive bias
- [ ] Add data augmentation (reverse complement, noise injection) for better generalisation

## Limitations

- The reportable experiment uses a 1,000-sequence sample, not the full Ribonanza corpus.
- Three repeated holdouts quantify seed sensitivity but do not replace external validation.
- Grouping prevents exact-sequence overlap but does not yet cluster near-identical mutants or library families.
- Random/grouped score differences combine leakage removal with ordinary split-composition variation; they are evidence of optimism, not a causal estimate.
- ViennaRNA MFE is a predicted single secondary structure and does not represent a full structural ensemble or pseudoknots.
- Held-out values are internal repeated-test estimates, not Kaggle leaderboard scores.
- The 3D branch is an unvalidated prototype and is outside the main empirical claim.

## References

- Stanford Ribonanza RNA Folding Competition: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding
- ViennaRNA Package: https://www.tbi.univie.ac.at/RNA/
- Lorenz et al. (2011). "ViennaRNA Package 2.0." *Algorithms for Molecular Biology*, 6(1), 26.
- Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Townshend et al. (2021). "Geometric deep learning of RNA structure." *Science*, 373(6558), 1047-1051.

## License

Released under the [MIT License](LICENSE). Ribonanza competition data are not
redistributed and remain subject to their original terms.
