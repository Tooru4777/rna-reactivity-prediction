# RNA Reactivity Prediction

An independent learning project based on public data from the completed
[Stanford Ribonanza RNA Folding competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding).

I developed this project to practise computational RNA analysis and deep-learning
workflows while transitioning from experimental mRNA research to computational
biology. It was not submitted to the original competition and is not presented as
a competition result or publication-level study.

## Motivation

RNA molecules fold into complex 3D structures that determine their biological function. Accurately predicting these structures from sequence alone remains an open challenge in computational biology. This project explores deep learning approaches to two related problems:

1. **Reactivity Prediction**: Predicting per-nucleotide chemical reactivity values (2A3_MaP and DMS_MaP) that serve as proxies for RNA flexibility and accessibility.
2. **3D Coordinate Prediction**: Predicting the spatial (x, y, z) coordinates of each nucleotide in a folded RNA molecule.

## Architecture

### Reactivity Model

```
Input (7-dim)  →  CNN (2-layer + BatchNorm)  →  Bi-LSTM  →  Transformer Encoder  →  Output (2-dim)
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
| Dual-layer CNN + BatchNorm | Feature hierarchy: primitive patterns → higher-order motifs |
| Bi-LSTM before Transformer | Implicit positional encoding; captures sequential dependencies |
| Transformer residual connection | Stabilises early training; model can fall back to LSTM features |
| L1 Loss (MAE) over MSE | More robust to outlier reactivity values in the dataset |
| Clamp targets only (not predictions) | Avoids gradient vanishing — [see debugging story below](#debugging-gradient-vanishing) |

### 3D Structure Model (Experimental Prototype)

A simpler CNN + Bi-LSTM architecture that maps one-hot encoded sequences directly to (x, y, z) coordinates per nucleotide.

This part of the repository is a pipeline prototype rather than a validated
structure-prediction result. Raw coordinate MSE is not rotation/translation
invariant, and the current prototype does not yet use alignment-aware metrics
or an equivariant architecture. I keep it here to document what I tried and
what I still need to learn.

## Project Structure

```
rna-reactivity-prediction/
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
│   ├── kaggle_training_script.py   ← Reactivity model (merged for Kaggle Notebook)
│   ├── kaggle_3d_training_script.py ← 3D model (merged for Kaggle Notebook)
│   └── download_data.sh       ← Dataset download helper
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

### Quick Test (Local, CPU)

```bash
# Explicit smoke tests use deterministic synthetic data.
# Synthetic outputs are never presented or saved as research results.
python src/train_reactivity.py --smoke-test --epochs 1
python src/train_3d.py --smoke-test
```

### Full Training (Kaggle GPU)

1. Upload `kaggle/kaggle_training_script.py` to a Kaggle Notebook
2. Attach the [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) dataset
3. Enable GPU acceleration (P100 or T4)
4. Run all cells — training takes ~2 hours for 50 epochs

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

A systematic ablation study quantifies the contribution of each architectural component. All variants were trained under identical conditions (seed=42, 15 epochs, 1000 samples, L1 loss with target-only clamping).

> **Status of these results:** this table records an early learning run on a
> local 1,000-row subset. The subset and generated CSV logs were not committed,
> and the original split was row-wise. I therefore treat these numbers as
> preliminary observations, not a reproducible benchmark. The current code now
> requires real data explicitly and uses sequence-grouped splitting; the table
> will be replaced after a clean rerun.

| Variant | Parameters | Preliminary CV Loss (Clipped MAE) | Best Epoch | Training Time |
|---------|-----------|---------------------------|------------|---------------|
| CNN Only | 43,074 | 0.1281 | 15 | 19s |
| CNN + Bi-LSTM | 702,786 | **0.1186** | 15 | 85s |
| CNN + LSTM + Transformer | 2,282,306 | 0.1334 | 14 | 363s |
| Full Model (+ ViennaRNA 7d) | 2,283,266 | 0.1258 | 15 | 363s |

**Preliminary observations**:
1. Under this one split and seed, CNN + Bi-LSTM had the lowest CV loss.
2. The Transformer variants did not improve this small-data run. Higher CV
   loss alone cannot distinguish overfitting from optimisation or
   hyperparameter effects.
3. ViennaRNA features improved the Transformer result within this run, but
   multiple seeds and a leakage-aware rerun are required before making a
   biological conclusion.

**Interpretation**: These results helped me learn that model complexity is not
a substitute for careful validation. A future rerun will report multiple seeds,
mean ± standard deviation, fixed sequence-grouped splits, and a held-out test
set.

*Current rerun command: `python experiments/run_ablation.py --data path/to/train_data.csv`*

*See [experiments/experiment_notes.md](experiments/experiment_notes.md) for detailed hyperparameter tuning history.*

### Training Curves

![Training curves showing loss vs. epoch for all four model variants](results/training_curves.png)

The plot is retained as a record of the early experiment. It should not be
treated as the output of the current leakage-aware pipeline.



## Learning Journey

This project represents my first end-to-end deep learning project applied to a real bioinformatics problem. I started from foundational ML concepts and progressively built a competition-inspired research pipeline. Key learning milestones:

1. **Data Engineering**: Learned to handle variable-length biological sequences with padding and masking
2. **Architecture Design**: Tested hybrid architectures (CNN → LSTM → Transformer) and learned that **more complex ≠ better**, especially when data and validation are limited
3. **Debugging ML Models**: Discovered and fixed a gradient vanishing bug caused by incorrect loss clamping
4. **Systematic Evaluation**: Designed and executed an ablation study to quantify the contribution of each component
5. **Competition ML**: Learned the importance of aligning training loss with evaluation metrics

See [docs/learning_journal.md](docs/learning_journal.md) for a more detailed reflection.

## Reproducibility

All experiments use fixed random seeds for reproducible results:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

The training commands fail fast when real data are missing. Synthetic data are
available only through an explicit `--smoke-test` flag and are intended for
software checks, never model evaluation.

To rerun the full ablation study after downloading the competition data:
```bash
pip install -r requirements.txt
python experiments/run_ablation.py --data path/to/train_data.csv
python experiments/plot_results.py    # Generate visualisations
```

Research safeguards in the current pipeline:

- identical RNA sequences are kept in the same train/CV/test partition;
- LSTM packing and Transformer padding masks prevent padded positions from
  entering the learned sequence context;
- epoch metrics are weighted by the total number of valid nucleotide targets;
- missing data cause a clear error instead of a silent synthetic fallback;
- smoke-test samples are deterministic per index.

## Limitations

- The preliminary ablation table must be rerun with the current grouped split.
- Results currently use one seed; uncertainty across seeds is not yet reported.
- ViennaRNA MFE is a useful prior but not experimental ground-truth structure.
- The 3D branch is an educational prototype and lacks alignment-aware
  evaluation.
- This is my first end-to-end computational RNA project. I am documenting both
  successful steps and mistakes as I transition from wet-lab mRNA work toward
  reproducible dry-lab research.

## Future Directions

- [ ] **Scale to full Kaggle dataset** (~50k samples) to test whether the Transformer advantage emerges at scale — the ablation study suggests it needs more data
- [ ] Explore **Graph Neural Networks** (GNN) to directly model base-pairing interactions, following Joshi et al. (2023) on RNA structure prediction with GNNs
- [ ] Investigate **SE(3)-equivariant architectures** for physically consistent 3D predictions, inspired by Townshend et al. (2021) geometric deep learning for RNA
- [ ] **Multi-task learning**: jointly predict reactivity and secondary structure, testing the hypothesis that 2D structure as an auxiliary task provides useful inductive bias
- [ ] Evaluate biologically justified augmentation rather than assuming
      reverse-complement invariance

## References

- Stanford Ribonanza RNA Folding Competition: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding
- ViennaRNA Package: https://www.tbi.univie.ac.at/RNA/
- Lorenz et al. (2011). "ViennaRNA Package 2.0." *Algorithms for Molecular Biology*, 6(1), 26.
- Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Townshend et al. (2021). "Geometric deep learning of RNA structure." *Science*, 373(6558), 1047-1051.

## License

Released under the [MIT License](LICENSE).

