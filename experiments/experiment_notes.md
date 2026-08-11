# Experiment Notes

> Historical development log. Early observations below came from exploratory
> single-split runs and are not treated as final evidence. The current
> padding-safe, grouped, multi-seed protocol and its versioned result artifacts
> supersede these notes for model selection and external reporting. BatchNorm
> was later replaced by per-position LayerNorm after a padding-statistics audit.

## Hyperparameter Tuning Log

### Experiment 1: Baseline Architecture (v1)
- **Date**: 2026-05-15
- **Architecture**: Single CNN (64 filters) + Bi-LSTM (128 hidden) + Transformer (2 layers, 8 heads)
- **Input**: 7-dim (ACGU + ViennaRNA structure)
- **Loss**: L1Loss (MAE), reduction='none' with masking
- **Optimiser**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Observations**:
  - Model trains but CV loss plateaus early
  - Suspected underfitting due to limited CNN capacity

---

### Experiment 2: Deeper CNN + Aligned Metric (v2)
- **Date**: 2026-05-19
- **Changes**:
  - Dual-layer CNN (64 → 128 filters) with BatchNorm after each layer
  - `cnn_out_dim` increased from 64 → 128
  - Added Dropout between CNN layers
  - Training loss aligned with Kaggle's Clipped MAE evaluation metric
  - `weight_decay` reduced: 1e-4 → 1e-5 (was over-regularising)
  - `scheduler patience` reduced: 3 → 2 (more aggressive LR decay)
- **Observations**:
  - BatchNorm significantly improved gradient flow through CNN layers
  - But discovered a critical bug: clamping predictions in training loss caused gradient vanishing
  - Model stopped learning after ~10 epochs

---

### Experiment 3: Gradient Vanishing Fix + Residual Connection (v3)
- **Date**: 2026-05-19
- **Changes**:
  - Fixed gradient vanishing: only clamp targets, not predictions, during training
  - Added residual connection from LSTM output to Transformer output
  - Evaluation still uses fully clipped MAE (both preds and targets in [0,1])
- **Observations**:
  - Training loss now decreases smoothly across all epochs
  - Residual connection improved early-epoch stability
  - Model converges to a reasonable CV loss

---

## Key Takeaways

1. **Training vs. evaluation metrics can differ**: It's valid (and sometimes necessary) to use a different loss formulation during training than the final evaluation metric, as long as they optimise the same objective.

2. **Padding changes normalization conclusions**: BatchNorm appeared to
   stabilise the early exploratory model, but a later padding audit showed that
   its batch statistics changed real-position outputs when padding changed. The
   reportable model uses per-position LayerNorm and explicitly zeroes padded CNN
   activations.

3. **Residual connections are cheap insurance**: The `transformer_out + lstm_out` residual costs almost nothing computationally but prevents the Transformer from hurting performance when it hasn't learned useful patterns yet.

4. **Weight decay needs careful tuning**: 1e-4 was too aggressive for this dataset size, causing underfitting. Reducing to 1e-5 improved generalisation.

## TODO: Future Experiments

- [ ] Try replacing LSTM with GRU (fewer parameters, faster training)
- [ ] Experiment with larger Transformer (4 layers instead of 2)
- [ ] Add positional encoding explicitly instead of relying on LSTM
- [ ] Try multi-task learning: predict reactivity + secondary structure jointly
- [ ] Data augmentation: reverse complement sequences

---

### Experiment 4: Ablation Study & ViennaRNA Integration (v4)
- **Date**: 2026-07-14
- **Changes**:
  - Extracted hyperparameter configuration to `configs/ablation_config.yaml`
  - Added ViennaRNA package to supply 7-dimensional features for the full model
  - Trained 4 model variants (CNN Only, CNN+Bi-LSTM, CNN+LSTM+Transformer, Full Model) under identically controlled settings
- **Historical observations**:
  - CNN + Bi-LSTM had the lowest CV score in this one row-random split (0.1186).
  - The Transformer variant scored 0.1334, while the same architecture with
    ViennaRNA features scored 0.1258.
  - This run used the earlier padding implementation and a single contaminated
    row-random split. It is retained as development history and does not support
    the current model ranking or a general ViennaRNA-effect claim.

---

### Experiment 5: Padding-safe repeated grouped evaluation (reportable)

- **Validated rerun date**: 2026-08-11
- **Training commit**: `fec86dc0b2fd23e2e5433b6c65380a9baad1a9f5`
- **Compute**: one Kaggle Tesla P100 16 GB
- **Cohort**: 1,000 quality-eligible sequences, 1,820 profiles
- **Protocol**: row-random and exact-sequence-grouped 70/15/15 splits; seeds
  42, 123, and 2026; CV checkpoint selection followed by held-out testing
- **Metric**: nucleotide-weighted clipped MAE
- **Results**:
  - Full model (+ ViennaRNA 7d): grouped CV `0.1947 ± 0.0055`, repeated
    held-out test `0.1919 ± 0.0026`
  - Same Transformer without ViennaRNA: grouped CV `0.2268 ± 0.0048`, test
    `0.2228 ± 0.0040`
  - CNN + Bi-LSTM: grouped CV `0.2175 ± 0.0059`, test `0.2137 ± 0.0042`
  - All grouped train/CV/test exact-sequence overlaps were zero
- **Interpretation**:
  - ViennaRNA channels improved the controlled Transformer comparison by
    `0.0320` grouped CV MAE and the full model ranked first in this experiment.
  - Random/grouped score gaps were model-dependent; exact overlap proves the
    random split was contaminated, but score deltas also reflect split
    composition.
  - Results are internal repeated holdouts, not external validation or a
    competition leaderboard result.

See [`results/kaggle-v10-fec86dc`](../results/kaggle-v10-fec86dc/README.md) for
the compact evidence package.
