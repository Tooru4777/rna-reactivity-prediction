# Learning Journal

## Background

I started this project as my first hands-on deep learning project applied to a real bioinformatics problem. My prior knowledge included foundational machine learning concepts from online courses, but I had never built a production-scale model or worked with biological sequence data before.

## Week 1: Understanding the Problem (May 2026)

### What I studied
- Read the [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) competition description and discussion forums
- Reviewed RNA biology basics: nucleotides (A, C, G, U), 5'→3' directionality, secondary structure (stems, loops, bulges)
- Understood the difference between 2A3_MaP and DMS_MaP chemical probing experiments

### Key insight
RNA chemical reactivity is a proxy for 3D structure: nucleotides that are more "exposed" (unpaired, in loops) tend to have higher reactivity. This means predicting reactivity is related to predicting structure, but is more tractable because we have direct experimental measurements.

### Challenge
The competition dataset is massive (~1.6 GB). I needed to learn how to handle data that doesn't fit in memory, which led me to understand PyTorch's `Dataset` and `DataLoader` abstraction.

## Week 2: Building the Pipeline

### Data preprocessing
- Implemented one-hot encoding for RNA sequences (A → [1,0,0,0], etc.)
- Learned about zero-padding and masking for variable-length sequences
- This was tricky: I initially forgot to mask the loss, and the model learned to predict zeros everywhere (because that minimises loss at padded positions)

### Model architecture decisions
1. **Why CNN first?** RNA has local structural motifs (hairpins, bulges) that span 3-7 nucleotides. A 1D CNN with kernel_size=5 is perfect for detecting these.
2. **Why Bi-LSTM?** RNA folding involves long-range base-pairing (e.g., position 10 pairs with position 150). A bidirectional LSTM can capture these dependencies.
3. **Why Transformer on top?** Self-attention can directly model position-to-position interactions without the sequential bottleneck of LSTM.

### What didn't work
- I initially tried a pure Transformer model, but it performed poorly. I think this is because RNA is inherently sequential, and the positional encoding wasn't sufficient. Using LSTM as an implicit positional encoder solved this.

## Week 3: Training and Debugging

### The gradient vanishing incident
This was the most educational experience of the project. My training loop looked correct, the loss decreased for the first few epochs, then completely plateaued. I spent hours checking:
- Data loading (correct ✓)
- Model architecture (correct ✓)
- Learning rate (tried multiple values, no help)

Eventually, I traced the issue to this line:
```python
preds_clipped = torch.clamp(predictions, 0.0, 1.0)
```

During training, this kills gradients for any prediction outside [0, 1]. But early in training, the model frequently predicts values outside this range. With zero gradients, it can never learn to correct them. The fix was simple but non-obvious: only clamp the targets during training, and clamp both during evaluation.

### What I learned
- Always check gradient flow when training stalls
- Training loss and evaluation metric don't have to be identical
- `torch.clamp` is not differentiable at the clamp boundaries

## Reflections

### What went well
- Built a complete ML pipeline from data loading to visualisation
- Discovered and fixed a real bug through systematic debugging
- Learned to use ViennaRNA for secondary structure prediction as a feature

### What I would do differently
- Start with a simpler baseline and add complexity incrementally
- Set up proper experiment tracking (e.g., Weights & Biases) from the beginning
- Write unit tests for the data pipeline before training
- Split by RNA sequence rather than by measurement row, so paired experiments
  cannot leak across training and validation
- Make synthetic data an explicit smoke-test mode instead of a silent fallback

### A research-rigour lesson from the first ablation

My first ablation was useful for learning the mechanics of controlled model
comparison, but it was not yet a publication-quality benchmark. The local data
subset and result tables were not versioned, and the row-wise split could place
different measurements of an identical RNA sequence in separate partitions.
I initially wrote conclusions that were more confident than the evidence
allowed.

I have since changed the pipeline to fail when real data are missing, split by
sequence group, ignore padding inside recurrent and attention layers, and
weight metrics by valid nucleotide targets. I keep the preliminary table in
the README with a clear warning because showing how I corrected the workflow
is a more honest account of my transition from wet-lab work to computational
research than silently removing the early attempt.

### Skills developed
- PyTorch: custom Datasets, masked loss, model checkpointing
- Bioinformatics: RNA structure, chemical probing data, ViennaRNA
- ML engineering: debugging training dynamics, hyperparameter tuning
- Software engineering: modular code design, Git version control

## Next Steps

I plan to explore:
1. **Graph Neural Networks** for explicitly modelling base-pairing interactions
2. **SE(3)-equivariant architectures** for physically consistent 3D predictions
3. **Attention visualisation** to understand what structural features the model learns
