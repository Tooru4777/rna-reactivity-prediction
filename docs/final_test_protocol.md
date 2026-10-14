# Final-test protocol draft

Status: DRAFT; no final-test evaluation is authorized by this document alone.
Freeze a dated, committed protocol after the convergence study determines the
model, hyperparameters, training seeds and any confirmatory contrasts.

1. Keep V17 and the convergence run as development evidence. Their entire
   1,000-sequence cohort is exposed because random/grouped protocols use it.
2. Select a new current quality-filtered test cohort using sequence-only rules,
   excluding all development sequences and qualifying similarity neighbours.
   Use the same recorded identity/coverage definition and directed search audit.
   Freeze source, cohort and assignment hashes before evaluating test targets.
3. Primary model selection uses only mean clustered CV nucleotide-weighted MAE
   across the fixed training seeds. Random/grouped results are diagnostic.
   Do not average all split protocols into the selection criterion.
4. Register the final model and its training settings, test size, test-cluster
   count, assay coverage and exact metrics before unlocking test evaluation.
   Include train-only per-assay mean and median baselines.
5. Primary metric: nucleotide-weighted clipped MAE. Secondary metrics:
   macro-sequence MAE and assay-specific MAE. Report seed variability separately
   from paired cluster-bootstrap uncertainty; do not pool repeated seed
   predictions as independent sequence observations.
6. A selected-model-only final test supports absolute generalization evidence.
   Any confirmatory real-vs-zero/shuffled contrast requires preregistering those
   models and evaluating the fixed contrasts once. Otherwise keep structural
   comparisons explicitly exploratory and CV-only.
7. Archive results, source/code hashes and test exposure history. Further
   tuning after viewing the final test requires a new independent test cohort.

Remaining decisions: convergence result, model/hyperparameters, test cohort
size and source availability, and whether confirmatory control contrasts are
worth the additional scope. These are unresolved; this draft is not frozen.
