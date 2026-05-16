# SOLUTION

Tried:
- Aggregation: single-layer (tried every layer) vs selected multi-layer response-mean aggregation (tried with top-10 layers in single-layer aggregation version),
- Model: linear / MLP / MLP+dropout / SVM / kNN probes,
- Split: single split vs stratified k-fold.

## Experiments intentionally skipped in this baseline

- Multi-layer concatenation:
  - Better expressiveness but higher feature dimension and slower probe training.
- 5-fold cross-validation:
  - Better stability estimates but too expensive for quick Colab iteration.
- Deeper MLP probes:
  - Can improve performance, but increases tuning burden and runtime.
