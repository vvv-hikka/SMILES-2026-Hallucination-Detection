# SOLUTION

Tried:
- Aggregation: single-layer (tried every layer) vs selected multi-layer response-mean aggregation (tried with top-10 layers in single-layer aggregation version),
- Model: linear / MLP / MLP+dropout / SVM / kNN probes,
- Split: single split vs stratified k-fold.

Final choice (minimal, stable):
- `splitting.py`: stratified 5-fold split for more reliable validation.
- `probe.py`: keep MLP but add dropout + AdamW + slightly longer training.
- `aggregation.py`: response-mean pooling with fixed selected layers

Why:
- multi-layer features helped some models but were unstable across splits,
- k-fold gave more trustworthy model selection than a single split,
- regularized MLP was the best accuracy/stability trade-off under minimal code changes.
