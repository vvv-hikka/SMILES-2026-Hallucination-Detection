# SOLUTION

## Reproducibility instructions

Exact commands:

```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

Expected outputs:
- `results.json`
- `predictions.csv`

Required environment:
- Python with dependencies from `requirements.txt`
- GPU runtime recommended (Colab CUDA was used for experiments)

Important implementation details for reproducibility:
- The same pre-trained model is used (`Qwen/Qwen2.5-0.5B`).
- Response boundary indices are computed with the same tokenizer settings used in extraction (`add_special_tokens=True`, truncation, same `MAX_LENGTH`).

## Solution description

Final approach:
- Build features via contrastive aggregation over two selected layers (`[18, 20]`).
- Train a regularized linear classifier in PCA space (`pca=64`, `C=0.03`).
- Tune classification threshold on the validation split for accuracy.

Why these choices:
- Logistic regression + PCA was significantly more stable than higher-capacity probes.
- Multi-layer contrastive features outperformed single-layer baselines in the final selection.

## Experiments and failed attempts

Tried but not in final solution:
- Single-layer aggregation sweep (many individual layers):
  - Some layers looked good alone, but did not reliably translate into better multi-layer combinations.
- Larger multi-layer sets (for example, adding early layers or 3-4 layer concatenations):
  - Often increased feature dimension and overfitting without consistent accuracy gains.
- Probe alternatives: linear/MLP/MLP+dropout/SVM/kNN and multiple optimizer settings:
  - Either unstable across folds or inferior on mean test accuracy.
- Geometric feature augmentation and weighted pooling variants:
  - Added complexity but regressed final accuracy, so current pipeline does not include those.

Possible next direction:
- Exploring layer-set construction using low inter-layer dependence criteria (e.g., selecting layers with less correlated representations) might have improved performance at current stage, since strong single-layer performance does not guarantee that their combination improves downstream results.
