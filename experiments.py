"""experiments.py — Grid search for layer sets and PCA dimensions.

Runs a controlled ablation using the same split protocol for every config and
reports:
  - mean test AUROC
  - std test AUROC across folds
  - mean train AUROC
  - train-test AUROC gap (overfitting signal)
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import aggregation as agg
from evaluate import run_evaluation
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

DATA_FILE = "./data/dataset.csv"
BATCH_SIZE = 8
USE_GEOMETRIC = False


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_layer_set(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _response_start_indices(
    prompts: Sequence[str],
    responses: Sequence[str],
    tokenizer,
    max_length: int,
) -> list[int]:
    starts: list[int] = []
    for prompt, response in zip(prompts, responses):
        full_text = f"{prompt}{response}"
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        start_idx = min(len(prompt_tokens), max(len(full_tokens) - 1, 0))
        starts.append(start_idx)
    return starts


def _extract_features(
    texts: Sequence[str],
    response_starts: Sequence[int],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    use_geometric: bool,
) -> np.ndarray:
    feats: list[torch.Tensor] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Extracting features",
        unit="batch",
    ):
        batch_texts = texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            global_idx = start + i
            feat = agg.aggregation_and_feature_extraction(
                hidden_states=hidden[i],
                attention_mask=mask[i],
                use_geometric=use_geometric,
                response_start_idx=int(response_starts[global_idx]),
            )
            feats.append(feat.cpu())
    return np.vstack([f.numpy() for f in feats])


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _std(values: list[float]) -> float:
    return float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer-sets",
        nargs="+",
        default=["18,20", "16,20", "8,18,20", "0,8,18,20"],
        help='Space-separated list, each like "18,20"',
    )
    parser.add_argument(
        "--pca-dims",
        nargs="+",
        type=int,
        default=[64, 96, 128, 192],
    )
    parser.add_argument("--output-file", default="experiment_results.json")
    args = parser.parse_args()

    layer_sets = [_parse_layer_set(s) for s in args.layer_sets]
    pca_dims = [int(x) for x in args.pca_dims]

    dev = _device()
    print(f"Device: {dev}")

    df = pd.read_csv(DATA_FILE)
    prompts = [str(x) for x in df["prompt"].tolist()]
    responses = [str(x) for x in df["response"].tolist()]
    texts = [f"{p}{r}" for p, r in zip(prompts, responses)]
    y = np.array([int(float(v)) for v in df["label"]], dtype=int)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(dev)

    response_starts = _response_start_indices(prompts, responses, tokenizer, MAX_LENGTH)
    splits = split_data(y, df)
    print(f"Using {len(splits)} folds from split_data().")

    results: list[dict] = []
    started = time.time()

    for layer_set in layer_sets:
        agg.LAYER_SET = list(layer_set)
        print(f"\n=== Layer set: {layer_set} ===")
        X = _extract_features(
            texts=texts,
            response_starts=response_starts,
            model=model,
            tokenizer=tokenizer,
            device=dev,
            batch_size=BATCH_SIZE,
            use_geometric=USE_GEOMETRIC,
        )
        print(f"Feature matrix: {X.shape}")

        for pca_dim in pca_dims:
            print(f"  -> PCA dim {pca_dim}")

            class ProbeWithConfig(HallucinationProbe):
                def __init__(self) -> None:
                    super().__init__(pca_components=pca_dim)

            fold_results = run_evaluation(splits, X, y, ProbeWithConfig)
            train_aurocs = [r["train_auroc"] for r in fold_results]
            test_aurocs = [r["test_auroc"] for r in fold_results]

            mean_train = _mean(train_aurocs)
            mean_test = _mean(test_aurocs)
            result = {
                "layer_set": list(layer_set),
                "pca_dim": pca_dim,
                "feature_dim": int(X.shape[1]),
                "n_folds": len(fold_results),
                "mean_train_auroc": mean_train,
                "mean_test_auroc": mean_test,
                "std_test_auroc": _std(test_aurocs),
                "train_test_auroc_gap": mean_train - mean_test,
                "fold_test_aurocs": test_aurocs,
            }
            results.append(result)

    ranked = sorted(results, key=lambda r: r["mean_test_auroc"], reverse=True)

    print("\n" + "=" * 110)
    print("Ranked experiments (higher mean_test_auroc is better)")
    print("=" * 110)
    print(
        f"{'#':>2}  {'layers':<16} {'pca':>5} {'feat_dim':>8} "
        f"{'mean_test':>10} {'std_test':>9} {'mean_train':>10} {'gap':>8}"
    )
    print("-" * 110)
    for i, r in enumerate(ranked, start=1):
        layers = ",".join(str(x) for x in r["layer_set"])
        print(
            f"{i:>2}  {layers:<16} {r['pca_dim']:>5} {r['feature_dim']:>8} "
            f"{r['mean_test_auroc'] * 100:>9.2f}% {r['std_test_auroc'] * 100:>8.2f}% "
            f"{r['mean_train_auroc'] * 100:>9.2f}% {r['train_test_auroc_gap'] * 100:>7.2f}%"
        )
    print("-" * 110)
    print(f"Total runtime: {time.time() - started:.1f} s")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "layer_sets": layer_sets,
                    "pca_dims": pca_dims,
                    "n_folds": len(splits),
                },
                "results": ranked,
            },
            f,
            indent=2,
        )
    print(f"Saved experiment results to '{args.output_file}'")


if __name__ == "__main__":
    main()

