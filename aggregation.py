"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch

LAYER_SET = [0, 8, 18, 7]
# LAYER_SET_WITH_GEOMETRIC = [6, 12]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """

    layer_indices = LAYER_SET

    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    if real_positions.numel() == 0:
        return torch.zeros(len(layer_indices) * hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)

    token_positions = real_positions

    pooled_layers = []
    for layer_idx in layer_indices:
        layer = hidden_states[layer_idx]
        pooled_layers.append(layer[token_positions].mean(dim=0))

    return torch.cat(pooled_layers, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # layer_indices = LAYER_SET_WITH_GEOMETRIC

    # real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    # if real_positions.numel() == 0:
    #     return torch.zeros(4, dtype=hidden_states.dtype, device=hidden_states.device)

    # layer_means = []
    # for layer_idx in layer_indices:
    #     layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
    #     layer_means.append(layer[real_positions].mean(dim=0))

    # stacked = torch.stack(layer_means, dim=0)  # (n_layers, hidden_dim)
    # norms = torch.norm(stacked, dim=1)  # (n_layers,)
    # drift = torch.norm(stacked[1:] - stacked[:-1], dim=1) if stacked.size(0) > 1 else torch.zeros(1, dtype=hidden_states.dtype, device=hidden_states.device)
    # n_tokens = torch.tensor(float(real_positions.numel()), dtype=hidden_states.dtype, device=hidden_states.device)

    # return torch.stack(
    #     [
    #         norms.mean(),
    #         norms.std(unbiased=False),
    #         drift.mean(),
    #         torch.log1p(n_tokens),
    #     ]
    # )


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    response_start_idx: int | None = None,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    layer_indices = LAYER_SET
    agg_features = aggregate(
        hidden_states,
        attention_mask,
        layer_indices=layer_indices,
        response_start_idx=response_start_idx,
    )

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
