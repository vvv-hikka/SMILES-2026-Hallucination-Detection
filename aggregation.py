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
import torch.nn.functional as F

LAYER_SET = [8, 18, 20]
# LAYER_SET_WITH_GEOMETRIC = [6, 12]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: int | None = None,
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
    hidden_dim = hidden_states.shape[-1]
    if real_positions.numel() == 0:
        # For each layer: response_weighted_mean, prompt_mean, delta + 3 scalars.
        return torch.zeros(
            len(layer_indices) * (hidden_dim * 3 + 3),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    if response_start_idx is None:
        response_start_idx = int(real_positions[0].item())

    response_positions = real_positions[real_positions >= int(response_start_idx)]
    if response_positions.numel() == 0:
        response_positions = real_positions

    prompt_positions = real_positions[real_positions < int(response_start_idx)]
    if prompt_positions.numel() == 0:
        prompt_positions = real_positions

    pooled_layers = []
    for layer_idx in layer_indices:
        layer = hidden_states[layer_idx]
        response_tokens = layer[response_positions]
        prompt_tokens = layer[prompt_positions]
        
        w = torch.linspace(
            1.0,
            2.0,
            steps=response_tokens.shape[0],
            device=response_tokens.device,
            dtype=response_tokens.dtype,
        ).unsqueeze(1)
        response_mean = (response_tokens * w).sum(dim=0) / w.sum()
        prompt_mean = layer[prompt_positions].mean(dim=0)
        delta = response_mean - prompt_mean
        response_std = response_tokens.std(dim=0, unbiased=False).mean()
        cosine = F.cosine_similarity(
            response_mean.unsqueeze(0), prompt_mean.unsqueeze(0), dim=1
        ).squeeze(0)
        norm_ratio = torch.norm(response_mean) / (torch.norm(prompt_mean) + 1e-6)
        pooled_layers.append(
            torch.cat(
                [
                    response_mean,
                    prompt_mean,
                    delta,
                    torch.stack([response_std, cosine, norm_ratio]),
                ],
                dim=0,
            )
        )

    return torch.cat(pooled_layers, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: int | None = None,
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
    layer_indices = LAYER_SET
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    if real_positions.numel() == 0:
        return torch.zeros(10, dtype=hidden_states.dtype, device=hidden_states.device)

    if response_start_idx is None:
        response_start_idx = int(real_positions[0].item())

    response_positions = real_positions[real_positions >= int(response_start_idx)]
    if response_positions.numel() == 0:
        response_positions = real_positions
    prompt_positions = real_positions[real_positions < int(response_start_idx)]
    if prompt_positions.numel() == 0:
        prompt_positions = real_positions

    response_means = []
    prompt_means = []
    cosines = []
    for layer_idx in layer_indices:
        layer = hidden_states[layer_idx]
        r = layer[response_positions].mean(dim=0)
        p = layer[prompt_positions].mean(dim=0)
        response_means.append(r)
        prompt_means.append(p)
        cosines.append(F.cosine_similarity(r.unsqueeze(0), p.unsqueeze(0), dim=1).squeeze(0))

    r_stack = torch.stack(response_means, dim=0)
    p_stack = torch.stack(prompt_means, dim=0)
    r_norms = torch.norm(r_stack, dim=1)
    p_norms = torch.norm(p_stack, dim=1)
    cos_stack = torch.stack(cosines, dim=0)

    if r_stack.size(0) > 1:
        r_drift = torch.norm(r_stack[1:] - r_stack[:-1], dim=1)
        p_drift = torch.norm(p_stack[1:] - p_stack[:-1], dim=1)
        drift_r_mean = r_drift.mean()
        drift_p_mean = p_drift.mean()
    else:
        zero = torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device)
        drift_r_mean = zero
        drift_p_mean = zero

    n_real = torch.tensor(float(real_positions.numel()), dtype=hidden_states.dtype, device=hidden_states.device)
    n_resp = torch.tensor(float(response_positions.numel()), dtype=hidden_states.dtype, device=hidden_states.device)
    resp_frac = n_resp / (n_real + 1e-6)

    return torch.stack(
        [
            r_norms.mean(),
            r_norms.std(unbiased=False),
            p_norms.mean(),
            p_norms.std(unbiased=False),
            cos_stack.mean(),
            cos_stack.std(unbiased=False),
            drift_r_mean,
            drift_p_mean,
            torch.log1p(n_resp),
            resp_frac,
        ]
    )


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
    agg_features = aggregate(
        hidden_states,
        attention_mask,
        response_start_idx,
    )

    if use_geometric:
        geo_features = extract_geometric_features(
            hidden_states, attention_mask, response_start_idx=response_start_idx
        )
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
