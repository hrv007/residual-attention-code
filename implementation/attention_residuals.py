"""
Full Attention Residuals (Full AttnRes) implementation.

Paper: "Attention Residuals" (arXiv:2603.15031) — Kimi Team

Replaces the standard residual connection:
    h_l = h_{l-1} + f_{l-1}(h_{l-1})             [standard PreNorm]

with learned depth-wise attention (paper Eq. 1):
    h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i

where the attention weights are (paper Eq. 2):
    alpha_{i->l} = softmax_i( w_l^T * RMSNorm(v_i) )

Source vector definitions (paper Section 3.1):
    v_0 = h_1   (token embedding — "zeroth" layer output)
    v_i = f_i(h_i)  for i >= 1  (sublayer transformation output)

Pseudo-query w_l ∈ R^d: one learned vector per sublayer.
Zero-initialized so softmax starts uniform → neutral depth mixing at init.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Applied to the source vectors v_i before computing depth-attention logits
    (paper Section 3.2). Equalizes key magnitudes so the pseudo-query dot
    product measures directional alignment, not raw vector magnitude.
    Without this, high-magnitude layers would dominate attention regardless
    of semantic relevance.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


class FullAttnResOp(nn.Module):
    """
    Full Attention Residuals operator.

    For each layer l, computes:
        alpha_{i->l} = softmax_i( w_l^T RMSNorm(v_i) )
        h_l = sum_i alpha_{i->l} * v_i

    where w_l is a learned d-dimensional pseudo-query per layer.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # w_l ∈ R^d — the learned pseudo-query for this sublayer (paper Eq. 2).
        # Zero-init: when w_l=0 all logits are 0, softmax gives 1/S for all S
        # sources → perfect uniform average at the start of training.
        # This avoids arbitrary depth biases before any layer has learned
        # meaningful representations (paper Section 3.3 initialization).
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_dim))
        # RMSNorm applied to keys k_i = v_i — normalizes magnitudes across
        # source layers so attention is content-driven, not magnitude-driven.
        self.norm = RMSNorm(hidden_dim)

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_outputs: list of tensors [v_0, v_1, ..., v_{l-1}],
                           each of shape [B, T, D]
        Returns:
            h_l: tensor of shape [B, T, D] — new hidden state for layer l
        """
        # V = [v_0, v_1, ..., v_{l-1}] stacked along dim=0 (the depth axis).
        # V[i] is the i-th source vector v_i with shape [B, T, D].
        V = torch.stack(layer_outputs, dim=0)  # [S, B, T, D]  (S = number of sources)
        S, B, T, D = V.shape

        # K = RMSNorm(v_i) — the "keys" for depth attention (paper Eq. 2).
        # Normalization ensures logits reflect direction, not magnitude.
        K = self.norm(V)  # [S, B, T, D]

        # logits_{i->l} = w_l^T * RMSNorm(v_i)  (paper Eq. 2, numerator exponent).
        # einsum: for each source s, batch b, token t → dot product over d.
        # Each token independently computes its own depth-attention logits.
        logits = torch.einsum('d, s b t d -> s b t', self.pseudo_query, K)  # [S, B, T]

        # alpha_{i->l} = softmax over source dimension (paper Eq. 2).
        # Produces a probability distribution over S preceding layer outputs.
        alpha = torch.softmax(logits, dim=0)  # [S, B, T]

        # h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i  (paper Eq. 1).
        # Each token's new hidden state is a learned convex combination of
        # all prior representations — replacing the fixed h_{l-1} source.
        h = torch.einsum('s b t, s b t d -> b t d', alpha, V)  # [B, T, D]
        return h


class FullAttnResLayer(nn.Module):
    """
    A single transformer sub-layer (Attention or MLP) wrapped with
    Full Attention Residuals instead of standard residuals.
    """

    def __init__(self, hidden_dim: int, sublayer: nn.Module):
        """
        Args:
            hidden_dim: model hidden dimension d
            sublayer: the inner transformation f_l (e.g. self-attention or MLP)
        """
        super().__init__()
        self.sublayer = sublayer
        self.attn_res = FullAttnResOp(hidden_dim)

    def forward(
        self,
        layer_outputs: List[torch.Tensor],
        *sublayer_args,
        **sublayer_kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            layer_outputs: all previous v_i values (including token embedding v_0)
        Returns:
            (h_l, updated_layer_outputs)
        """
        # Step 1: compute h_l via depth attention over [v_0, ..., v_{l-1}] (Eq. 1).
        h = self.attn_res(layer_outputs)

        # Step 2: apply the sublayer f_l to the aggregated hidden state h_l.
        # This corresponds to v_l = f_l(h_l) in the paper's notation.
        f_out = self.sublayer(h, *sublayer_args, **sublayer_kwargs)

        # Step 3: register v_l = f_out in the growing list for future layers.
        # layer_outputs grows by 1 per sublayer call → O(L*d) total memory.
        updated = layer_outputs + [f_out]
        return h, updated
