"""
Block Attention Residuals (Block AttnRes) implementation.

Paper: "Attention Residuals" (arXiv:2603.15031) — Kimi Team, Section 4

Motivation: Full AttnRes requires storing ALL L layer outputs → O(L*d) memory.
Block AttnRes reduces this to O(N*d) by attending over block-level summaries.

Algorithm (paper Section 4):
  - Partition L layers into N blocks of S = L/N layers each.
  - Within a block, accumulate sublayer outputs into a running partial sum:
        b_n^i = b_n^{i-1} + f_{n,i}(h_{n,i})   (intra-block standard residual)
  - At each sublayer, attend over:
        [b_0, b_1, ..., b_{n-1}, b_n^i]          (inter-block attention)
    where b_k (k < n) = completed block summary (b_k^S), b_0 = token embedding.
  - At block boundary: commit b_n^S → blocks list, reset partial sum.

Memory: O(N*d) per token (stores only N block summaries, not all L outputs).
Paper reports N ≈ 8 blocks recovers most gains of Full AttnRes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


def block_attn_res(
    blocks: List[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """
    Inter-block attention: attend over block reps + partial sum.

    Args:
        blocks: list of N completed block representations, each [B, T, D]
                blocks[0] = b_0 = token embedding h_1
        partial_block: intra-block partial sum b_n^i, shape [B, T, D]
        proj: linear layer mapping [D] -> [D] for pseudo-query projection
        norm: RMSNorm applied to keys

    Returns:
        h: new hidden state [B, T, D]
    """
    # V = [b_0, ..., b_{n-1}, b_n^i]: N completed block summaries + current partial.
    # This is the full set of depth sources available to the current sublayer.
    V = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
    N1, B, T, D = V.shape

    # K = RMSNorm(v_i) — normalize block summaries to equalize their magnitudes
    # before computing pseudo-query dot products (paper Section 4).
    K = norm(V)  # [N+1, B, T, D]

    # logits_{i->l} = w_l^T * RMSNorm(b_i)   (Block AttnRes version of paper Eq. 2).
    # proj.weight has shape [1, D]; squeeze() gives [D] — the pseudo-query vector.
    logits = torch.einsum('n b t d, d -> n b t', K, proj.weight.squeeze())  # [N+1, B, T]

    # alpha = softmax over N+1 depth sources (block summaries + current partial).
    alpha = torch.softmax(logits, dim=0)  # [N+1, B, T]

    # h = sum_i alpha_i * b_i — weighted combination of block-level representations.
    h = torch.einsum('n b t, n b t d -> b t d', alpha, V)  # [B, T, D]
    return h


class BlockAttnResLayer(nn.Module):
    """
    Single transformer sub-layer with Block Attention Residuals.

    Each layer tracks:
      - blocks: completed block summaries [b_0, ..., b_{n-1}]
      - partial_block: running partial sum b_n^i within current block
    """

    def __init__(
        self,
        hidden_dim: int,
        sublayer: nn.Module,
        block_size: int,
        layer_number: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sublayer = sublayer
        self.block_size = block_size
        self.layer_number = layer_number  # 1-indexed layer index

        # Per-layer pseudo-query (D->1 projection, only weight used as query vector)
        self.attn_res_proj = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.attn_res_proj.weight)
        self.attn_res_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        *sublayer_args,
        **sublayer_kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            blocks: completed block representations [b_0, ..., b_{n-1}]
            partial_block: current intra-block partial sum b_n^i  [B,T,D]
        Returns:
            (h_l, updated_blocks, updated_partial_block)
        """
        # Inter-block attention: h = sum_i alpha_i * b_i  (paper Section 4).
        # Attends over completed block summaries + current intra-block partial sum.
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # Apply the sublayer f_l to the depth-aggregated hidden state.
        f_out = self.sublayer(h, *sublayer_args, **sublayer_kwargs)

        # Intra-block accumulation: b_n^{i+1} = b_n^i + f_l(h_l).
        # This is a standard residual within the block, not depth attention.
        new_partial = partial_block + f_out

        # Block boundary check (paper Section 4): fires when this sublayer
        # is the last one in the current block (layer_number % block_size == 0).
        new_blocks = blocks
        new_partial_out = new_partial
        if self.layer_number % self.block_size == 0:
            # Commit the completed partial sum b_n^S as the block summary b_n.
            new_blocks = blocks + [new_partial]
            # Reset partial sum to zero for the next block.
            new_partial_out = torch.zeros_like(partial_block)

        return h, new_blocks, new_partial_out


class BlockAttnResTransformerLayer(nn.Module):
    """
    Full transformer layer (Attention + MLP) with Block AttnRes.
    Applies AttnRes before both Attention and MLP sublayers.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_sublayer: nn.Module,
        mlp_sublayer: nn.Module,
        block_size: int,
        layer_number: int,  # 1-indexed, counting transformer blocks (ATTN+MLP = 1 block)
        norm_attn: nn.Module,
        norm_mlp: nn.Module,
    ):
        super().__init__()
        self.attn = attn_sublayer
        self.mlp = mlp_sublayer
        self.block_size = block_size
        self.layer_number = layer_number
        self.norm_attn = norm_attn
        self.norm_mlp = norm_mlp

        # Two AttnRes ops: one before attention, one before MLP
        self.attn_res_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_res_norm = RMSNorm(hidden_dim)
        self.mlp_res_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.mlp_res_norm = RMSNorm(hidden_dim)

        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)

    def _attend(self, blocks, partial, proj, norm):
        """Apply block attention residuals with given proj/norm."""
        V = torch.stack(blocks + [partial], dim=0)
        K = norm(V)
        logits = torch.einsum('n b t d, o d -> n b t', K, proj.weight)
        alpha = torch.softmax(logits, dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', alpha, V)
        return h

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            blocks: list of completed block reps [b_0,...,b_{n-1}], each [B,T,D]
            partial_block: current partial sum b_n^i [B,T,D]; None means zeros
            hidden_states: NOT used as residual source (AttnRes replaces it)

        Returns:
            (h_out, new_blocks, new_partial)
        """
        B, T, D = hidden_states.shape
        if partial_block is None:
            partial_block = hidden_states  # first layer: partial = embedding

        # --- Attention sub-layer ---
        h_attn = self._attend(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
        attn_out = self.attn(self.norm_attn(h_attn))
        partial_block = partial_block + attn_out

        # At block boundary after attention: reset if needed
        # (paper applies block boundary check per transformer block, not sublayer)

        # --- MLP sub-layer ---
        h_mlp = self._attend(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.mlp(self.norm_mlp(h_mlp))
        partial_block = partial_block + mlp_out

        # Check block boundary (after full transformer block = attn + mlp)
        new_blocks = blocks
        new_partial = partial_block
        if self.layer_number % self.block_size == 0:
            new_blocks = blocks + [partial_block]
            new_partial = None  # will be re-initialized at next layer

        return h_mlp + mlp_out, new_blocks, new_partial
