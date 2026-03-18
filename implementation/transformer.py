"""
Minimal GPT-style transformer with Full and Block Attention Residuals.

Provides:
  - StandardTransformer: baseline with standard residuals + PreNorm
  - FullAttnResTransformer: Full AttnRes replacing standard residuals
  - BlockAttnResTransformer: Block AttnRes (memory-efficient variant)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (used throughout the paper).
    Applied as PreNorm before each sublayer in the standard baseline,
    and applied to depth-attention keys k_i = v_i in all AttnRes variants.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, head_dim]
        q = q.transpose(1, 2)  # [B, H, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expand * hidden_dim, bias=False)
        self.fc2 = nn.Linear(expand * hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ─────────────────────────────────────────────
# Standard Transformer (Baseline)
# ─────────────────────────────────────────────

class StandardTransformerBlock(nn.Module):
    """PreNorm transformer block with standard residuals."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard PreNorm residual: h_l = h_{l-1} + f_{l-1}(RMSNorm(h_{l-1})).
        # Fixed unit-weight accumulation — the baseline that AttnRes replaces.
        # Problem (paper Section 2): every layer contributes equally regardless
        # of how useful its representation is for downstream computation.
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    """Baseline GPT-style transformer with standard residual connections."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        # Standard Xavier uniform init for all weight matrices.
        # No special treatment needed — standard residuals have no
        # initialization-sensitive routing parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.lm_head(x)


# ─────────────────────────────────────────────
# Full Attention Residuals Transformer
# ─────────────────────────────────────────────

class FullAttnResBlock(nn.Module):
    """
    Transformer block with Full Attention Residuals.

    Replaces h_{l-1} (standard residual source) with a softmax-weighted
    aggregation over ALL previous layer outputs.

    Key formula:
        alpha_{i->l} = softmax_i( w_l^T RMSNorm(v_i) )
        h_l = sum_i alpha_{i->l} * v_i
        output = f_l(h_l)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Pre-attention AttnRes
        self.attn_res_query_attn = nn.Parameter(torch.zeros(hidden_dim))
        self.attn_res_norm_attn = RMSNorm(hidden_dim)

        # Pre-MLP AttnRes
        self.attn_res_query_mlp = nn.Parameter(torch.zeros(hidden_dim))
        self.attn_res_norm_mlp = RMSNorm(hidden_dim)

        self.norm_attn = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm_mlp = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, dropout=dropout)

    def _attn_res(
        self,
        layer_outputs: List[torch.Tensor],
        query: torch.Tensor,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """
        Core depth-attention operator (paper Eq. 1–2):
            alpha_{i->l} = softmax_i( w_l^T * RMSNorm(v_i) )
            h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
        """
        # Stack all source vectors v_i along a new depth dimension.
        V = torch.stack(layer_outputs, dim=0)   # [S, B, T, D]
        # RMSNorm on keys: prevents high-magnitude sources from dominating.
        K = norm(V)                              # [S, B, T, D]
        # Depth-attention logits: w_l^T * RMSNorm(v_i), one scalar per (source, token).
        logits = torch.einsum('d, s b t d -> s b t', query, K)  # [S, B, T]
        # Softmax over depth dimension → probability over S source layers.
        alpha = torch.softmax(logits, dim=0)     # [S, B, T]
        # Weighted sum: h_l = sum_i alpha_i * v_i  (the new hidden state).
        h = torch.einsum('s b t, s b t d -> b t d', alpha, V)   # [B, T, D]
        return h

    def forward(
        self, layer_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            layer_outputs: [v_0, v_1, ..., v_{l-1}] (all previous outputs including embedding)
        Returns:
            (h_final, updated_layer_outputs with two new entries: attn_out, mlp_out)
        """
        # --- Attention sub-layer ---
        # h = depth-attention aggregation over [v_0, ..., v_{l-1}] (paper Eq. 1).
        # attn_res_query_attn is the pseudo-query w_l for this attention sublayer.
        h = self._attn_res(layer_outputs, self.attn_res_query_attn, self.attn_res_norm_attn)
        # Apply self-attention to the depth-aggregated hidden state.
        # v_attn = f_attn(h_l) — registered as source for all future layers.
        attn_out = self.attn(self.norm_attn(h))
        layer_outputs = layer_outputs + [attn_out]  # v_list grows by 1

        # --- MLP sub-layer ---
        # After appending attn_out, the MLP can already attend over it.
        # This means within a single block, the MLP sees the current
        # attention output as a depth source — finer-grained than per-block.
        h = self._attn_res(layer_outputs, self.attn_res_query_mlp, self.attn_res_norm_mlp)
        # v_mlp = f_mlp(h_l) — registered as source for all future layers.
        mlp_out = self.mlp(self.norm_mlp(h))
        layer_outputs = layer_outputs + [mlp_out]  # v_list grows by 1 again

        # h is the depth-aggregate used as input to the MLP (last _attn_res call).
        # Returning h (not mlp_out) keeps semantics consistent with FullAttnResLayer.
        return h, layer_outputs


class FullAttnResTransformer(nn.Module):
    """
    GPT-style transformer with Full Attention Residuals.

    Each layer selectively aggregates ALL previous layer outputs via
    learned softmax attention weights (one pseudo-query per layer).
    Memory: O(Ld) per token.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            FullAttnResBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Critical (paper Section 3.3): pseudo-queries w_l must start at zero.
            # At w_l=0 all logits are 0 → softmax is uniform (1/S) over all S sources.
            # This ensures neutral depth mixing at the start — the model learns
            # specialization gradually, without arbitrary initial routing bias.
            if 'attn_res_query' in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # v_0 = token embedding — the "zeroth source" in the paper's notation.
        # All subsequent layers can attend back to the raw token representation.
        h = self.embedding(input_ids) + self.pos_embedding(pos)  # [B, T, D]

        # layer_outputs = [v_0, v_1, ..., v_{l-1}] — grows by 2 per block
        # (one entry for attn sublayer, one for MLP sublayer).
        # Final length = 2*L + 1, memory cost = O(L*d) per token.
        layer_outputs: List[torch.Tensor] = [h]

        for block in self.blocks:
            h, layer_outputs = block(layer_outputs)

        h = self.norm_out(h)
        return self.lm_head(h)


# ─────────────────────────────────────────────
# Block Attention Residuals Transformer
# ─────────────────────────────────────────────

class BlockAttnResBlock(nn.Module):
    """
    Transformer block with Block Attention Residuals.

    Partitions layers into blocks of size S. Within a block:
      - Maintains a partial sum b_n^i of layer outputs so far
      - Attends over [b_0, ..., b_{n-1}, b_n^i]
    At block boundaries, the partial sum becomes the new block representation b_n.

    Memory: O(Nd) per token where N = L/S blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        block_size: int,
        layer_idx: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.layer_idx = layer_idx  # 0-indexed transformer block number

        # Pre-attention AttnRes (pseudo-query)
        self.attn_res_query_attn = nn.Parameter(torch.zeros(hidden_dim))
        self.attn_res_norm_attn = RMSNorm(hidden_dim)

        # Pre-MLP AttnRes (pseudo-query)
        self.attn_res_query_mlp = nn.Parameter(torch.zeros(hidden_dim))
        self.attn_res_norm_mlp = RMSNorm(hidden_dim)

        self.norm_attn = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm_mlp = RMSNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, dropout=dropout)

    def _block_attn_res(
        self,
        blocks: List[torch.Tensor],
        partial: torch.Tensor,
        query: torch.Tensor,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """
        Block-level depth attention (paper Section 4).
        Sources: completed block summaries b_0..b_{n-1} + current partial b_n^i.
            alpha_i = softmax_i( w_l^T * RMSNorm(b_i) )
            h = sum_i alpha_i * b_i
        """
        # V = [b_0, ..., b_{n-1}, b_n^i]: all available block-level representations.
        V = torch.stack(blocks + [partial], dim=0)  # [N+1, B, T, D]
        # Normalize to equalize block summary magnitudes (same rationale as Full AttnRes).
        K = norm(V)                                  # [N+1, B, T, D]
        # Depth-attention logits over N+1 block sources.
        logits = torch.einsum('d, n b t d -> n b t', query, K)  # [N+1, B, T]
        # Softmax over block dimension → attention weights over depth blocks.
        alpha = torch.softmax(logits, dim=0)                     # [N+1, B, T]
        # h = weighted sum of block representations.
        h = torch.einsum('n b t, n b t d -> b t d', alpha, V)   # [B, T, D]
        return h

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            blocks: completed block reps [b_0, ..., b_{n-1}], each [B, T, D]
            partial_block: intra-block accumulated sum b_n^i [B, T, D]
        Returns:
            (new_blocks, new_partial_block)
        """
        # Depth attention over [b_0,...,b_{n-1}, b_n^i] using the pre-attn pseudo-query.
        # b_n^i is the partial sum accumulated so far in the current block.
        h = self._block_attn_res(blocks, partial_block, self.attn_res_query_attn, self.attn_res_norm_attn)

        # Block boundary (paper Section 4): commit the INCOMING partial sum b_n^i as
        # the block summary b_n BEFORE accumulating this layer's output.
        # This way b_n captures all outputs up to (not including) the current layer,
        # preserving the causal order of block summaries.
        new_blocks = blocks
        if (self.layer_idx + 1) % self.block_size == 0:
            # b_n = partial_block (the completed intra-block sum).
            new_blocks = blocks + [partial_block]
            partial_block = None  # reset: next accumulation starts fresh

        # --- Attention sub-layer ---
        # f_attn(h): transform depth-aggregated state through self-attention.
        attn_out = self.attn(self.norm_attn(h))
        # Intra-block accumulation: b_n^{i+1} = b_n^i + attn_out.
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # Second depth attention: now includes attn_out in partial and (possibly) new block.
        # Uses the pre-MLP pseudo-query — separate learned routing for the MLP sublayer.
        h = self._block_attn_res(new_blocks, partial_block, self.attn_res_query_mlp, self.attn_res_norm_mlp)

        # --- MLP sub-layer ---
        # f_mlp(h): transform depth-aggregated state through the feed-forward network.
        mlp_out = self.mlp(self.norm_mlp(h))
        # Intra-block accumulation: b_n^{i+2} = b_n^{i+1} + mlp_out.
        partial_block = partial_block + mlp_out

        return new_blocks, partial_block


class BlockAttnResTransformer(nn.Module):
    """
    GPT-style transformer with Block Attention Residuals.

    Partitions L layers into N = L//block_size blocks.
    Memory per token: O(N*d) instead of O(L*d) for Full AttnRes.

    Recommended: N ~= 8 recovers most gains from Full AttnRes.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        block_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            BlockAttnResBlock(hidden_dim, num_heads, block_size, i, dropout)
            for i in range(num_layers)
        ])
        self.norm_out = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.seq_len = seq_len
        self.block_size = block_size
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Same zero-init rationale as Full AttnRes (paper Section 3.3):
            # ensures uniform depth mixing at the start of training.
            if 'attn_res_query' in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # h = token embedding; serves as both b_0 (first block summary) and
        # the initial partial_block b_0^0 (paper Section 4).
        h = self.embedding(input_ids) + self.pos_embedding(pos)  # [B, T, D]

        # blocks = [b_0 = embedding] — the always-present "zeroth block" summary.
        # partial_block = h — intra-block accumulator starts at the embedding.
        # Memory cost: O(N*d) per token where N grows by 1 every block_size layers.
        blocks: List[torch.Tensor] = [h]
        partial_block = h

        for block in self.blocks:
            blocks, partial_block = block(blocks, partial_block)

        # partial_block holds the final accumulated hidden state of the last block.
        h = self.norm_out(partial_block)
        return self.lm_head(h)
