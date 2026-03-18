"""
Experiment: Compare Standard vs Full AttnRes vs Block AttnRes on character-level language modeling.

Uses a tiny synthetic dataset for fast reproducibility.
Tracks: training loss, hidden state magnitude growth, gradient distribution.
"""

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from implementation.transformer import (
    StandardTransformer,
    FullAttnResTransformer,
    BlockAttnResTransformer,
)


# ─────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────

def make_dataset(vocab_size: int = 64, seq_len: int = 64, n_samples: int = 1024, seed: int = 42):
    """Simple random token dataset for LM pre-training simulation."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, vocab_size, size=(n_samples, seq_len + 1))
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    return inputs, targets


def get_dataloaders(vocab_size, seq_len, n_samples, batch_size, seed=42):
    inputs, targets = make_dataset(vocab_size, seq_len, n_samples, seed)
    n_train = int(0.9 * n_samples)
    train_dataset = torch.utils.data.TensorDataset(inputs[:n_train], targets[:n_train])
    val_dataset = torch.utils.data.TensorDataset(inputs[n_train:], targets[n_train:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # [B, T, V]
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Hidden state magnitude analysis
# ─────────────────────────────────────────────

def measure_hidden_state_magnitudes(model, x: torch.Tensor) -> List[float]:
    """
    Measure the L2 norm of hidden states after each transformer block.
    Registers forward hooks to capture intermediate activations.
    """
    magnitudes = []
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                magnitudes.append(output.detach().float().norm(dim=-1).mean().item())
            elif isinstance(output, tuple):
                # BlockAttnResBlock returns (new_blocks: list, partial_block: tensor);
                # FullAttnResBlock returns (h: tensor, layer_outputs: list).
                # Find the first tensor element to capture the hidden state.
                for elem in output:
                    if isinstance(elem, torch.Tensor):
                        magnitudes.append(elem.detach().float().norm(dim=-1).mean().item())
                        break
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return magnitudes


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def main():
    # ── Experiment configuration ──────────────────────────────────────────────
    # Kept small for fast reproducibility on a laptop/colab.
    # The paper's production experiments use much larger models (48B params)
    # but the same architectural principles apply at this toy scale.
    VOCAB_SIZE = 64     # character-level vocabulary
    SEQ_LEN = 64        # context window length T
    HIDDEN_DIM = 128    # model dimension d
    NUM_LAYERS = 8      # total transformer blocks L
    NUM_HEADS = 4       # attention heads (head_dim = 128/4 = 32)
    BLOCK_SIZE = 2      # Block AttnRes: S=2 layers/block → N=4 blocks (O(4d) memory)
                        # Paper recommends N≈8 for large models; 4 is fine at this scale.
    BATCH_SIZE = 32
    N_SAMPLES = 2048
    LR = 3e-4           # AdamW learning rate
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {DEVICE}")
    print(f"Config: L={NUM_LAYERS} layers, d={HIDDEN_DIM}, N_blocks={NUM_LAYERS//BLOCK_SIZE}")

    # Data
    train_loader, val_loader = get_dataloaders(VOCAB_SIZE, SEQ_LEN, N_SAMPLES, BATCH_SIZE)

    # Three models corresponding to paper Figure 1 (a), (b), (c):
    #   (a) Standard:    h_l = h_{l-1} + f_l(h_{l-1})  (fixed unit-weight residuals)
    #   (b) Full AttnRes: h_l = sum_i alpha_{i->l} * v_i  (O(L*d) memory)
    #   (c) Block AttnRes: attend over block summaries  (O(N*d) memory)
    models = {
        'Standard':    StandardTransformer(VOCAB_SIZE, SEQ_LEN, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS),
        'Full AttnRes': FullAttnResTransformer(VOCAB_SIZE, SEQ_LEN, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS),
        'Block AttnRes': BlockAttnResTransformer(VOCAB_SIZE, SEQ_LEN, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, BLOCK_SIZE),
    }

    for name, model in models.items():
        n = count_params(model)
        print(f"{name}: {n:,} parameters")
        models[name] = model.to(DEVICE)

    optimizers = {
        name: optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
        for name, model in models.items()
    }

    # Training loop
    history: Dict[str, Dict[str, List[float]]] = {
        name: {'train': [], 'val': []} for name in models
    }

    for epoch in range(1, EPOCHS + 1):
        for name, model in models.items():
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizers[name], DEVICE)
            val_loss = eval_epoch(model, val_loader, DEVICE)
            history[name]['train'].append(train_loss)
            history[name]['val'].append(val_loss)
            dt = time.time() - t0
            # Print every 5 epochs to track convergence speed — paper reports
            # AttnRes converges faster due to richer depth-gradient pathways.
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | {name:15s} | train={train_loss:.4f} val={val_loss:.4f} | {dt:.1f}s")

    # ─────────────────────────────────────────────
    # Plot 1: Validation loss curves
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for name, hist in history.items():
        ax.plot(hist['val'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('(a) Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────
    # Plot 2: Hidden state magnitude per layer
    # ─────────────────────────────────────────────
    # Paper finding (Section 5 / Figure training_dynamics):
    # Standard PreNorm: hidden-state magnitudes grow with depth (dilution problem).
    # AttnRes: magnitudes stay bounded because h_l is a softmax-weighted average
    # of sources — the convex combination cannot exceed the max source magnitude.
    sample_x = next(iter(val_loader))[0][:4].to(DEVICE)
    ax = axes[1]
    for name, model in models.items():
        mags = measure_hidden_state_magnitudes(model, sample_x)
        ax.plot(range(1, len(mags) + 1), mags, marker='o', label=name)
    ax.set_xlabel('Transformer Block Index')
    ax.set_ylabel('Output Magnitude')
    ax.set_title('(b) Output Magnitude per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────
    # Plot 3: Gradient magnitude per layer
    # ─────────────────────────────────────────────
    # Paper finding (Section 5 / Figure training_dynamics):
    # Standard: gradient norms decay toward early layers (vanishing gradient).
    # AttnRes: each layer output v_i receives direct gradient from every later
    # layer that attends to it — creating short-circuit backward paths that
    # distribute gradients more uniformly across depth.
    ax = axes[2]
    criterion = nn.CrossEntropyLoss()
    for name, model in models.items():
        model.train()
        x_s, y_s = next(iter(train_loader))
        x_s, y_s = x_s.to(DEVICE), y_s.to(DEVICE)
        optimizer = optimizers[name]
        optimizer.zero_grad()
        logits = model(x_s)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y_s.reshape(-1))
        loss.backward()

        grad_norms = []
        for block in model.blocks:
            # Average gradient norm across all parameters in a block.
            # A flat curve across block indices = uniform gradient flow (desired).
            block_grads = [
                p.grad.norm().item()
                for p in block.parameters()
                if p.grad is not None
            ]
            if block_grads:
                grad_norms.append(np.mean(block_grads))
            else:
                grad_norms.append(0.0)
        ax.plot(range(1, len(grad_norms) + 1), grad_norms, marker='s', label=name)
        optimizer.zero_grad()

    ax.set_xlabel('Transformer Block Index')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('(c) Gradient Magnitude per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_dynamics.png', dpi=150, bbox_inches='tight')
    print("\nSaved results/training_dynamics.png")

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n=== Final Validation Loss ===")
    for name, hist in history.items():
        best = min(hist['val'])
        final = hist['val'][-1]
        print(f"  {name:15s}: final={final:.4f}, best={best:.4f}")

    # Pseudo-query norm analysis — verifies the paper's Section 3.3 claim:
    # w_l starts at zero (uniform depth mixing) and grows during training as
    # each layer learns which depth sources are most useful.
    # If all norms stay near zero, the model isn't learning depth routing.
    print("\n=== AttnRes Weight Analysis (Block AttnRes) ===")
    block_model = models['Block AttnRes']
    print("Pseudo-query norms per layer (should grow from zero during training):")
    for i, blk in enumerate(block_model.blocks):
        q_attn = blk.attn_res_query_attn.norm().item()  # w_l for attn sublayer
        q_mlp  = blk.attn_res_query_mlp.norm().item()   # w_l for MLP sublayer
        print(f"  Layer {i:2d}: attn_query_norm={q_attn:.4f}, mlp_query_norm={q_mlp:.4f}")


if __name__ == '__main__':
    main()
