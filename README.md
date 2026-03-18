# Attention Residuals

A PyTorch implementation of **Attention Residuals (AttnRes)**, a drop-in replacement for standard residual connections in Transformers. Based on the paper:

> **Attention Residuals** — Kimi Team (arXiv: [2603.15031](https://arxiv.org/abs/2603.15031))

---

## Overview

Standard residual connections hardcode a fixed unit-weight accumulation:

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

This causes two well-known problems:
- **PreNorm dilution** — as depth grows, sublayer contributions become increasingly diluted.
- **Uniform depth mixing** — every layer can only "see" its direct predecessor.

**AttnRes** replaces fixed accumulation with softmax attention over *all* preceding layer outputs:

$$h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i, \quad \alpha_{i \to l} = \text{softmax}_i\left(w_l^\top \, \text{RMSNorm}(v_i)\right)$$

where $w_l \in \mathbb{R}^d$ is a learned pseudo-query (one per layer, zero-initialized).

---

## Variants

### Full AttnRes
Each layer attends over **all** previous layer outputs. Memory cost: $O(L \cdot d)$ per token.

### Block AttnRes
Layers are partitioned into $N$ blocks. Attention is computed over **block-level summaries** rather than individual layer outputs. Memory cost: $O(N \cdot d)$ — with $N \approx 8$ blocks, it recovers most of the gains of Full AttnRes.

---

## Project Structure

```
.
├── implementation/
│   ├── attention_residuals.py       # FullAttnResOp and FullAttnResLayer (core operators)
│   ├── block_attention_residuals.py # Block AttnRes operators
│   └── transformer.py               # StandardTransformer, FullAttnResTransformer,
│                                    #   BlockAttnResTransformer
├── experiments/
│   └── run_experiment.py            # Training & evaluation on synthetic dataset
├── results/                         # Generated plots saved here
└── blog.md                          # Detailed technical writeup
```

---

## Quickstart

### Install dependencies

```bash
pip install torch numpy matplotlib
```

### Run the experiment

```bash
cd experiments
python run_experiment.py
```

This trains three models — Standard, Full AttnRes, Block AttnRes — on a synthetic character-level dataset and saves comparison plots to `results/training_dynamics.png`.

---

## Key Design Choices

| Choice | Why it matters |
|---|---|
| **Zero-init pseudo-queries** | At init, softmax is uniform → neutral averaging over all depths until training specializes weights |
| **RMSNorm on keys** | Equalizes source magnitudes so depth attention reflects directional alignment, not raw magnitude |
| **Separate queries for attn & MLP sublayers** | Each sublayer independently learns which depth context is most relevant |
| **Block size ≈ 8** | Coarse-grained summaries capture the dominant depth correlation patterns at a fraction of the full-AttnRes memory cost |

---

## Results (from the paper)

AttnRes consistently outperforms the standard baseline:

- **Scaling laws:** Block AttnRes matches the loss of a baseline trained with **1.25× more compute**.
- **Downstream (Kimi Linear 48B, 1.4T tokens):** GPQA-Diamond +7.5, HumanEval +3.1, Math +3.6.
- **Training dynamics:** Bounded output magnitudes across depth; more uniform gradient distribution.

---

## Implementation Notes

### Full AttnRes forward pass

```python
# v_0 = token embedding
layer_outputs = [embedding]

for block in transformer_blocks:
    h, layer_outputs = block(layer_outputs)
    # layer_outputs grows by 2 per block (attn_out + mlp_out)
```

### Block AttnRes forward pass

```python
blocks = [embedding]   # b_0 = token embedding
partial_block = embedding

for block in transformer_blocks:
    blocks, partial_block = block(blocks, partial_block)
    # At block boundary: partial_block committed to blocks[], then reset
```

---

## Citation

```bibtex
@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Kimi Team and Chen, Guangyu and Zhang, Yu and Su, Jianlin and Xu, Weixin
                   and Pan, Siyuan and Wang, Yaoyu and Wang, Yucheng and Chen, Guanduo
                   and Yin, Bohong and Chen, Yutian and Yan, Junjie and Wei, Ming and Zhang, Y.
                   and Meng, Fanqing and Hong, Chao and Xie, Xiaotong and Liu, Shaowei
                   and Lu, Enzhe and Tai, Yunpeng and Chen, Yanru and Men, Xin and Guo, Haiqing
                   and Charles, Y. and Lu, Haoyu and Sui, Lin and Zhu, Jinguo and Zhou, Zaida
                   and He, Weiran and Huang, Weixiao and Xu, Xinran and Wang, Yuzhi
                   and Lai, Guokun and Du, Yulun and Wu, Yuxin and Yang, Zhilin and Zhou, Xinyu},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2603.15031},
  primaryclass  = {cs.CL}
}
```
