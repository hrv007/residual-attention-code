"""
Attention Residuals (AttnRes) - Implementation Package.

Implements Full AttnRes and Block AttnRes from:
"Attention Residuals" (arXiv:2603.15031), Kimi Team.
"""

from .attention_residuals import FullAttnResOp, FullAttnResLayer, RMSNorm
from .transformer import (
    StandardTransformer,
    FullAttnResTransformer,
    BlockAttnResTransformer,
)

__all__ = [
    "RMSNorm",
    "FullAttnResOp",
    "FullAttnResLayer",
    "StandardTransformer",
    "FullAttnResTransformer",
    "BlockAttnResTransformer",
]
