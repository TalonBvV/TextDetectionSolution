"""
PatchFormer: Lightweight Semantic Segmentation Architecture

A high-resolution semantic segmentation model optimized for text detection
and CPU/TFLite deployment.

v2.0: Enhanced architecture for competitive text detection benchmarks.
"""

from .config import PatchFormerConfig, get_config
from .model import PatchFormer, PatchFormerV2
from .losses import PatchFormerLoss, PatchFormerV2Loss

__version__ = "2.1.0"
__all__ = [
    # Models
    "PatchFormer",
    "PatchFormerV2",
    # Config
    "PatchFormerConfig", 
    "get_config",
    # Losses
    "PatchFormerLoss",
    "PatchFormerV2Loss",
]
