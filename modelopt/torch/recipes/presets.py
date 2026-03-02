"""Preset registry: maps preset names to existing config dicts from config.py.

This avoids any duplication — we import and reference the same dicts that
mtq.quantize() uses today. get_preset() returns deep copies to avoid mutation.

Imports from modelopt.torch.quantization.config are lazy to avoid requiring
torch at import time (useful for schema validation and demo scripts).
"""

from __future__ import annotations

import copy
from typing import Any

_PRESET_REGISTRY: dict[str, dict[str, Any]] | None = None


def _load_registry() -> dict[str, dict[str, Any]]:
    """Lazily load preset configs from modelopt.torch.quantization.config."""
    global _PRESET_REGISTRY
    if _PRESET_REGISTRY is not None:
        return _PRESET_REGISTRY

    from modelopt.torch.quantization.config import (
        FP8_DEFAULT_CFG,
        FP8_PER_CHANNEL_PER_TOKEN_CFG,
        INT4_AWQ_CFG,
        INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        INT8_DEFAULT_CFG,
        INT8_SMOOTHQUANT_CFG,
        INT8_WEIGHT_ONLY_CFG,
        MXFP4_DEFAULT_CFG,
        MXFP6_DEFAULT_CFG,
        MXFP8_DEFAULT_CFG,
        NVFP4_AWQ_CLIP_CFG,
        NVFP4_AWQ_FULL_CFG,
        NVFP4_AWQ_LITE_CFG,
        NVFP4_DEFAULT_CFG,
        W4A8_AWQ_BETA_CFG,
    )

    _PRESET_REGISTRY = {
        "fp8": FP8_DEFAULT_CFG,
        "fp8_pc_pt": FP8_PER_CHANNEL_PER_TOKEN_CFG,
        "int8": INT8_DEFAULT_CFG,
        "int8_sq": INT8_SMOOTHQUANT_CFG,
        "int8_wo": INT8_WEIGHT_ONLY_CFG,
        "int4": INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        "int4_awq": INT4_AWQ_CFG,
        "nvfp4": NVFP4_DEFAULT_CFG,
        "nvfp4_awq": NVFP4_AWQ_LITE_CFG,
        "nvfp4_awq_lite": NVFP4_AWQ_LITE_CFG,
        "nvfp4_awq_clip": NVFP4_AWQ_CLIP_CFG,
        "nvfp4_awq_full": NVFP4_AWQ_FULL_CFG,
        "w4a8_awq": W4A8_AWQ_BETA_CFG,
        "mxfp8": MXFP8_DEFAULT_CFG,
        "mxfp6": MXFP6_DEFAULT_CFG,
        "mxfp4": MXFP4_DEFAULT_CFG,
    }
    return _PRESET_REGISTRY


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of the preset config dict."""
    registry = _load_registry()
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return copy.deepcopy(registry[name])


# For read-only access to the registry (lazy)
PRESET_REGISTRY = property(lambda self: _load_registry())
