# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preset registry with tiered resolution.

Tier 1 (preferred): Live import from modelopt.torch.quantization.config
    - Gets the real, authoritative preset dicts
    - Requires nvidia-modelopt[torch] installed

Tier 2 (fallback): Bundled snapshot dicts
    - Hardcoded copies of the preset dicts
    - Works on Mac without GPU, in CI, in dry-run mode
    - Must be periodically synced with upstream via scripts/sync_presets.py
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

_PRESET_REGISTRY: dict[str, dict[str, Any]] | None = None
_PRESET_SOURCE: str = "unknown"


def _load_registry() -> dict[str, dict[str, Any]]:
    """Lazily load preset configs with tiered fallback."""
    global _PRESET_REGISTRY, _PRESET_SOURCE
    if _PRESET_REGISTRY is not None:
        return _PRESET_REGISTRY

    # Tier 1: Try live import from nvidia-modelopt
    try:
        from modelopt.torch.quantization.config import (
            FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
            FP8_DEFAULT_CFG,
            FP8_PER_CHANNEL_PER_TOKEN_CFG,
            INT4_AWQ_CFG,
            INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            INT8_DEFAULT_CFG,
            INT8_SMOOTHQUANT_CFG,
            INT8_WEIGHT_ONLY_CFG,
            MAMBA_MOE_FP8_AGGRESSIVE_CFG,
            MAMBA_MOE_FP8_CONSERVATIVE_CFG,
            MAMBA_MOE_NVFP4_AGGRESSIVE_CFG,
            MAMBA_MOE_NVFP4_CONSERVATIVE_CFG,
            MXFP4_DEFAULT_CFG,
            MXFP4_MLP_WEIGHT_ONLY_CFG,
            MXFP6_DEFAULT_CFG,
            MXFP8_DEFAULT_CFG,
            MXINT8_DEFAULT_CFG,
            NVFP4_AWQ_CLIP_CFG,
            NVFP4_AWQ_FULL_CFG,
            NVFP4_AWQ_LITE_CFG,
            NVFP4_DEFAULT_CFG,
            NVFP4_FP8_MHA_CONFIG,
            NVFP4_MLP_ONLY_CFG,
            NVFP4_MLP_WEIGHT_ONLY_CFG,
            NVFP4_SVDQUANT_DEFAULT_CFG,
            NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
            NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG,
            W4A8_AWQ_BETA_CFG,
            W4A8_MXFP4_FP8_CFG,
            W4A8_NVFP4_FP8_CFG,
        )

        _PRESET_REGISTRY = {
            # Core formats
            "fp8": FP8_DEFAULT_CFG,
            "fp8_pc_pt": FP8_PER_CHANNEL_PER_TOKEN_CFG,
            "fp8_pb_wo": FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
            "int8": INT8_DEFAULT_CFG,
            "int8_sq": INT8_SMOOTHQUANT_CFG,
            "int8_wo": INT8_WEIGHT_ONLY_CFG,
            "int4": INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            "int4_awq": INT4_AWQ_CFG,
            # NVFP4 family
            "nvfp4": NVFP4_DEFAULT_CFG,
            "nvfp4_awq": NVFP4_AWQ_LITE_CFG,
            "nvfp4_awq_lite": NVFP4_AWQ_LITE_CFG,
            "nvfp4_awq_clip": NVFP4_AWQ_CLIP_CFG,
            "nvfp4_awq_full": NVFP4_AWQ_FULL_CFG,
            "nvfp4_mse": NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG,
            "nvfp4_local_hessian": NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG,
            "nvfp4_fp8_mha": NVFP4_FP8_MHA_CONFIG,
            "nvfp4_svdquant": NVFP4_SVDQUANT_DEFAULT_CFG,
            "nvfp4_mlp_only": NVFP4_MLP_ONLY_CFG,
            "nvfp4_mlp_wo": NVFP4_MLP_WEIGHT_ONLY_CFG,
            # W4A8 variants
            "w4a8_awq": W4A8_AWQ_BETA_CFG,
            "w4a8_nvfp4_fp8": W4A8_NVFP4_FP8_CFG,
            "w4a8_mxfp4_fp8": W4A8_MXFP4_FP8_CFG,
            # MX formats
            "mxfp8": MXFP8_DEFAULT_CFG,
            "mxfp6": MXFP6_DEFAULT_CFG,
            "mxfp4": MXFP4_DEFAULT_CFG,
            "mxint8": MXINT8_DEFAULT_CFG,
            "mxfp4_mlp_wo": MXFP4_MLP_WEIGHT_ONLY_CFG,
            # Mamba MOE
            "mamba_moe_fp8_aggressive": MAMBA_MOE_FP8_AGGRESSIVE_CFG,
            "mamba_moe_fp8_conservative": MAMBA_MOE_FP8_CONSERVATIVE_CFG,
            "mamba_moe_nvfp4_aggressive": MAMBA_MOE_NVFP4_AGGRESSIVE_CFG,
            "mamba_moe_nvfp4_conservative": MAMBA_MOE_NVFP4_CONSERVATIVE_CFG,
        }
        _PRESET_SOURCE = "live"
        return _PRESET_REGISTRY
    except ImportError:
        pass

    # Tier 2: Bundled snapshot
    from ._bundled_presets import BUNDLED_PRESETS

    warnings.warn(
        "nvidia-modelopt not installed. Using bundled preset snapshot. "
        "Install nvidia-modelopt[torch] for live presets.",
        UserWarning,
        stacklevel=2,
    )
    _PRESET_REGISTRY = BUNDLED_PRESETS
    _PRESET_SOURCE = "bundled"
    return _PRESET_REGISTRY


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of the preset config dict."""
    registry = _load_registry()
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return copy.deepcopy(registry[name])


def get_preset_source() -> str:
    """Return 'live' or 'bundled' indicating which tier is active."""
    _load_registry()
    return _PRESET_SOURCE


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(_load_registry().keys())
