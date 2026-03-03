"""Format registry: maps human-readable format names to quantizer attribute kwargs.

Each entry has separate "weight" and "activation" defaults since they sometimes differ
(e.g., int8 weights use axis=0, activations use axis=None).

Values are derived directly from the existing preset constants in
modelopt/torch/quantization/config.py.
"""

from __future__ import annotations

from typing import Any

# Format name → {"weight": {...}, "activation": {...}}
FORMAT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "fp8": {
        "weight": {"num_bits": (4, 3), "axis": None},
        "activation": {"num_bits": (4, 3), "axis": None},
    },
    "nvfp4": {
        "weight": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "activation": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
    },
    "int8": {
        "weight": {"num_bits": 8, "axis": 0},
        "activation": {"num_bits": 8, "axis": None},
    },
    "int4": {
        "weight": {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
        "activation": {"enable": False},
    },
    "mxfp8": {
        "weight": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "activation": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
    },
    "mxfp6": {
        "weight": {
            "num_bits": (3, 2),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "activation": {
            "num_bits": (3, 2),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
    },
    "mxfp4": {
        "weight": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "activation": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
    },
}

# KV cache format registry — used for kv_cache section in recipes.
# Maps format name to the quant_cfg patterns for KV cache quantizers.
KV_FORMAT_REGISTRY: dict[str, dict[str, Any]] = {
    "fp8": {
        "*[kv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "nvfp4": {
        "*[kv]_bmm_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
    "fp8_affine": {
        "*[kv]_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
            "bias": {-2: None, -4: None, "type": "static"},
        },
        "default": {"enable": False},
    },
    "nvfp4_affine": {
        "*[kv]_bmm_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
            "bias": {-2: None, -4: None, "type": "static"},
        },
        "default": {"enable": False},
    },
    "nvfp4_rotate": {
        "*q_bmm_quantizer": {"enable": False, "rotate": True},
        "*k_bmm_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
            "rotate": True,
        },
        "*v_bmm_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
}


def get_format(name: str) -> dict[str, dict[str, Any]]:
    """Look up a format by name. Raises KeyError if not found."""
    if name not in FORMAT_REGISTRY:
        available = sorted(FORMAT_REGISTRY.keys())
        raise KeyError(f"Unknown format '{name}'. Available: {available}")
    return FORMAT_REGISTRY[name]


def get_kv_format(name: str) -> dict[str, Any]:
    """Look up a KV cache format by name. Raises KeyError if not found."""
    if name not in KV_FORMAT_REGISTRY:
        available = sorted(KV_FORMAT_REGISTRY.keys())
        raise KeyError(f"Unknown KV cache format '{name}'. Available: {available}")
    return KV_FORMAT_REGISTRY[name]
