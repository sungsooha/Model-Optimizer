"""Recipe loading for ModelOpt.

Usage:
    from modelopt.torch.recipes import load_recipe

    # Load a recipe YAML file
    result = load_recipe("path/to/recipe.yaml")

    # For quantization recipes:
    config = result["quantize_config"]  # dict for mtq.quantize()
    model = mtq.quantize(model, config, forward_loop=forward_loop)

    # For auto-quantize recipes:
    kwargs = result["auto_quantize_kwargs"]  # kwargs for mtq.auto_quantize()
    model, state = mtq.auto_quantize(model, **kwargs, data_loader=..., ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .formats import FORMAT_REGISTRY, KV_FORMAT_REGISTRY
from .presets import PRESET_REGISTRY
from .resolver import resolve_recipe
from .schema import RecipeConfig


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Load a YAML recipe and resolve it to mtq-compatible config dicts.

    Args:
        path: Path to the recipe YAML file.

    Returns:
        A dict with keys depending on the recipe type:
        - "quantize_config": config dict for mtq.quantize()
        - "auto_quantize_kwargs": kwargs dict for mtq.auto_quantize()
        - "calibration": calibration params dict (if specified)
        - "export": export params dict (if specified)
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    recipe = RecipeConfig.model_validate(raw)
    return resolve_recipe(recipe)


__all__ = [
    "load_recipe",
    "RecipeConfig",
    "FORMAT_REGISTRY",
    "KV_FORMAT_REGISTRY",
    "PRESET_REGISTRY",
]
