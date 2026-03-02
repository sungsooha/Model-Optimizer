"""Pydantic schema models for recipe YAML validation.

These models define the structure of recipe YAML files. The resolver
(resolver.py) translates validated schema objects into the config dicts
that mtq.quantize() and mtq.auto_quantize() accept.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator


class CalibrationConfig(BaseModel):
    """Calibration data configuration."""

    dataset: str | list[str] = "cnn_dailymail"
    num_samples: int | list[int] = 512
    max_sequence_length: int = 512
    batch_size: int = 1


class KVCacheConfig(BaseModel):
    """KV cache quantization configuration."""

    format: str  # fp8, nvfp4


class AlgorithmConfig(BaseModel):
    """Quantization algorithm configuration.

    Extra fields (e.g., fp8_scale_sweep) are passed through to the algorithm dict.
    """

    method: str  # max, smoothquant, awq_lite, awq_clip, awq_full, mse, etc.
    alpha_step: float | None = None
    max_co_batch_size: int | None = None

    model_config = {"extra": "allow"}


class QuantizerSpec(BaseModel):
    """Specifies quantization for weights or activations."""

    format: str | None = None  # human-readable: fp8, nvfp4, int4, int8
    num_bits: int | list[int] | None = None  # expert-mode escape hatch
    axis: int | None = None
    block_sizes: dict[str, Any] | None = None
    enable: bool = True
    stages: list[QuantizerSpec] | None = None  # for multi-stage (W4A8)


class OverrideEntry(BaseModel):
    """Per-layer or per-module-class override."""

    pattern: str | None = None  # glob: "*lm_head*"
    module_class: str | None = None  # class: "nn.LayerNorm"
    enable: bool | None = None
    format: str | None = None
    weights: QuantizerSpec | None = None
    activations: QuantizerSpec | None = None
    num_bits: int | list[int] | None = None
    axis: int | None = None


class QuantizationSection(BaseModel):
    """Quantization technique configuration."""

    preset: str | None = None
    weights: QuantizerSpec | None = None
    activations: QuantizerSpec | None = None
    algorithm: AlgorithmConfig | str | None = None
    kv_cache: KVCacheConfig | None = None
    calibration: CalibrationConfig | None = None
    overrides: list[OverrideEntry] = []
    disabled_patterns: list[str] = []

    @model_validator(mode="after")
    def validate_preset_or_custom(self):
        if self.preset and (self.weights or self.activations):
            raise ValueError(
                "Cannot specify both 'preset' and 'weights'/'activations'. "
                "Use preset with overrides, or specify weights/activations from scratch."
            )
        return self


class AutoQuantizeFormatEntry(BaseModel):
    """A candidate format for auto-quantize search."""

    preset: str  # e.g., "nvfp4_awq", "fp8"


class AutoQuantizeSection(BaseModel):
    """Auto-quantize configuration for per-layer format search."""

    effective_bits: float
    formats: list[AutoQuantizeFormatEntry]
    method: str = "gradient"
    num_calib_steps: int = 512
    num_score_steps: int = 128
    disabled_patterns: list[str] = []
    kv_cache: KVCacheConfig | None = None
    calibration: CalibrationConfig | None = None


class ExportConfig(BaseModel):
    """Export configuration."""

    format: Literal["hf", "tensorrt_llm"] = "hf"
    output_dir: str = "./output"
    tensor_parallel: int = 1
    pipeline_parallel: int = 1


class ModelConfig(BaseModel):
    """Model specification."""

    path: str
    trust_remote_code: bool = False
    attn_implementation: str | None = None


class RecipeMetadata(BaseModel):
    """Optional recipe metadata."""

    name: str | None = None
    description: str | None = None
    author: str | None = None
    tags: list[str] = []


class RecipeConfig(BaseModel):
    """Top-level recipe schema."""

    version: str = "1.0"
    metadata: RecipeMetadata | None = None
    model: ModelConfig | None = None
    quantization: QuantizationSection | None = None
    auto_quantize: AutoQuantizeSection | None = None
    export: ExportConfig | None = None

    @model_validator(mode="after")
    def validate_exclusive_sections(self):
        if self.quantization and self.auto_quantize:
            raise ValueError(
                "'quantization' and 'auto_quantize' are mutually exclusive. "
                "Use one or the other."
            )
        return self
