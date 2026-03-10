# Recipe Examples

YAML-based recipes for model optimization. Each recipe defines what optimization to apply — presets, algorithms, calibration, and multi-technique pipelines.

## Quick Start

```bash
# Validate a recipe
python -m modelopt.torch.recipes.cli validate examples/recipes/ptq/ptq_fp8.yaml

# Show execution plan (what ModelOpt APIs will be called)
python -m modelopt.torch.recipes.cli plan examples/recipes/ptq/ptq_fp8.yaml --verbose

# Dry-run an experiment (sweep of models × recipes × clusters)
python -m modelopt.torch.recipes.cli dry-run --config examples/recipes/experiments/sweep_demo.yaml
```

## Recipes

### [`ptq/`](ptq/) — Post-Training Quantization

| Recipe | What It Does |
|--------|-------------|
| `ptq_fp8.yaml` | FP8 quantization — simplest recipe, one-line preset |
| `ptq_nvfp4_awq.yaml` | NVFP4 with AWQ algorithm + FP8 KV cache |
| `ptq_nvfp4_local_hessian.yaml` | NVFP4 with local Hessian calibration (scale setting) |
| `ptq_nvfp4_local_hessian_overrides.yaml` | Local Hessian + per-layer overrides |
| `ptq_nvfp4_skip_first_last.yaml` | NVFP4 with first/last layers at full precision |
| `custom_w4a8_overrides.yaml` | Fully custom INT4 weights + FP8 activations (no preset) |

### [`qat/`](qat/) — Quantization-Aware Training

| Recipe | What It Does |
|--------|-------------|
| `qat_int8.yaml` | INT8 QAT — one field (`mode: qat`) changes PTQ to QAT |

### [`multi_technique/`](multi_technique/) — Multi-Technique Pipelines

| Recipe | What It Does |
|--------|-------------|
| `sparse_quantize.yaml` | 2:4 sparsity (SparseGPT) → FP8 quantization |
| `distill_fp8.yaml` | FP8 quantization → knowledge distillation from 70B teacher |
| `prune_gradnas_fp8.yaml` | GradNAS pruning (50% FLOPs) → FP8 quantization |

### [`auto/`](auto/) — Auto-Quantize

| Recipe | What It Does |
|--------|-------------|
| `auto_quantize.yaml` | Per-layer format search within bit budget |

### [`experiments/`](experiments/) — Experiment Configs

Experiment configs define a sweep (models × recipes × clusters) with evaluation settings.

| Config | What It Does |
|--------|-------------|
| `sweep_demo.yaml` | Basic sweep: 2 models × 4 recipes × 2 clusters = 16 jobs |
| `experiment.yaml` | Sweep with eval overrides: per-model benchmarks, per-launcher TP |

## How Recipes Work

A recipe is a YAML file with technique sections. The simplest recipe:

```yaml
version: "1.0"
quantization:
  preset: fp8
```

`preset: fp8` resolves to the full ModelOpt config dict (`FP8_DEFAULT_CFG`). You can add overrides on top:

```yaml
quantization:
  preset: nvfp4
  algorithm: { method: awq_lite }
  kv_cache: { format: fp8 }
  overrides:
    - pattern: "*layers.0*"
      enable: false          # first layer at full precision
```

Multi-technique recipes include multiple sections. Execution order follows ModelOpt semantics:

```text
pruning → sparsity → quantization → distillation
```

## Python API

```python
import modelopt.torch.quantization as mtq
from modelopt.torch.recipes import load_recipe

# Load and resolve a recipe
result = load_recipe("examples/recipes/ptq/ptq_fp8.yaml")

# Use the resolved config with ModelOpt
config = result["quantize_config"]
# model and calibrate are user-provided (HuggingFace model + calibration loop)
model = mtq.quantize(model, config, forward_loop=calibrate)
```

## Tests

```bash
# Run all recipe tests (34 tests)
python -m pytest tests/unit/torch/recipes/ -v
```
