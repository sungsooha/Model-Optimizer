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

"""Tests for end-to-end recipe loading."""

import pytest

from modelopt.torch.recipes import load_recipe


def test_load_fp8(examples_dir):
    result = load_recipe(examples_dir / "ptq" / "ptq_fp8.yaml")
    assert "quantize_config" in result
    qcfg = result["quantize_config"]
    assert "quant_cfg" in qcfg
    assert "*weight_quantizer" in qcfg["quant_cfg"]


def test_load_nvfp4_awq(examples_dir):
    result = load_recipe(examples_dir / "ptq" / "ptq_nvfp4_awq.yaml")
    assert "quantize_config" in result
    assert "export" in result


def test_load_auto_quantize(examples_dir):
    result = load_recipe(examples_dir / "auto" / "auto_quantize.yaml")
    assert "auto_quantize_kwargs" in result
    kwargs = result["auto_quantize_kwargs"]
    assert "quantization_formats" in kwargs
    assert len(kwargs["quantization_formats"]) > 0


def test_load_all_examples(examples_dir):
    for yaml_file in sorted(examples_dir.rglob("*.yaml")):
        if "experiments" in yaml_file.parts:
            continue  # skip sweep/experiment configs
        result = load_recipe(yaml_file)
        assert isinstance(result, dict), f"Failed: {yaml_file.name}"
        assert len(result) > 0, f"Empty result: {yaml_file.name}"


def test_scale_type_override():
    """scale_type shorthand merges into block_sizes.type, preserving existing block_sizes."""
    import yaml

    from modelopt.torch.recipes.schema.models import RecipeConfig
    from modelopt.torch.recipes.schema.resolver import resolve_recipe

    # Test 1: Override existing preset pattern — preserves block_sizes fields
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: nvfp4_local_hessian
      overrides:
        - pattern: "*weight_quantizer"
          scale_type: dynamic
    """)
    )
    result = resolve_recipe(recipe)
    qcfg = result["quantize_config"]["quant_cfg"]
    wq = qcfg["*weight_quantizer"]
    # scale_type should have changed block_sizes.type
    assert wq["block_sizes"]["type"] == "dynamic"
    # Original block_sizes fields preserved from the preset
    assert wq["block_sizes"][-1] == 16
    assert wq["block_sizes"]["scale_bits"] == (4, 3)
    # Other quantizer fields preserved
    assert wq["num_bits"] == (2, 1)

    # Test 2: New pattern — creates entry with just scale_type
    recipe2 = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: nvfp4_local_hessian
      overrides:
        - pattern: "*self_attn*weight_quantizer"
          scale_type: dynamic
    """)
    )
    result2 = resolve_recipe(recipe2)
    qcfg2 = result2["quantize_config"]["quant_cfg"]
    attn = qcfg2["*self_attn*weight_quantizer"]
    assert attn["block_sizes"]["type"] == "dynamic"


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_recipe("/nonexistent/path.yaml")
