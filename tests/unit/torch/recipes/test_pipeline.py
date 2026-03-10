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

"""Tests for pipeline orchestrator."""

import yaml

from modelopt.torch.recipes.pipeline import load_and_plan, plan_pipeline
from modelopt.torch.recipes.schema.models import RecipeConfig


def test_ptq_single_step():
    """PTQ recipe produces a single quantization step."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 1
    assert plan.steps[0].technique == "quantization (ptq)"
    assert "quantize" in plan.steps[0].api_call
    assert plan.steps[0].config.get("quant_cfg") is not None


def test_qat_step():
    """QAT recipe shows training API."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      mode: qat
      preset: fp8
      training:
        learning_rate: 1e-5
        num_epochs: 2
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 1
    assert plan.steps[0].technique == "quantization (qat)"
    assert "train()" in plan.steps[0].api_call
    assert plan.steps[0].training["learning_rate"] == 1e-5
    assert plan.steps[0].training["num_epochs"] == 2


def test_distillation_step():
    """Distillation recipe produces quantization + distillation steps."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
    distillation:
      teacher: "meta-llama/Llama-3-70B"
      criterion: kl_div
      kd_loss_weight: 0.5
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 2
    assert plan.steps[0].technique == "quantization (ptq)"
    assert plan.steps[1].technique == "distillation"
    assert plan.steps[1].config["teacher_model"] == "meta-llama/Llama-3-70B"
    assert plan.steps[1].config["loss_balancer"]["kd_loss_weight"] == 0.5


def test_sparsity_quantization_order():
    """Sparsity runs before quantization."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    sparsity:
      method: sparse_gpt
      sparsity: 0.5
      pattern: "2:4"
    quantization:
      preset: fp8
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 2
    assert plan.steps[0].technique == "sparsity"
    assert plan.steps[1].technique == "quantization (ptq)"
    assert plan.steps[0].config["method"] == "sparse_gpt"
    assert plan.steps[0].config["pattern"] == "2:4"


def test_three_technique_pipeline():
    """Sparsity + quantization + distillation in correct order."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    sparsity:
      method: wanda
      sparsity: 0.5
    quantization:
      preset: fp8
    distillation:
      teacher: "meta-llama/Llama-3-70B"
    """)
    )
    plan = plan_pipeline(recipe)
    assert len(plan.steps) == 3
    assert plan.steps[0].technique == "sparsity"
    assert plan.steps[1].technique == "quantization (ptq)"
    assert plan.steps[2].technique == "distillation"


def test_per_technique_calibration():
    """Each technique gets its own calibration config."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    sparsity:
      method: sparse_gpt
      calibration:
        dataset: pile
        num_samples: 2048
    quantization:
      preset: fp8
      calibration:
        dataset: cnn_dailymail
        num_samples: 512
    """)
    )
    plan = plan_pipeline(recipe)
    assert plan.steps[0].calibration["dataset"] == "pile"
    assert plan.steps[0].calibration["num_samples"] == 2048
    assert plan.steps[1].calibration["dataset"] == "cnn_dailymail"
    assert plan.steps[1].calibration["num_samples"] == 512


def test_export_in_plan():
    """Export config appears in the plan."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
    export:
      format: hf
      tensor_parallel: 8
    """)
    )
    plan = plan_pipeline(recipe)
    assert plan.export is not None
    assert plan.export["format"] == "hf"
    assert plan.export["tensor_parallel"] == 8


def test_dry_run_output():
    """Dry-run produces readable output."""
    recipe = RecipeConfig.model_validate(
        yaml.safe_load("""
    version: "1.0"
    quantization:
      preset: fp8
    """)
    )
    plan = plan_pipeline(recipe, recipe_path="test.yaml")
    output = plan.dry_run()
    assert "Pipeline Plan: test.yaml" in output
    assert "quantization (ptq)" in output
    assert "Step 1" in output


def test_load_and_plan_all_examples():
    """All example recipes produce valid pipeline plans."""
    from pathlib import Path

    recipes_dir = Path("examples/recipes")
    for path in sorted(recipes_dir.rglob("*.yaml")):
        if "experiments" in path.parts:
            continue  # skip sweep/experiment configs
        plan = load_and_plan(str(path))
        assert len(plan.steps) >= 1, f"{path} produced empty plan"
        assert plan.recipe_path == str(path)
