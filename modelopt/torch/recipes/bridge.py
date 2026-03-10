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

"""Bridge between recipe system (load_recipe) and quant_flow (PTQConfig.hf_ptq).

Converts load_recipe() output into the dict format expected by hf_ptq.py.
This is the single coupling point between the recipe system and quant_flow.
"""

from __future__ import annotations

from typing import Any


def recipe_to_hf_ptq_args(resolved: dict[str, Any]) -> dict[str, Any]:
    """Convert load_recipe() output to hf_ptq-compatible config dict.

    load_recipe() returns:
        {"quantize_config": {"quant_cfg": {...}, "algorithm": "awq_lite"},
         "calibration": {"dataset": "cnn_dailymail", "num_samples": 512, ...},
         "export": {"format": "hf", "output_dir": "...", ...}}

    hf_ptq expects:
        {"qformat": "fp8", "dataset": "cnn_dailymail", "calib_size": 512, ...}
    """
    hf_ptq: dict[str, Any] = {}

    # Map quantize_config → qformat
    if "quantize_config" in resolved:
        qcfg = resolved["quantize_config"]
        # The resolved config dict is the full mtq.quantize() input.
        # Store it for direct use; also extract qformat for hf_ptq CLI compatibility.
        hf_ptq["_resolved_quantize_config"] = qcfg

    # Map calibration → hf_ptq args
    if "calibration" in resolved:
        cal = resolved["calibration"]
        if cal.get("dataset"):
            hf_ptq["dataset"] = cal["dataset"]
        if cal.get("num_samples"):
            hf_ptq["calib_size"] = cal["num_samples"]
        if cal.get("max_sequence_length"):
            hf_ptq["seq_len"] = cal["max_sequence_length"]
        if cal.get("batch_size"):
            hf_ptq["batch_size"] = cal["batch_size"]

    # Map export → hf_ptq args
    if "export" in resolved:
        exp = resolved["export"]
        if exp.get("output_dir"):
            hf_ptq["export_path"] = exp["output_dir"]
        if exp.get("tensor_parallel"):
            hf_ptq["tp_size"] = exp["tensor_parallel"]

    return hf_ptq


def summarize_recipe(recipe_path: str, resolved: dict[str, Any], raw_yaml: dict) -> dict[str, Any]:
    """Extract human-readable summary from recipe YAML and resolved config.

    Returns a dict with keys: preset, algorithm, kv_cache, overrides_count, type.
    """
    quant = raw_yaml.get("quantization", {})
    auto_q = raw_yaml.get("auto_quantize", {})

    if quant:
        algo = quant.get("algorithm", "default")
        if isinstance(algo, dict):
            algo = algo.get("method", "default")

        return {
            "type": "quantize",
            "preset": quant.get("preset", "custom"),
            "algorithm": algo,
            "kv_cache": quant.get("kv_cache", {}).get("format", "none")
            if quant.get("kv_cache")
            else "none",
            "overrides_count": len(quant.get("overrides", [])),
            "disabled_patterns_count": len(quant.get("disabled_patterns", [])),
        }
    elif auto_q:
        return {
            "type": "auto_quantize",
            "preset": "auto",
            "algorithm": auto_q.get("method", "gradient"),
            "effective_bits": auto_q.get("effective_bits"),
            "formats_count": len(auto_q.get("formats", [])),
            "kv_cache": "none",
            "overrides_count": 0,
            "disabled_patterns_count": 0,
        }

    return {"type": "unknown", "preset": "none"}
