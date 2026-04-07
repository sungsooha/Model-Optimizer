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


import os
import re
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from vllm_ptq_utils import calibrate_fun, get_quant_config
from vllm_reload_utils import (
    convert_dict_to_vllm,
    convert_modelopt_state_to_vllm,
    load_state_dict_from_path,
    restore_from_modelopt_state_vllm,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.vllm import disable_compilation
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

quant_config: dict[str, Any] = {
    "dataset": os.environ.get("QUANT_DATASET", "cnn_dailymail"),
    "calib_size": int(os.environ.get("QUANT_CALIB_SIZE", 512)),
    "quant_cfg": os.environ.get("QUANT_CFG", None),
    "kv_quant_cfg": os.environ.get("KV_QUANT_CFG", None),
    "quant_file_path": os.environ.get("QUANT_FILE_PATH", None),
    "modelopt_state_path": os.environ.get("MODELOPT_STATE_PATH", None),
    "calib_batch_size": int(os.environ.get("CALIB_BATCH_SIZE", 1)),
}


def _extract_projection_pqs(
    modelopt_weights: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract individual projection _pre_quant_scale values before QKV/gate_up merge.

    Returns dict mapping layer index to projection pqs:
        {layer_idx: {'qkv': {'q': tensor, 'k': tensor, 'v': tensor},
                     'gate_up': {'gate': tensor, 'up': tensor}}}
    """
    layer_pqs: dict[int, dict[str, dict[str, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for key, value in modelopt_weights.items():
        if not isinstance(value, dict) or "_pre_quant_scale" not in value:
            continue
        # Match q/k/v_proj.input_quantizer
        qkv_match = re.search(
            r"layers\.(\d+)\.self_attn\.([qkv])_proj\.input_quantizer$", key
        )
        if qkv_match:
            layer_idx = int(qkv_match.group(1))
            proj = qkv_match.group(2)
            layer_pqs[layer_idx]["qkv"][proj] = value["_pre_quant_scale"].clone()
            continue
        # Match gate/up_proj.input_quantizer
        gate_up_match = re.search(
            r"layers\.(\d+)\.mlp\.(gate|up)_proj\.input_quantizer$", key
        )
        if gate_up_match:
            layer_idx = int(gate_up_match.group(1))
            proj = gate_up_match.group(2)
            layer_pqs[layer_idx]["gate_up"][proj] = value["_pre_quant_scale"].clone()

    return dict(layer_pqs)


def _compensate_awq_pqs_merge(
    model: torch.nn.Module,
    layer_pqs: dict[int, dict[str, dict[str, torch.Tensor]]],
) -> None:
    """Compensate fused weights for AWQ _pre_quant_scale max-merge mismatch.

    AWQ bakes 1/individual_pqs into each projection's weight during calibration.
    After QKV/gate_up fusion, merged_pqs = max(proj_pqs...) is applied to the shared
    input. This creates a per-channel error: merged_pqs/individual_pqs != 1.

    Fix: weight[proj_rows, :] *= individual_pqs / merged_pqs (column-wise correction).
    """
    if not layer_pqs:
        return

    compensated = 0
    for name, module in model.named_modules():
        # Match qkv_proj or gate_up_proj modules
        layer_match = re.search(r"layers\.(\d+)\.", name)
        if not layer_match:
            continue
        layer_idx = int(layer_match.group(1))
        if layer_idx not in layer_pqs:
            continue
        if not hasattr(module, "weight") or not hasattr(module, "output_partition_sizes"):
            continue

        if name.endswith(".qkv_proj") and "qkv" in layer_pqs[layer_idx]:
            pqs = layer_pqs[layer_idx]["qkv"]
            proj_keys = ["q", "k", "v"]
        elif name.endswith(".gate_up_proj") and "gate_up" in layer_pqs[layer_idx]:
            pqs = layer_pqs[layer_idx]["gate_up"]
            proj_keys = ["gate", "up"]
        else:
            continue

        if not all(p in pqs for p in proj_keys):
            continue

        pqs_tensors = [pqs[p] for p in proj_keys]
        merged_pqs = torch.stack(pqs_tensors).max(dim=0)[0]
        partition_sizes = module.output_partition_sizes

        row_start = 0
        for proj_key, size in zip(proj_keys, partition_sizes):
            ratio = pqs[proj_key].to(module.weight.device) / merged_pqs.to(
                module.weight.device
            )
            # ratio is [hidden_size], weight is [out_features, hidden_size]
            # Multiply columns: weight[rows, j] *= ratio[j]
            module.weight.data[row_start : row_start + size, :] *= ratio
            row_start += size

        compensated += 1

    if compensated > 0:
        print(
            f"AWQ pqs merge compensation applied to {compensated} fused projections"
        )


def _fakequant_run_prolog_worker(self) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        self.model_runner.model_config.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = self.model_runner.model
    # Unwrap CUDAGraphRunner (or similar wrappers) before modelopt operations.
    # Without this, restore_from_modelopt_state_vllm → _check_init_modellike →
    # init_model_from_model_like triggers CUDAGraphRunner.__call__ during init,
    # which asserts _forward_context is not None (only set during inference).
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    if quant_config["modelopt_state_path"]:
        print(f"Loading modelopt state from {quant_config['modelopt_state_path']}")
        # Load on CPU to avoid failures when the checkpoint was saved from a different
        # GPU mapping
        modelopt_state = torch.load(
            quant_config["modelopt_state_path"], weights_only=True, map_location="cpu"
        )
        modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
        map_fun = (
            self.model_runner.model.hf_to_vllm_mapper.apply_dict
            if hasattr(self.model_runner.model, "hf_to_vllm_mapper")
            else None
        )
        # convert modelopt state to vllm format
        modelopt_state = convert_modelopt_state_to_vllm(modelopt_state, map_fun=map_fun)
        # restore model from modelopt state
        restore_from_modelopt_state_vllm(model, modelopt_state)

        if modelopt_weights is not None:
            # Extract individual projection pqs BEFORE merge (needed for AWQ compensation)
            projection_pqs = _extract_projection_pqs(modelopt_weights)
            # convert quantizer state values to vllm format (merges Q/K/V pqs via max)
            modelopt_weights = convert_dict_to_vllm(modelopt_weights, map_fun=map_fun)
            # set quantizer state to model's state_dict
            mtq.utils.set_quantizer_state_dict(model, modelopt_weights)
            # Compensate fused weights for AWQ pqs max-merge mismatch.
            # Must happen after state restore (pqs loaded) and before fold_weight.
            _compensate_awq_pqs_merge(model, projection_pqs)

    else:
        if quant_config["quant_file_path"]:
            print("Will load quant, so only do a single sample calibration")
            quant_config["calib_size"] = 1

        calib_dataloader = get_dataset_dataloader(
            dataset_name=quant_config["dataset"],
            tokenizer=tokenizer,
            batch_size=quant_config["calib_batch_size"],
            num_samples=quant_config["calib_size"],
            device=self.device,
        )

        calibrate_loop = calibrate_fun(calib_dataloader, self)

        if hasattr(model, "unwrap"):
            model = model.unwrap()

        quant_cfg = get_quant_config(quant_config, model)

        # quantize model
        with disable_compilation(model):
            print("Quantizing model...")
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        quantizer_file_path = quant_config["quant_file_path"]
        if quantizer_file_path:
            # Get amax and other quantizer state from the quantizer file
            # this can be used with Megatron-LM exported model using export_mcore_gpt_to_hf_vllm_fq
            current_state_dict = load_state_dict_from_path(self, quantizer_file_path, model)
            model.load_state_dict(current_state_dict)

            # Only barrier if distributed is actually initialized (avoids deadlocks).
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    mtq.fold_weight(model)
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert not module.is_enabled, f"quantizer {name} is still enabled"


class FakeQuantWorker(BaseWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        with disable_compilation(model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> float:
        if (
            quant_config["quant_cfg"]
            or quant_config["kv_quant_cfg"]
            or quant_config["modelopt_state_path"]
        ):
            _fakequant_run_prolog_worker(self)
        return super().compile_or_warm_up_model()
