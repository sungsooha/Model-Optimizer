# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from vllm.distributed.parallel_state import get_tp_group

from modelopt.torch.opt.conversion import (
    ModelLikeModule,
    ModeloptStateManager,
    _check_init_modellike,
)
from modelopt.torch.quantization.conversion import (
    convert_to_quantized_model,
    restore_quantizer_state,
)
from modelopt.torch.quantization.utils import is_quantized


def _values_equal(v1: Any, v2: Any) -> bool:
    """Compare values, handling dicts with tensors."""
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() != v2.keys():
            return False
        return all(
            torch.equal(v1[k], v2[k]) if isinstance(v1[k], torch.Tensor) else v1[k] == v2[k]
            for k in v1
        )
    elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        return torch.equal(v1, v2)
    return v1 == v2


def _convert_key_for_vllm(key: str, value: Any) -> tuple[str, str | None, Any]:
    """
    Transform a single key from HuggingFace format to vLLM format.

    Returns:
        Tuple of (action, new_key_or_group, value) where action is one of:
        - "copy": Copy value to new_key directly
        - "group": Add to merge group identified by new_key
        - "skip": Skip this key entirely
    """
    if "quantizer" not in key:
        return ("copy", key, value)

    # Skip softmax_quantizer and lm_head quantizers(not needed in vLLM)
    if "softmax_quantizer" in key or (key.startswith("lm_head.") and "quantizer" in key):
        return ("skip", None, None)

    # Check if this is a q/k/v projection that needs merging
    qkv_match = re.search(r"(.*\.)([qkv])_proj\.([^.]+_quantizer)(\..+)?$", key)
    if qkv_match:
        suffix = qkv_match.group(4) or ""
        group_key = qkv_match.group(1) + "qkv_proj." + qkv_match.group(3) + suffix
        return ("group", group_key, value)

    # Check if this is an expert gate/up projection
    # if "mixer" not in key:
    expert_gate_up_match = re.search(
        r"(.*\.experts)\.\d+\.(gate|up)_proj\.([^.]+_quantizer)(\..+)?$", key
    )
    if expert_gate_up_match:
        suffix = expert_gate_up_match.group(4) or ""
        group_key = expert_gate_up_match.group(1) + ".w13_" + expert_gate_up_match.group(3) + suffix
        return ("group", group_key, value)

    # Check if this is a non-expert gate/up projection that needs merging
    if "mixer" not in key and "experts" not in key:
        gate_up_match = re.search(r"(.*\.)(gate|up)_proj\.([^.]+_quantizer)(\..+)?$", key)
        if gate_up_match:
            suffix = gate_up_match.group(4) or ""
            group_key = gate_up_match.group(1) + "gate_up_proj." + gate_up_match.group(3) + suffix
            return ("group", group_key, value)

    # Check if this is an expert down_proj
    # if "mixer" not in key:
    expert_down_match = re.search(r"(.*\.experts)\.\d+\.down_proj\.([^.]+_quantizer)(\..+)?$", key)
    if expert_down_match:
        suffix = expert_down_match.group(3) or ""
        group_key = expert_down_match.group(1) + ".w2_" + expert_down_match.group(2) + suffix
        return ("group", group_key, value)

    # Transform bmm_quantizer keys: self_attn.q/k/v_bmm_quantizer -> self_attn.attn.q/k/v_bmm_quantizer
    bmm_match = re.search(r"(.*\.self_attn)\.([qkv]_bmm_quantizer.*)$", key) or re.search(
        r"(.*\.mixer)\.([qkv]_bmm_quantizer.*)$", key
    )
    if bmm_match:
        new_key = bmm_match.group(1) + ".attn." + bmm_match.group(2)
        return ("copy", new_key, value)

    # Copy other quantizer keys as-is (like o_proj, down_proj)
    return ("copy", key, value)


def _group_keys_for_vllm(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], defaultdict[str, list[tuple[str, Any]]]]:
    """
    Process state dict and group keys that need merging.

    Returns:
        Tuple of (direct_copy_dict, merge_groups)
    """
    vllm_state_dict = {}
    merge_groups = defaultdict(list)

    for key, value in state_dict.items():
        action, new_key, new_value = _convert_key_for_vllm(key, value)
        if new_key is None or new_value is None:
            assert action == "skip", (
                f"Expected action to be 'skip' for key {key}, value {value}, got {action}"
            )
            continue
        if action == "copy":
            vllm_state_dict[new_key] = new_value
        elif action == "group":
            merge_groups[new_key].append((key, new_value))
        # action == "skip" does nothing

    return vllm_state_dict, merge_groups


def _merge_values_by_max_or_concat(merged_key: str, key_value_pairs: list[tuple[str, Any]]) -> Any:
    """
    Merge values by taking max for amax, concatenating for others.
    Used for quantizer state weights (tensor values).

    For GQA models, amax tensors may have different sizes (Q has more heads
    than K/V). In this case, use torch.cat instead of torch.stack+max, since
    each amax corresponds to its own weight group in the fused projection.
    """
    values = [value for _, value in key_value_pairs]

    # Check if values are dicts (OrderedDict) containing tensors
    if isinstance(values[0], dict):
        merged_value = {}
        for dict_key in values[0]:
            tensors = [v[dict_key] for v in values]
            if "_amax" in dict_key:
                merged_value[dict_key] = _merge_amax_tensors(tensors)
            else:
                merged_value[dict_key] = torch.cat(tensors, dim=0)
        return merged_value
    else:
        # Values are tensors directly
        if "_amax" in merged_key:
            merged_value = _merge_amax_tensors(values)
        else:
            merged_value = torch.cat(values, dim=0)
        return merged_value


def _merge_amax_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Merge amax tensors: stack+max if same size (MHA), cat if different (GQA)."""
    if all(t.shape == tensors[0].shape for t in tensors[1:]):
        return torch.stack(tensors).max(dim=0)[0]
    else:
        return torch.cat(tensors, dim=0)


def _merge_values_require_identical(merged_key: str, key_value_pairs: list[tuple[str, Any]]) -> Any:
    """
    Merge values by requiring all values to be identical.
    Used for quantizer state (config/metadata).

    For GQA models, shape-dependent fields (_amax_shape_for_export,
    _pytorch_state_metadata) may differ across Q/K/V projections because
    Q has more heads than K/V. These fields are merged by summing the
    output dimension rather than requiring identical values.
    """
    keys = [k for k, _ in key_value_pairs]
    values = [v for _, v in key_value_pairs]
    first_value = values[0]

    # If all values are identical, return early
    if all(_values_equal(val, first_value) for val in values[1:]):
        return first_value

    # Values differ — try smart merge for dict metadata (GQA case)
    if isinstance(first_value, dict):
        return _smart_merge_metadata(merged_key, keys, values)

    raise ValueError(
        f"Cannot merge keys into '{merged_key}': values differ.\n"
        f"  '{keys[0]}' has value: {first_value}\n"
        f"  '{keys[1]}' has value: {values[1]}"
    )


# Fields whose first element (output dim) should be summed when merging Q/K/V
_SHAPE_SUM_FIELDS = {"_amax_shape_for_export"}


def _smart_merge_metadata(
    merged_key: str, keys: list[str], values: list[dict[str, Any]]
) -> dict[str, Any]:
    """Merge quantizer metadata dicts, handling GQA-asymmetric shape fields.

    Most fields must be identical (num_bits, block_sizes, etc.).
    Shape-dependent fields are merged by summing the output dimension.
    _pytorch_state_metadata buffer shapes are summed similarly.
    """
    merged = {}
    for field in values[0]:
        field_values = [v[field] for v in values]

        if field in _SHAPE_SUM_FIELDS:
            # Sum the output dimension (first element of tuple)
            # e.g., (4096, -1) + (1024, -1) + (1024, -1) → (6144, -1)
            first = field_values[0]
            if isinstance(first, tuple) and len(first) >= 1:
                summed_dim = sum(fv[0] for fv in field_values)
                merged[field] = (summed_dim,) + first[1:]
            else:
                merged[field] = field_values[0]
        elif field == "_pytorch_state_metadata":
            # Merge buffer shapes by summing the first dim
            merged[field] = _merge_pytorch_state_metadata(field_values)
        else:
            # Require identical for all other fields
            if not all(_values_equal(fv, field_values[0]) for fv in field_values[1:]):
                raise ValueError(
                    f"Cannot merge keys into '{merged_key}': "
                    f"field '{field}' differs across projections.\n"
                    f"  '{keys[0]}' has {field}: {field_values[0]}\n"
                    f"  '{keys[1]}' has {field}: {field_values[1]}"
                )
            merged[field] = field_values[0]

    return merged


def _merge_pytorch_state_metadata(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge _pytorch_state_metadata by summing buffer shape dim 0."""
    merged = {}
    for key in metas[0]:
        vals = [m[key] for m in metas]
        if key == "buffers" and isinstance(vals[0], dict):
            merged_buffers = {}
            for buf_name in vals[0]:
                buf_vals = [v[buf_name] for v in vals]
                if isinstance(buf_vals[0], dict) and "shape" in buf_vals[0]:
                    # Sum first dim of shape, keep rest identical
                    shapes = [bv["shape"] for bv in buf_vals]
                    summed_shape = torch.Size(
                        [sum(s[0] for s in shapes)] + list(shapes[0][1:])
                    )
                    merged_buf = dict(buf_vals[0])
                    merged_buf["shape"] = summed_shape
                    merged_buffers[buf_name] = merged_buf
                else:
                    merged_buffers[buf_name] = buf_vals[0]
            merged[key] = merged_buffers
        else:
            merged[key] = vals[0]
    return merged


def convert_dict_to_vllm(
    state_dict: dict[str, Any],
    max_or_concat: bool = True,
    map_fun: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Common implementation for converting quantizer state from HF to vLLM format.

    Args:
        state_dict: Input state dict
        max_or_concat: Whether to merge grouped values by taking max/concatenate or require identical
        map_fun: Function to map the state dict to vLLM format
    """
    vllm_state_dict, merge_groups = _group_keys_for_vllm(state_dict)

    merge_fn = _merge_values_by_max_or_concat if max_or_concat else _merge_values_require_identical

    # Merge grouped values
    for merged_key, key_value_pairs in merge_groups.items():
        if len(key_value_pairs) > 1:
            merged_value = merge_fn(merged_key, key_value_pairs)
            vllm_state_dict[merged_key] = merged_value
        else:
            # Single key, just rename it
            _, value = key_value_pairs[0]
            vllm_state_dict[merged_key] = value
    return map_fun(vllm_state_dict) if map_fun is not None else vllm_state_dict


def convert_modelopt_state_to_vllm(
    modelopt_state: dict[str, Any],
    map_fun: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Convert modelopt state from HuggingFace format to vLLM compatible format.

    This function converts the quantizer state from HuggingFace format to vLLM compatible format.

    Args:
        modelopt_state: HuggingFace modelopt state dict

    Returns:
        vLLM compatible modelopt state dict
    """
    modelopt_state_dict = modelopt_state.pop("modelopt_state_dict", [])
    for idx, current_mode in enumerate(modelopt_state_dict):
        current_mode_metadata = current_mode[1].pop("metadata", {})
        current_mode_quant_state = current_mode_metadata.pop("quantizer_state", {})
        if current_mode_quant_state:
            current_mode_metadata["quantizer_state"] = convert_dict_to_vllm(
                current_mode_quant_state, max_or_concat=False, map_fun=map_fun
            )
        else:
            current_mode_metadata.pop("quantizer_state", None)
        current_mode[1]["metadata"] = current_mode_metadata
        modelopt_state_dict[idx] = (current_mode[0], current_mode[1])
    modelopt_state["modelopt_state_dict"] = modelopt_state_dict
    return modelopt_state


def filter_modelopt_state_quantizer_state_for_model(
    modelopt_state: dict[str, Any], model: torch.nn.Module
) -> None:
    """
    Align quantizer_state in modelopt_state metadata with the model.

    - Removes keys not in the model (handles TP sharding - each rank has a subset).
    - Removes keys only when the quantizer is disabled (in the model).
    - Adds keys for quantizers in the model but not in metadata (e.g. disabled/excluded).
    Modifies modelopt_state in place. Call after convert_to_quantized_model so the model has
    quantizers.

    Args:
        modelopt_state: Modelopt state dict (modified in place)
        model: Model with quantizers (must already be converted)
    """
    from modelopt.torch.quantization.conversion import quantizer_state
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
    from modelopt.torch.utils import get_unwrapped_name

    model_qstate = quantizer_state(model)
    model_keys = set(model_qstate.keys())
    # Build name -> is_enabled for quantizers in the model
    disabled_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, (TensorQuantizer, SequentialQuantizer)):
            unwrapped_name = get_unwrapped_name(name, model)
            if not getattr(module, "is_enabled", True):
                disabled_keys.add(unwrapped_name)

    for mode_entry in modelopt_state.get("modelopt_state_dict", []):
        metadata = mode_entry[1].get("metadata", {})
        if "quantizer_state" in metadata:
            saved = metadata["quantizer_state"]
            # Keep keys that exist in the model, but remove if quantizer is disabled
            filtered = {
                k: v for k, v in saved.items() if k in model_keys and k not in disabled_keys
            }
            # Add state for quantizers in model but not in metadata (e.g. disabled/excluded)
            for k in model_keys - filtered.keys():
                filtered[k] = model_qstate[k]
            metadata["quantizer_state"] = filtered


def restore_from_modelopt_state_vllm(
    model: torch.nn.Module, modelopt_state: dict[str, Any]
) -> torch.nn.Module:
    """
    vLLM-specific restore that filters quantizer_state to match the model before restore.

    Handles TP sharding (each rank has a subset of quantizers) and excluded disabled quantizers
    by running convert first, filtering metadata to model keys, then restoring. Uses the same
    restore logic as restore_from_modelopt_state but with filtering for quantize modes.
    """
    model = model if isinstance(model, torch.nn.Module) else ModelLikeModule(model)
    manager = ModeloptStateManager(model=model, init_state=True)
    manager.load_state_dict(
        modelopt_state["modelopt_state_dict"], modelopt_state["modelopt_version"]
    )

    for i, (m, config, metadata) in enumerate(manager.modes_with_states()):
        if i == 0:
            model = _check_init_modellike(model, m)
        # For quantize modes: convert first (if not already), filter metadata to model keys, then restore state.
        # This handles TP (model has subset of quantizers) and excluded disabled quantizers.
        if "quantizer_state" in metadata:
            if not is_quantized(model):
                convert_to_quantized_model(model, config)
            filter_modelopt_state_quantizer_state_for_model(
                {"modelopt_state_dict": manager._state}, model
            )
            # Re-fetch metadata after filtering (manager._state was modified in place)
            metadata = manager._state[i][1]["metadata"]
            model = restore_quantizer_state(model, config, metadata)
        else:
            model = m.restore(model, config, metadata)

    if not manager.has_state and isinstance(model, ModelLikeModule):
        model = model.init_modellike()
    assert not isinstance(model, ModelLikeModule), "Model must be a regular Module now!"
    return model


def process_state_dict_for_tp(saved_qstate_dict, current_state_dict):
    """Shard quantizer tensors for tensor parallelism by matching expected shapes."""
    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    tp_world_size = tp_group.world_size

    result = {}
    for key, value in saved_qstate_dict.items():
        if key in current_state_dict:
            expected = current_state_dict[key]
            if not hasattr(value, "shape") or not hasattr(expected, "shape"):
                result[key] = value
                continue
            expected_shape = expected.shape
            value_shape = value.shape
            if value_shape != expected_shape:
                # Verify compatible rank before indexing
                if len(value_shape) != len(expected_shape):
                    raise ValueError(
                        f"Cannot infer TP shard dim for {key}: rank mismatch "
                        f"(checkpoint rank={len(value_shape)}, expected rank={len(expected_shape)})"
                    )
                # Find the dimension that was tensor-parallel sharded.
                # We expect exactly one dimension to satisfy:
                #   checkpoint_dim == expected_dim * tp_world_size
                shard_dims = [
                    d
                    for d in range(len(expected_shape))
                    if value_shape[d] == expected_shape[d] * tp_world_size
                ]
                if len(shard_dims) != 1:
                    raise ValueError(
                        f"Cannot infer TP shard dim for {key}: "
                        f"expected_shape={tuple(expected_shape)}, checkpoint_shape={tuple(value_shape)}"
                    )

                shard_dim = shard_dims[0]
                shard_size = expected_shape[shard_dim]
                start = tp_rank * shard_size
                end = start + shard_size
                if end > value_shape[shard_dim]:
                    raise ValueError(
                        f"TP shard out of bounds for {key}: "
                        f"expected_shape={tuple(expected_shape)}, checkpoint_shape={tuple(value_shape)}"
                    )
                value = value.narrow(shard_dim, start, shard_size).contiguous()
        result[key] = value

    return result


def load_state_dict_from_path(
    fakequant_runner: Any, quantizer_file_path: str, model: Any
) -> dict[str, Any]:
    fakequant_runner.model_runner._dummy_run(1)
    print(f"Loading quantizer values from {quantizer_file_path}")
    # Load on CPU to avoid failures when the checkpoint was saved from a different
    # GPU mapping
    saved_quant_dict = torch.load(quantizer_file_path, weights_only=True, map_location="cpu")
    # convert quant keys to vLLM format
    if hasattr(fakequant_runner.model_runner.model, "hf_to_vllm_mapper"):
        saved_quant_dict = fakequant_runner.model_runner.model.hf_to_vllm_mapper.apply_dict(
            saved_quant_dict
        )
        saved_quant_dict = {
            key.replace("quantizer_", "quantizer._"): value
            for key, value in saved_quant_dict.items()
            if "quantizer_" in key
        }
    saved_quant_dict = convert_dict_to_vllm(saved_quant_dict)

    current_state_dict = model.state_dict()
    # Count quant keys in checkpoint and model
    checkpoint_quant_keys = [key for key in saved_quant_dict if "quantizer" in key]
    model_quant_keys = [key for key in current_state_dict if "quantizer" in key]
    for key in checkpoint_quant_keys:
        if key not in model_quant_keys:
            print(f"Key {key} not found in model state dict, but exists in checkpoint")
    for key in model_quant_keys:
        if key not in checkpoint_quant_keys:
            raise ValueError(f"Key {key} not found in checkpoint state dict, but exists in model")

    checkpoint_quant_count = len(checkpoint_quant_keys)
    model_quant_count = len(model_quant_keys)

    # Ensure counts match
    if checkpoint_quant_count != model_quant_count:
        warnings.warn(
            f"Mismatch in quantizer state key counts: checkpoint has {checkpoint_quant_count} "
            f"quant keys but model has {model_quant_count} quantizer state keys. "
            f"This can happen if the model is using PP."
        )

    # Update quant values
    saved_quant_dict = process_state_dict_for_tp(saved_quant_dict, current_state_dict)
    for key, value in saved_quant_dict.items():
        if key in current_state_dict:
            current_state_dict[key] = value.to(current_state_dict[key].device)
    return current_state_dict
