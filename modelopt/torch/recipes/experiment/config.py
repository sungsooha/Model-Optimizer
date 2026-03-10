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

"""Sweep configuration schema.

Self-contained dataclasses — no inheritance from quant_flow classes.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any

import yaml

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SweepEvalConfig:
    """Evaluation configuration for sweep jobs.

    Supports per-model and per-launcher overrides for benchmark selection
    and deployment parameters.
    """

    engine: str = "vllm"
    tensor_parallel_size: int = 8
    tasks: list[str] = field(default_factory=list)
    benchmark_set: str | None = None
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    launcher_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def resolve_for_job(self, model: str, launcher: str) -> dict[str, Any]:
        """Return eval config dict with overrides applied for a specific job."""
        tasks = list(self.tasks)
        engine = self.engine
        tp = self.tensor_parallel_size

        # Apply model overrides (e.g., code model → different benchmarks)
        m_over = self.model_overrides.get(model, {})
        if "tasks" in m_over:
            tasks = m_over["tasks"]
        if "engine" in m_over:
            engine = m_over["engine"]
        if "tensor_parallel_size" in m_over:
            tp = m_over["tensor_parallel_size"]

        # Apply launcher overrides (e.g., GB200 → lower TP)
        l_over = self.launcher_overrides.get(launcher, {})
        if "tensor_parallel_size" in l_over:
            tp = l_over["tensor_parallel_size"]
        if "engine" in l_over:
            engine = l_over["engine"]

        result: dict[str, Any] = {
            "deployment": {
                "engine": engine,
                "tensor_parallel_size": tp,
            },
            "evaluation": {
                "tasks": [{"name": t} for t in tasks],
            },
        }
        if self.benchmark_set:
            result["evaluation"]["benchmark_set"] = self.benchmark_set
        return result


@dataclass
class SweepExecutionConfig:
    """Execution constraints for sweep jobs."""

    time_limit: int = 1200


@dataclass
class SweepConfig:
    """Top-level sweep configuration.

    Defines the Cartesian product dimensions:
    models × recipes × launchers = total jobs.
    """

    models: list[str] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)
    launchers: list[str] = field(default_factory=list)
    eval: SweepEvalConfig = field(default_factory=SweepEvalConfig)
    execution: SweepExecutionConfig = field(default_factory=SweepExecutionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        """Load sweep config from YAML file.

        Supports two formats:
        - Legacy: eval config nested under sweep.eval
        - New: separate top-level evaluation: section with overrides
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        sweep = raw.get("sweep", raw)
        execution_raw = sweep.get("execution", {})

        # Support both: top-level "evaluation:" (new) and "sweep.eval:" (legacy)
        eval_raw = raw.get("evaluation", sweep.get("eval", {}))

        return cls(
            models=sweep.get("models", []),
            recipes=sweep.get("recipes", []),
            launchers=sweep.get("launchers", []),
            eval=SweepEvalConfig(
                engine=eval_raw.get("engine", eval_raw.get("backend", "vllm")),
                tensor_parallel_size=eval_raw.get("tensor_parallel_size", 8),
                tasks=eval_raw.get("tasks", []),
                benchmark_set=eval_raw.get("benchmark_set"),
                model_overrides=eval_raw.get("model_overrides", {}),
                launcher_overrides=eval_raw.get("launcher_overrides", {}),
            ),
            execution=SweepExecutionConfig(
                time_limit=execution_raw.get("time_limit", 1200),
            ),
        )

    @property
    def total_jobs(self) -> int:
        """Total number of jobs in the sweep (Cartesian product)."""
        return len(self.models) * len(self.recipes) * len(self.launchers)

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.models:
            errors.append("No models specified")
        if not self.recipes:
            errors.append("No recipes specified")
        if not self.launchers:
            errors.append("No launchers specified")
        if not self.eval.tasks:
            errors.append("No eval tasks specified")
        return errors
