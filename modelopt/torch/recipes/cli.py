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

"""CLI entry point: modelopt-recipes validate/dry-run/info."""

from __future__ import annotations

import argparse
import sys


def main():
    """CLI entry point for recipe validation, dry-run, and info commands."""
    parser = argparse.ArgumentParser(
        prog="modelopt-recipes",
        description="Recipe system and sweep controller for NVIDIA Model Optimizer",
    )
    sub = parser.add_subparsers(dest="command")

    # validate
    val = sub.add_parser("validate", help="Validate a recipe YAML")
    val.add_argument("recipe", help="Path to recipe YAML")

    # dry-run (sweep)
    dr = sub.add_parser("dry-run", help="Dry-run a sweep config")
    dr.add_argument("--config", required=True, help="Sweep config YAML")
    dr.add_argument("--verbose", action="store_true")
    dr.add_argument("--output", help="Export jobs to JSON")

    # plan
    pl = sub.add_parser("plan", help="Show execution plan for a recipe")
    pl.add_argument("recipe", help="Path to recipe YAML")
    pl.add_argument("--verbose", action="store_true", help="Show resolved configs")

    # info
    sub.add_parser("info", help="Show preset source and available presets")

    args = parser.parse_args()

    if args.command == "validate":
        from modelopt.torch.recipes import load_recipe

        try:
            result = load_recipe(args.recipe)
            keys = list(result.keys())
            print(f"Valid. Resolved keys: {keys}")
        except Exception as e:
            print(f"Invalid: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "dry-run":
        from modelopt.torch.recipes.experiment import SweepConfig, SweepController

        config = SweepConfig.from_yaml(args.config)
        errors = config.validate()
        if errors:
            print("Sweep config validation failed:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)

        controller = SweepController(config)
        print(controller.dry_run(verbose=args.verbose))

        if args.output:
            controller.export_jobs(args.output)
            print(f"\nJobs exported to: {args.output}")

    elif args.command == "plan":
        from modelopt.torch.recipes.pipeline import load_and_plan

        try:
            plan = load_and_plan(args.recipe)
            print(plan.dry_run(verbose=args.verbose))
        except Exception as e:
            print(f"Plan failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "info":
        from modelopt.torch.recipes.schema.presets import get_preset_source, list_presets

        print(f"Preset source: {get_preset_source()}")
        presets = list_presets()
        print(f"Available presets ({len(presets)}):")
        for name in presets:
            print(f"  - {name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
