from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from run_nav_counterfactual import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    GOAL_RADIUS_M,
    load_robot_goal,
    verify_counterfactual_policies,
)
from score_cf import (
    DEFAULT_REGISTRY,
    get_bag_duration_s,
    read_failure_causes,
    score_robot_goal,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "metrics" / "nav_causal_scores.csv"
NAME_PATTERN = re.compile(
    r"^cfnav_(?P<scenario>.+)_seed(?P<seed>\d+)"
    r"_planner(?P<ped_planner>[^_]+)"
    r"_robot(?P<robot_policy>[^_]+)"
    r"_target(?P<target_agent_id>\d+)$"
)


def parse_bag_name(name: str) -> dict[str, str | int]:
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unsupported nav counterfactual bag name: {name}")
    parsed: dict[str, str | int] = match.groupdict()
    parsed["seed"] = int(parsed["seed"])
    parsed["target_agent_id"] = int(parsed["target_agent_id"])
    return parsed


def read_manifest(path: Path) -> dict[tuple[str, int, str, str, int], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {
            "scenario",
            "seed",
            "ped_planner",
            "robot_policy",
            "target_agent_id",
            "target_rank",
            "baseline_planner",
            "factual_min_robot_distance_m",
            "factual_minimum_goal_distance_m",
            "factual_succeeded",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"nav manifest is missing columns: {sorted(missing)}"
            )

        rows: dict[tuple[str, int, str, str, int], dict[str, str]] = {}
        for row in reader:
            key = (
                row["scenario"],
                int(row["seed"]),
                row["ped_planner"],
                row["robot_policy"],
                int(row["target_agent_id"]),
            )
            if key in rows:
                raise ValueError(f"duplicate nav manifest row for {key}")
            rows[key] = row
    return rows


def score_nav_counterfactuals(bags_dir: Path, manifest_path: Path, registry_path: Path, output_path: Path, goal_radius_m: float) -> int:
    try:
        manifest = read_manifest(manifest_path)
        causes = read_failure_causes(registry_path)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Error reading inputs: {error}", file=sys.stderr)
        return 1

    if not bags_dir.is_dir():
        print(f"Error: nav bag directory not found: {bags_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for trial_dir in sorted(path for path in bags_dir.iterdir() if path.is_dir()):
        try:
            info = parse_bag_name(trial_dir.name)
            scenario = str(info["scenario"])
            seed = int(info["seed"])
            ped_planner = str(info["ped_planner"])
            robot_policy = str(info["robot_policy"])
            target_agent_id = int(info["target_agent_id"])
            key = (
                scenario,
                seed,
                ped_planner,
                robot_policy,
                target_agent_id,
            )
            manifest_row = manifest.get(key)
            if manifest_row is None:
                raise ValueError("bag has no matching row in the nav manifest")

            robot_id, goal = load_robot_goal(trial_dir / "scenario.yaml")
            baseline_planner = manifest_row["baseline_planner"]
            verify_counterfactual_policies(
                trial_dir,
                target_agent_id,
                ped_planner,
                baseline_planner,
            )
            succeeded, minimum_goal_distance_m = score_robot_goal(
                trial_dir / "bag",
                robot_id,
                goal,
                goal_radius_m,
            )
            factual_key = (scenario, seed, ped_planner, robot_policy)
            factual_succeeded = (
                manifest_row["factual_succeeded"].lower() == "true"
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "ped_planner": ped_planner,
                    "robot_policy": robot_policy,
                    "target_agent_id": target_agent_id,
                    "target_rank": int(manifest_row["target_rank"]),
                    "baseline_planner": baseline_planner,
                    "factual_min_robot_distance_m": float(
                        manifest_row["factual_min_robot_distance_m"]
                    ),
                    "factual_minimum_goal_distance_m": float(
                        manifest_row["factual_minimum_goal_distance_m"]
                    ),
                    "counterfactual_minimum_goal_distance_m": round(
                        minimum_goal_distance_m,
                        6,
                    ),
                    "duration_s": round(get_bag_duration_s(trial_dir), 3),
                    "factual_succeeded": factual_succeeded,
                    "counterfactual_succeeded": succeeded,
                    "causal": not factual_succeeded and succeeded,
                    "factual_cause": causes.get(factual_key, "unknown"),
                }
            )
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            yaml.YAMLError,
        ) as error:
            skipped.append((trial_dir.name, str(error)))

    if not rows:
        print("Error: no valid nav counterfactual bags were scored.", file=sys.stderr)
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}", file=sys.stderr)
        return 1

    rows.sort(
        key=lambda row: (
            row["scenario"],
            row["seed"],
            row["ped_planner"],
            row["robot_policy"],
            row["target_rank"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["scenario"],
                row["seed"],
                row["ped_planner"],
                row["robot_policy"],
            )
        ].append(row)
    for key, group in grouped.items():
        causal_ids = [
            row["target_agent_id"]
            for row in group
            if bool(row["causal"])
        ]
        scenario, seed, planner, robot_policy = key
        print(
            f"{scenario} | seed={seed} | planner={planner} | "
            f"robot={robot_policy}: {len(causal_ids)}/{len(group)} "
            f"causal runtime humans {causal_ids}"
        )

    if skipped:
        print(f"Warning: skipped {len(skipped)} invalid bag directories.")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
    print(f"Scored {len(rows)} nav counterfactual runs: {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score dynamic-flow nav counterfactuals using robot success."
    )
    parser.add_argument("--bags-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=GOAL_RADIUS_M,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.goal_radius <= 0:
        print("Error: --goal-radius must be positive.", file=sys.stderr)
        return 1
    return score_nav_counterfactuals(
        bags_dir=args.bags_dir.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        registry_path=args.registry.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        goal_radius_m=args.goal_radius,
    )


if __name__ == "__main__":
    raise SystemExit(main())
