#!/usr/bin/env python3
"""Score human-only counterfactual bags using the PEDS success definition."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BAGS_DIR = SCRIPT_DIR / "data" / "counterfactual_bags_human_only"
DEFAULT_REGISTRY = SCRIPT_DIR / "data" / "metrics" / "failure_registry.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "metrics" / "causal_scores_human_only.csv"

GOAL_RADIUS_M = 0.5
ROBOT_KIND = 1

BAG_NAME_PATTERN = re.compile(
    r"^cf_(?P<scenario>.+)_seed(?P<seed>\d+)"
    r"_planner(?P<ped_planner>[^_]+)"
    r"_robot(?P<robot_policy>[^_]+)"
    r"_target(?P<target_agent_id>-?\d+)$"
)


def parse_bag_name(bag_name: str) -> dict[str, str | int]:
    match = BAG_NAME_PATTERN.fullmatch(bag_name)
    if match is None:
        raise ValueError(f"unsupported counterfactual bag name: {bag_name}")

    parsed: dict[str, str | int] = match.groupdict()
    parsed["seed"] = int(parsed["seed"])
    parsed["target_agent_id"] = int(parsed["target_agent_id"])
    return parsed


def get_bag_duration_s(trial_dir: Path) -> float:
    metadata_path = trial_dir / "bag" / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"rosbag metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    duration_ns = metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"]
    return float(duration_ns) / 1e9


def load_counterfactual_definition(
    trial_dir: Path,
    target_agent_id: int,
) -> tuple[int, tuple[float, float], str]:
    """Return robot ID, final robot goal, and target human planner."""
    scenario_path = trial_dir / "scenario.yaml"
    if not scenario_path.is_file():
        raise FileNotFoundError(f"recorded scenario snapshot not found: {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as scenario_file:
        scenario = yaml.safe_load(scenario_file)

    agents = scenario.get("agents") or []
    robots = [
        agent
        for agent in agents
        if int(agent.get("kind", 0)) == ROBOT_KIND
    ]
    if len(robots) != 1:
        raise ValueError(
            f"expected exactly one robot in {scenario_path}, found {len(robots)}"
        )

    robot = robots[0]
    goals = robot.get("goal_sequence") or []
    if not goals:
        raise ValueError(f"robot has no goal_sequence in {scenario_path}")
    final_goal = goals[-1]

    target_humans = [
        agent
        for agent in agents
        if int(agent.get("kind", 0)) != ROBOT_KIND
        and int(agent.get("agent_id", 0)) == target_agent_id
    ]
    if len(target_humans) != 1:
        raise ValueError(
            f"target {target_agent_id} is not exactly one explicit human in "
            f"{scenario_path}"
        )

    target_planner = str(target_humans[0].get("policy", ""))
    if not target_planner:
        raise ValueError(
            f"target human {target_agent_id} has no planner in {scenario_path}"
        )

    return (
        int(robot["agent_id"]),
        (float(final_goal["x"]), float(final_goal["y"])),
        target_planner,
    )


def score_robot_goal(
    bag_dir: Path,
    robot_id: int,
    goal: tuple[float, float],
    goal_radius_m: float = GOAL_RADIUS_M,
) -> tuple[bool, float]:
    """Apply the paper's success rule to recorded robot positions."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError(
            "the 'rosbags' package is required to score counterfactual bags"
        ) from error

    minimum_distance = math.inf
    with AnyReader([bag_dir]) as reader:
        for connection, _timestamp, raw_data in reader.messages():
            if not connection.topic.endswith("/agent_states"):
                continue
            message = reader.deserialize(raw_data, connection.msgtype)
            for agent in message.agents:
                if int(agent.agent_id) != robot_id:
                    continue
                distance = math.hypot(
                    float(agent.pose.x) - goal[0],
                    float(agent.pose.y) - goal[1],
                )
                minimum_distance = min(minimum_distance, distance)

    if not math.isfinite(minimum_distance):
        raise ValueError(f"robot {robot_id} has no recorded states in {bag_dir}")
    return minimum_distance <= goal_radius_m, minimum_distance


def read_failure_causes(
    registry_path: Path,
) -> dict[tuple[str, int, str, str], str]:
    with registry_path.open("r", encoding="utf-8", newline="") as registry_file:
        reader = csv.DictReader(registry_file)
        if reader.fieldnames is None:
            raise ValueError("failure registry has no header")

        required = {"scenario", "seed", "ped_planner", "robot_policy", "cause"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"failure registry is missing required columns: {sorted(missing)}"
            )

        causes: dict[tuple[str, int, str, str], str] = {}
        for row in reader:
            key = (
                row["scenario"],
                int(row["seed"]),
                row["ped_planner"],
                row["robot_policy"],
            )
            cause = row["cause"]
            previous = causes.setdefault(key, cause)
            if previous != cause:
                raise ValueError(
                    f"conflicting failure causes for trial {key}: "
                    f"{previous!r} and {cause!r}"
                )
    return causes


def score_counterfactuals(
    bags_dir: Path = DEFAULT_BAGS_DIR,
    registry_path: Path = DEFAULT_REGISTRY,
    output_path: Path = DEFAULT_OUTPUT,
    goal_radius_m: float = GOAL_RADIUS_M,
) -> int:
    if not bags_dir.is_dir():
        print(f"Error: counterfactual bag directory not found: {bags_dir}", file=sys.stderr)
        return 1
    if not registry_path.is_file():
        print(f"Error: failure registry not found: {registry_path}", file=sys.stderr)
        return 1

    try:
        causes = read_failure_causes(registry_path)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Error: could not read failure registry: {error}", file=sys.stderr)
        return 1

    trial_dirs = sorted(path for path in bags_dir.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for trial_dir in trial_dirs:
        try:
            info = parse_bag_name(trial_dir.name)
            scenario = str(info["scenario"])
            seed = int(info["seed"])
            ped_planner = str(info["ped_planner"])
            robot_policy = str(info["robot_policy"])
            target_agent_id = int(info["target_agent_id"])

            robot_id, goal, counterfactual_planner = load_counterfactual_definition(
                trial_dir,
                target_agent_id,
            )
            succeeded, minimum_goal_distance_m = score_robot_goal(
                trial_dir / "bag",
                robot_id,
                goal,
                goal_radius_m,
            )
            duration_s = get_bag_duration_s(trial_dir)
            key = (scenario, seed, ped_planner, robot_policy)

            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "ped_planner": ped_planner,
                    "robot_policy": robot_policy,
                    "target_agent_id": target_agent_id,
                    "counterfactual_planner": counterfactual_planner,
                    "duration_s": round(duration_s, 3),
                    "minimum_goal_distance_m": round(minimum_goal_distance_m, 6),
                    "succeeded": succeeded,
                    "causal": succeeded,
                    "factual_cause": causes.get(key, "unknown"),
                }
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as error:
            skipped.append((trial_dir.name, str(error)))

    if not rows:
        print("Error: no valid human-only counterfactual bags were scored.", file=sys.stderr)
        for bag_name, reason in skipped[:10]:
            print(f"  {bag_name}: {reason}", file=sys.stderr)
        return 1

    rows.sort(
        key=lambda row: (
            row["scenario"],
            row["seed"],
            row["ped_planner"],
            row["robot_policy"],
            row["target_agent_id"],
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
            row["target_agent_id"] for row in group if bool(row["causal"])
        ]
        scenario, seed, planner, robot_policy = key
        print(
            f"{scenario} | seed={seed} | planner={planner} | robot={robot_policy}: "
            f"{len(causal_ids)}/{len(group)} causal humans {causal_ids}"
        )

    if skipped:
        print(f"Warning: skipped {len(skipped)} invalid bag directories.")
        for bag_name, reason in skipped[:10]:
            print(f"  {bag_name}: {reason}")

    print(f"Scored {len(rows)} counterfactual runs: {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score human-only counterfactuals using minimum robot-to-goal "
            "distance, matching the PEDS paper."
        )
    )
    parser.add_argument("--bags-dir", type=Path, default=DEFAULT_BAGS_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=GOAL_RADIUS_M,
        help=f"success radius in metres (default: {GOAL_RADIUS_M})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.goal_radius <= 0:
        print("Error: --goal-radius must be positive.", file=sys.stderr)
        return 1
    return score_counterfactuals(
        bags_dir=args.bags_dir.expanduser().resolve(),
        registry_path=args.registry.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        goal_radius_m=args.goal_radius,
    )


if __name__ == "__main__":
    raise SystemExit(main())
