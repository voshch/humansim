import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from run_counterfactual import (
    DEFAULT_REGISTRY,
    ROBOT_KIND,
    SCRIPT_DIR,
    TRIAL_SELECTIONS,
    filter_trials,
    read_unique_failing_trials,
    select_trials,
)


DEFAULT_FACTUAL_DIR = SCRIPT_DIR / "data" / "nav_factual_bags"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "nav_counterfactual_bags"
DEFAULT_MANIFEST = (
    SCRIPT_DIR / "data" / "metrics" / "nav_counterfactual_manifest.csv"
)
GOAL_RADIUS_M = 0.5


@dataclass(frozen=True)
class HumanObservation:
    agent_id: int
    first_seen_s: float
    min_robot_distance_m: float


@dataclass(frozen=True)
class FactualTrace:
    robot_id: int
    minimum_goal_distance_m: float
    succeeded: bool
    humans: tuple[HumanObservation, ...]


@dataclass(frozen=True)
class NavTrialPlan:
    trial: dict[str, str]
    factual_bag_name: str
    trace: FactualTrace
    targets: tuple[HumanObservation, ...]


def _trial_slug(prefix: str, trial: dict[str, str]) -> str:
    return (
        f"{prefix}_{trial['scenario']}_seed{trial['seed']}"
        f"_planner{trial['ped_planner']}_robot{trial['robot_policy']}"
    )


def factual_bag_name(trial: dict[str, str]) -> str:
    return _trial_slug("factual", trial)


def counterfactual_bag_name(
    trial: dict[str, str],
    target_agent_id: int,
) -> str:
    return f"{_trial_slug('cfnav', trial)}_target{target_agent_id}"


def load_robot_goal(scenario_path: Path) -> tuple[int, tuple[float, float]]:
    with scenario_path.open("r", encoding="utf-8") as scenario_file:
        scenario = yaml.safe_load(scenario_file)

    agents = scenario.get("agents") or []
    robots = [
        agent for agent in agents
        if int(agent.get("kind", 0)) == ROBOT_KIND
    ]
    if len(robots) != 1:
        raise ValueError(
            f"expected exactly one robot in {scenario_path}, found {len(robots)}"
        )

    goals = robots[0].get("goal_sequence") or []
    if not goals:
        raise ValueError(f"robot has no goal_sequence in {scenario_path}")
    final_goal = goals[-1]
    return int(robots[0]["agent_id"]), (
        float(final_goal["x"]),
        float(final_goal["y"]),
    )


def analyze_factual_trace(trial_dir: Path, goal_radius_m: float = GOAL_RADIUS_M) -> FactualTrace:
    """Discover runtime humans and their closest factual robot distance."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError(
            "the 'rosbags' package is required inside the experiment environment"
        ) from error

    robot_id, goal = load_robot_goal(trial_dir / "scenario.yaml")
    first_timestamp_ns: int | None = None
    first_seen_s: dict[int, float] = {}
    minimum_robot_distance: dict[int, float] = {}
    minimum_goal_distance = math.inf

    with AnyReader([trial_dir / "bag"]) as reader:
        for connection, timestamp_ns, raw_data in reader.messages():
            if not connection.topic.endswith("/agent_states"):
                continue
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns

            message = reader.deserialize(raw_data, connection.msgtype)
            robots = [
                agent for agent in message.agents
                if int(agent.kind) == ROBOT_KIND
                and int(agent.agent_id) == robot_id
            ]
            humans = [
                agent for agent in message.agents
                if int(agent.kind) != ROBOT_KIND and int(agent.agent_id) > 0
            ]

            elapsed_s = (timestamp_ns - first_timestamp_ns) / 1e9
            for human in humans:
                first_seen_s.setdefault(int(human.agent_id), elapsed_s)

            if not robots:
                continue
            robot = robots[0]
            minimum_goal_distance = min(
                minimum_goal_distance,
                math.hypot(
                    float(robot.pose.x) - goal[0],
                    float(robot.pose.y) - goal[1],
                ),
            )
            for human in humans:
                distance = math.hypot(
                    float(human.pose.x) - float(robot.pose.x),
                    float(human.pose.y) - float(robot.pose.y),
                )
                agent_id = int(human.agent_id)
                minimum_robot_distance[agent_id] = min(
                    minimum_robot_distance.get(agent_id, math.inf),
                    distance,
                )

    if not math.isfinite(minimum_goal_distance):
        raise ValueError(f"robot {robot_id} has no recorded states in {trial_dir}")
    if not first_seen_s:
        raise ValueError(f"no runtime human IDs found in {trial_dir}")

    humans = tuple(
        sorted(
            (
                HumanObservation(
                    agent_id=agent_id,
                    first_seen_s=seen_s,
                    min_robot_distance_m=minimum_robot_distance.get(
                        agent_id,
                        math.inf,
                    ),
                )
                for agent_id, seen_s in first_seen_s.items()
            ),
            key=lambda observation: (
                observation.min_robot_distance_m,
                observation.first_seen_s,
                observation.agent_id,
            ),
        )
    )
    return FactualTrace(
        robot_id=robot_id,
        minimum_goal_distance_m=minimum_goal_distance,
        succeeded=minimum_goal_distance <= goal_radius_m,
        humans=humans,
    )


def select_target_humans(trace: FactualTrace, max_robot_distance_m: float | None, max_targets: int) -> tuple[HumanObservation, ...]:
    targets = list(trace.humans)
    if max_robot_distance_m is not None:
        targets = [
            target for target in targets
            if target.min_robot_distance_m <= max_robot_distance_m
        ]
    if max_targets > 0:
        targets = targets[:max_targets]
    return tuple(targets)


def build_launch_command(scenario_path: Path, trial: dict[str, str], bag_path: Path, target_agent_id: int = 0, counterfactual_planner: str = "") -> list[str]:
    if bool(target_agent_id) != bool(counterfactual_planner):
        raise ValueError(
            "target_agent_id and counterfactual_planner must either both be "
            "set or both be disabled"
        )

    command = [
        "ros2",
        "launch",
        "arena_humansim",
        "arena_humansim.launch.py",
        f"scenario:={scenario_path}",
        f"local_planner:={trial['ped_planner']}",
        "force_local_planner:=false",
        f"robot_policy:={trial['robot_policy']}",
        f"seed:={trial['seed']}",
    ]
    if target_agent_id:
        command.extend(
            [
                f"counterfactual_target_agent_id:={target_agent_id}",
                f"counterfactual_planner:={counterfactual_planner}",
            ]
        )
    command.extend(
        [
            f"record_dir:={bag_path}",
            "record:=true",
            "rtf:=0.0",
            "render:=false",
            "rviz:=false",
            "markers:=0",
            "robot_shutdown:=true",
        ]
    )
    return command


def launch_trial(
    command: list[str],
    bag_path: Path,
    clean_env: dict[str, str],
) -> None:
    if (bag_path / "bag" / "metadata.yaml").is_file():
        print(f"Already complete; reusing {bag_path.name}")
        return
    if bag_path.exists():
        raise RuntimeError(f"incomplete output already exists: {bag_path}")

    try:
        subprocess.run(
            command,
            check=True,
            env=clean_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr[-2000:] if error.stderr else "(no stderr)"
        raise RuntimeError(
            f"simulation failed for {bag_path.name}:\n{detail}"
        ) from error


def verify_counterfactual_policies(trial_dir: Path, target_agent_id: int, factual_planner: str, counterfactual_planner: str) -> None:
    """Confirm the target changed and every observed non-target human did not."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError("the 'rosbags' package is required for verification") from error

    seen_target = False
    with AnyReader([trial_dir / "bag"]) as reader:
        for connection, _timestamp_ns, raw_data in reader.messages():
            if not connection.topic.endswith("/agent_states"):
                continue
            message = reader.deserialize(raw_data, connection.msgtype)
            for agent in message.agents:
                if int(agent.kind) == ROBOT_KIND:
                    continue
                agent_id = int(agent.agent_id)
                expected = (
                    counterfactual_planner
                    if agent_id == target_agent_id
                    else factual_planner
                )
                if str(agent.policy) != expected:
                    raise ValueError(
                        f"human {agent_id} used policy {agent.policy!r}; "
                        f"expected {expected!r}"
                    )
                seen_target |= agent_id == target_agent_id

    if not seen_target:
        raise ValueError(
            f"counterfactual target human {target_agent_id} never appeared"
        )


def write_manifest(path: Path, plans: list[NavTrialPlan], baseline_planner: str) -> None:
    fieldnames = [
        "scenario",
        "seed",
        "ped_planner",
        "robot_policy",
        "cause",
        "target_agent_id",
        "target_rank",
        "baseline_planner",
        "first_seen_s",
        "factual_min_robot_distance_m",
        "factual_minimum_goal_distance_m",
        "factual_succeeded",
        "factual_bag",
        "counterfactual_bag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        for plan in plans:
            for rank, target in enumerate(plan.targets, start=1):
                writer.writerow(
                    {
                        "scenario": plan.trial["scenario"],
                        "seed": plan.trial["seed"],
                        "ped_planner": plan.trial["ped_planner"],
                        "robot_policy": plan.trial["robot_policy"],
                        "cause": plan.trial["cause"],
                        "target_agent_id": target.agent_id,
                        "target_rank": rank,
                        "baseline_planner": baseline_planner,
                        "first_seen_s": round(target.first_seen_s, 6),
                        "factual_min_robot_distance_m": round(
                            target.min_robot_distance_m,
                            6,
                        ),
                        "factual_minimum_goal_distance_m": round(
                            plan.trace.minimum_goal_distance_m,
                            6,
                        ),
                        "factual_succeeded": plan.trace.succeeded,
                        "factual_bag": plan.factual_bag_name,
                        "counterfactual_bag": counterfactual_bag_name(
                            plan.trial,
                            target.agent_id,
                        ),
                    }
                )


def group_trial_candidates(trials: list[dict[str, str]], selection: str) -> list[list[dict[str, str]]]:
    """Group ordered registry candidates so reduced modes can retry a group."""
    ordered = select_trials(trials, "all")
    if selection == "all":
        return [[trial] for trial in ordered]

    if selection == "scenario-seed":
        key_columns = ("scenario", "seed")
    elif selection == "scenario":
        key_columns = ("scenario",)
    else:
        raise ValueError(f"unsupported trial selection: {selection!r}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for trial in ordered:
        key = tuple(trial[column] for column in key_columns)
        groups.setdefault(key, []).append(trial)
    return list(groups.values())


def run_nav_counterfactuals(
    registry_path: Path,
    factual_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    baseline_planner: str,
    requested_scenarios: set[str] | None,
    requested_seeds: set[str] | None,
    requested_ped_planners: set[str] | None,
    requested_robot_policies: set[str] | None,
    trial_selection: str,
    max_robot_distance_m: float | None,
    max_targets: int,
    factual_only: bool,
    allow_factual_success: bool,
    dry_run: bool,
) -> int:
    try:
        trials = [
            trial
            for trial in filter_trials(
                read_unique_failing_trials(registry_path),
                requested_scenarios,
                requested_seeds,
                requested_ped_planners,
                requested_robot_policies,
            )
            if trial.get("bucket") == "nav"
        ]
        trial_groups = group_trial_candidates(trials, trial_selection)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Error: could not select nav failures: {error}", file=sys.stderr)
        return 1

    if not trial_groups:
        print("Error: no matching nav failure trials.", file=sys.stderr)
        return 1

    print(
        f"Selected {len(trial_groups)} groups from {len(trials)} candidate "
        f"nav failure trials using '{trial_selection}'."
    )
    if max_robot_distance_m is not None or max_targets > 0:
        limit_description = []
        if max_robot_distance_m is not None:
            limit_description.append(
                f"factual robot distance <= {max_robot_distance_m:g} m"
            )
        if max_targets > 0:
            limit_description.append(f"nearest {max_targets} per trial")
        print(
            "Exploratory target selection: " + ", ".join(limit_description)
        )

    clean_env = os.environ.copy()
    clean_env["PYTHONWARNINGS"] = "ignore"
    clean_env["RCUTILS_LOGGING_SEVERITY_THRESHOLD"] = "WARN"
    plans: list[NavTrialPlan] = []
    missing_factual = 0

    for index, candidates in enumerate(trial_groups, start=1):
        prepared = False
        for attempt, trial in enumerate(candidates, start=1):
            scenario_path = SCRIPT_DIR / "scenarios" / f"{trial['scenario']}.yaml"
            if not scenario_path.is_file():
                print(f"Error: scenario not found: {scenario_path}", file=sys.stderr)
                return 1
            try:
                with scenario_path.open("r", encoding="utf-8") as scenario_file:
                    scenario = yaml.safe_load(scenario_file)
                if not (scenario.get("flow") or {}).get("sources"):
                    raise ValueError("scenario has no dynamic flow sources")

                factual_name = factual_bag_name(trial)
                factual_path = factual_dir / factual_name
                factual_complete = (
                    factual_path / "bag" / "metadata.yaml"
                ).is_file()
                attempt_text = (
                    f" | candidate={attempt}/{len(candidates)}"
                    if len(candidates) > 1
                    else ""
                )
                print(
                    f"[factual {index}/{len(trial_groups)}] "
                    f"{trial['scenario']} | seed={trial['seed']} | "
                    f"planner={trial['ped_planner']} | "
                    f"robot={trial['robot_policy']}{attempt_text}"
                )
                if dry_run and not factual_complete:
                    print("  factual bag missing; runtime target count is unknown")
                    missing_factual += 1
                    break
                if not dry_run:
                    factual_dir.mkdir(parents=True, exist_ok=True)
                    launch_trial(
                        build_launch_command(scenario_path, trial, factual_path),
                        factual_path,
                        clean_env,
                    )

                trace = analyze_factual_trace(factual_path)
                if trace.succeeded and not allow_factual_success:
                    if trial_selection != "all" and attempt < len(candidates):
                        print(
                            "  registered failure now succeeds "
                            f"({trace.minimum_goal_distance_m:.3f} m); "
                            "trying the next candidate"
                        )
                        continue
                    raise ValueError(
                        "the reproduced factual trial reached the robot goal "
                        f"({trace.minimum_goal_distance_m:.3f} m), and no "
                        "untried candidate remains for this selection group"
                    )

                targets = select_target_humans(trace, max_robot_distance_m, max_targets)
                
                if not targets:
                    raise ValueError("no runtime humans satisfy the target filters")
                print(
                    f"  discovered {len(trace.humans)} humans; "
                    f"selected {len(targets)} counterfactual targets"
                )
                plans.append(
                    NavTrialPlan(
                        trial=trial,
                        factual_bag_name=factual_name,
                        trace=trace,
                        targets=targets,
                    )
                )
                prepared = True
                break
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                yaml.YAMLError,
            ) as error:
                print(
                    f"Error preparing nav trial {trial['scenario']} "
                    f"seed={trial['seed']}: {error}",
                    file=sys.stderr,
                )
                return 1

        if not prepared and not dry_run:
            print(
                f"Error: no reproducible factual failure found for "
                f"{candidates[0]['scenario']}.",
                file=sys.stderr,
            )
            return 1

    total_counterfactuals = sum(len(plan.targets) for plan in plans)
    if dry_run:
        print(
            f"Dry run: {total_counterfactuals} counterfactual runs can be "
            f"planned from existing factual bags; {missing_factual} factual "
            "runs are still required."
        )
        return 0

    write_manifest(manifest_path, plans, baseline_planner)
    print(
        f"Manifest contains {total_counterfactuals} runtime-human "
        f"interventions: {manifest_path}"
    )
    if factual_only:
        print("Factual preparation complete; no counterfactuals were launched.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    run_number = 0
    for plan in plans:
        scenario_path = (
            SCRIPT_DIR / "scenarios" / f"{plan.trial['scenario']}.yaml"
        )
        for target in plan.targets:
            run_number += 1
            bag_name = counterfactual_bag_name(
                plan.trial,
                target.agent_id,
            )
            bag_path = output_dir / bag_name
            print(
                f"[counterfactual {run_number}/{total_counterfactuals}] "
                f"{plan.trial['scenario']} | seed={plan.trial['seed']} | "
                f"planner={plan.trial['ped_planner']} | "
                f"robot={plan.trial['robot_policy']} | "
                f"runtime human={target.agent_id} -> {baseline_planner}"
            )
            try:
                launch_trial(
                    build_launch_command(
                        scenario_path,
                        plan.trial,
                        bag_path,
                        target_agent_id=target.agent_id,
                        counterfactual_planner=baseline_planner,
                    ),
                    bag_path,
                    clean_env,
                )
                verify_counterfactual_policies(
                    bag_path,
                    target.agent_id,
                    plan.trial["ped_planner"],
                    baseline_planner,
                )
            except (ValueError, RuntimeError) as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1

    print(f"Nav counterfactual sweep complete: {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover flow-spawned humans from factual nav bags and intervene "
            "on one runtime human at a time."
        )
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--factual-dir", type=Path, default=DEFAULT_FACTUAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-planner", default="straight")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--ped-planners", nargs="+", default=None)
    parser.add_argument("--robot-policies", nargs="+", default=None)
    parser.add_argument(
        "--trial-selection",
        choices=TRIAL_SELECTIONS,
        default="all",
    )
    parser.add_argument(
        "--max-robot-distance",
        type=float,
        default=None,
        help=(
            "exploratory filter: target only factual humans that came within "
            "this many metres of the robot"
        ),
    )
    parser.add_argument(
        "--max-targets-per-trial",
        type=int,
        default=0,
        help=(
            "exploratory limit after sorting humans by factual robot proximity; "
            "0 means all humans"
        ),
    )
    parser.add_argument(
        "--factual-only",
        action="store_true",
        help="record/analyze factual trials and write the manifest, then stop",
    )
    parser.add_argument(
        "--allow-factual-success",
        action="store_true",
        help="debugging only: do not reject a registry failure that now succeeds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect existing factual bags without launching simulations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_robot_distance is not None and args.max_robot_distance <= 0:
        print("Error: --max-robot-distance must be positive.", file=sys.stderr)
        return 1
    if args.max_targets_per_trial < 0:
        print("Error: --max-targets-per-trial cannot be negative.", file=sys.stderr)
        return 1

    return run_nav_counterfactuals(
        registry_path=args.registry.expanduser().resolve(),
        factual_dir=args.factual_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        baseline_planner=args.baseline_planner,
        requested_scenarios=set(args.scenarios) if args.scenarios else None,
        requested_seeds=set(args.seeds) if args.seeds else None,
        requested_ped_planners=(
            set(args.ped_planners) if args.ped_planners else None
        ),
        requested_robot_policies=(
            set(args.robot_policies) if args.robot_policies else None
        ),
        trial_selection=args.trial_selection,
        max_robot_distance_m=args.max_robot_distance,
        max_targets=args.max_targets_per_trial,
        factual_only=args.factual_only,
        allow_factual_success=args.allow_factual_success,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
