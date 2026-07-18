from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = SCRIPT_DIR / "data" / "metrics" / "failure_registry.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "counterfactual_bags_human_only"

ROBOT_KIND = 1
TRIAL_KEY_COLUMNS = ("scenario", "seed", "ped_planner", "robot_policy")
TRIAL_SELECTIONS = ("all", "scenario-seed", "scenario")


def load_scenario(scenario_name: str, scenarios_dir: Path = SCRIPT_DIR / "scenarios") -> dict[str, Any]:
    scenario_path = scenarios_dir / f"{scenario_name}.yaml"
    if not scenario_path.is_file():
        raise FileNotFoundError(f"scenario YAML not found: {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as scenario_file:
        scenario = yaml.safe_load(scenario_file)

    if not isinstance(scenario, dict):
        raise ValueError(f"scenario YAML is not a mapping: {scenario_path}")
    return scenario


def get_human_agent_ids(scenario: dict[str, Any]) -> list[int]:
    """Return stable IDs for explicit human agents, excluding every robot."""
    agents = scenario.get("agents") or []
    if not isinstance(agents, list):
        raise ValueError("scenario 'agents' must be a list")

    human_ids: list[int] = []
    for agent in agents:
        if not isinstance(agent, dict):
            raise ValueError("every scenario agent must be a mapping")
        if int(agent.get("kind", 0)) == ROBOT_KIND:
            continue

        agent_id = int(agent.get("agent_id", 0))
        if agent_id <= 0:
            raise ValueError(
                "counterfactual human agents need explicit positive agent_id values"
            )
        human_ids.append(agent_id)

    if len(human_ids) != len(set(human_ids)):
        raise ValueError("scenario contains duplicate human agent IDs")
    return human_ids


def make_counterfactual_scenario(factual_scenario: dict[str, Any], factual_planner: str, target_agent_id: int, baseline_planner: str = "straight") -> dict[str, Any]:
    """Set all humans factual except one target; leave robot entries untouched."""
    counterfactual = copy.deepcopy(factual_scenario)
    human_ids = get_human_agent_ids(counterfactual)
    if target_agent_id not in human_ids:
        raise ValueError(
            f"target {target_agent_id} is not an explicit human agent; "
            f"available human IDs: {human_ids}"
        )

    for agent in counterfactual["agents"]:
        if int(agent.get("kind", 0)) == ROBOT_KIND:
            continue
        agent["policy"] = (
            baseline_planner
            if int(agent["agent_id"]) == target_agent_id
            else factual_planner
        )

    return counterfactual


def read_unique_failing_trials(registry_path: Path) -> list[dict[str, str]]:
    with registry_path.open("r", encoding="utf-8", newline="") as registry_file:
        reader = csv.DictReader(registry_file)
        if reader.fieldnames is None:
            raise ValueError("failure registry has no header")

        missing = set((*TRIAL_KEY_COLUMNS, "cause")) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"failure registry is missing required columns: {sorted(missing)}"
            )

        unique_trials: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()
        for row in reader:
            if row["robot_policy"] == "drlvo":
                continue
            if row["ped_planner"] == "straight":
                continue

            key = tuple(row[column] for column in TRIAL_KEY_COLUMNS)
            if key in seen:
                continue
            seen.add(key)
            unique_trials.append(row)

    return unique_trials


def build_launch_command(scenario_path: Path, factual_planner: str, robot_policy: str, seed: str, bag_path: Path) -> list[str]:
    """Build a command using launch arguments that actually exist."""
    return [
        "ros2",
        "launch",
        "arena_humansim",
        "arena_humansim.launch.py",
        f"scenario:={scenario_path}",
        f"local_planner:={factual_planner}",
        "force_local_planner:=false",
        f"robot_policy:={robot_policy}",
        f"seed:={seed}",
        f"record_dir:={bag_path}",
        "record:=true",
        "rtf:=0.0",
        "render:=false",
        "rviz:=false",
        "markers:=0",
        "robot_shutdown:=true",
    ]


def bag_name_for_trial(scenario: str, seed: str, factual_planner: str, robot_policy: str, target_agent_id: int) -> str:
    return (
        f"cf_{scenario}_seed{seed}_planner{factual_planner}"
        f"_robot{robot_policy}_target{target_agent_id}"
    )


def filter_scenarios(trials: Iterable[dict[str, str]], requested_scenarios: set[str] | None) -> list[dict[str, str]]:
    if requested_scenarios is None:
        return list(trials)
    return [row for row in trials if row["scenario"] in requested_scenarios]


def filter_trials(
    trials: Iterable[dict[str, str]],
    requested_scenarios: set[str] | None = None,
    requested_seeds: set[str] | None = None,
    requested_ped_planners: set[str] | None = None,
    requested_robot_policies: set[str] | None = None,
) -> list[dict[str, str]]:
    filtered = filter_scenarios(trials, requested_scenarios)
    if requested_seeds is not None:
        filtered = [row for row in filtered if row["seed"] in requested_seeds]
    if requested_ped_planners is not None:
        filtered = [
            row for row in filtered
            if row["ped_planner"] in requested_ped_planners
        ]
    if requested_robot_policies is not None:
        filtered = [
            row for row in filtered
            if row["robot_policy"] in requested_robot_policies
        ]
    return filtered


def select_trials(trials: Iterable[dict[str, str]], selection: str = "all") -> list[dict[str, str]]:
    """Deterministically reduce the registry without arbitrary CSV row order."""
    if selection not in TRIAL_SELECTIONS:
        raise ValueError(
            f"trial selection must be one of {TRIAL_SELECTIONS}; got {selection!r}"
        )

    ordered = sorted(
        trials,
        key=lambda row: (
            row["scenario"],
            int(row["seed"]),
            row["ped_planner"],
            row["robot_policy"],
        ),
    )
    if selection == "all":
        return ordered

    key_columns = ("scenario", "seed") if selection == "scenario-seed" else ("scenario",)
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in ordered:
        key = tuple(row[column] for column in key_columns)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def run_counterfactual_sweep(
    registry_path: Path = DEFAULT_REGISTRY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    baseline_planner: str = "straight",
    requested_scenarios: set[str] | None = None,
    requested_seeds: set[str] | None = None,
    requested_ped_planners: set[str] | None = None,
    requested_robot_policies: set[str] | None = None,
    trial_selection: str = "all",
    dry_run: bool = False,
) -> int:
    if not registry_path.is_file():
        print(f"Error: registry not found: {registry_path}", file=sys.stderr)
        return 1

    try:
        filtered_trials = filter_trials(
            read_unique_failing_trials(registry_path),
            requested_scenarios,
            requested_seeds,
            requested_ped_planners,
            requested_robot_policies,
        )
        trials = select_trials(filtered_trials, trial_selection)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Error: could not read failure registry: {error}", file=sys.stderr)
        return 1

    if trial_selection != "all":
        print(
            f"Exploratory selection '{trial_selection}' retained "
            f"{len(trials)}/{len(filtered_trials)} filtered failing trials. "
        )

    scenario_cache: dict[str, dict[str, Any]] = {}
    human_ids_by_scenario: dict[str, list[int]] = {}
    skipped_scenarios: dict[str, str] = {}

    for scenario_name in sorted({row["scenario"] for row in trials}):
        try:
            scenario = load_scenario(scenario_name)
            human_ids = get_human_agent_ids(scenario)
            if not human_ids:
                if scenario.get("flow"):
                    skipped_scenarios[scenario_name] = (
                        "dynamic-flow humans require IDs from the factual trace"
                    )
                else:
                    skipped_scenarios[scenario_name] = "no explicit human agents"
                continue
            scenario_cache[scenario_name] = scenario
            human_ids_by_scenario[scenario_name] = human_ids
        except (OSError, ValueError, yaml.YAMLError) as error:
            skipped_scenarios[scenario_name] = str(error)

    supported_trials = [
        row for row in trials if row["scenario"] in human_ids_by_scenario
    ]
    total_runs = sum(
        len(human_ids_by_scenario[row["scenario"]]) for row in supported_trials
    )

    print(
        f"Planned {total_runs} human-only counterfactual runs from "
        f"{len(supported_trials)} failing trials."
    )
    for scenario_name in sorted(human_ids_by_scenario):
        scenario_trial_count = sum(
            row["scenario"] == scenario_name for row in supported_trials
        )
        if scenario_trial_count == 0:
            continue
        human_count = len(human_ids_by_scenario[scenario_name])
        print(
            f"  {scenario_name}: {scenario_trial_count} trials x "
            f"{human_count} humans = {scenario_trial_count * human_count} runs"
        )
    for scenario_name, reason in skipped_scenarios.items():
        print(f"Skipping {scenario_name}: {reason}")

    if not supported_trials:
        print("Error: no trials with explicit human agents are available.", file=sys.stderr)
        return 1

    if dry_run:
        print("Dry run complete; no simulations were launched.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_env = os.environ.copy()
    clean_env["PYTHONWARNINGS"] = "ignore"
    clean_env["RCUTILS_LOGGING_SEVERITY_THRESHOLD"] = "WARN"

    run_number = 0
    for trial in supported_trials:
        scenario_name = trial["scenario"]
        factual_planner = trial["ped_planner"]
        robot_policy = trial["robot_policy"]
        seed = trial["seed"]
        cause = trial["cause"]

        for target_agent_id in human_ids_by_scenario[scenario_name]:
            run_number += 1
            bag_name = bag_name_for_trial(
                scenario_name,
                seed,
                factual_planner,
                robot_policy,
                target_agent_id,
            )
            bag_path = output_dir / bag_name

            if (bag_path / "bag" / "metadata.yaml").is_file():
                print(f"[{run_number}/{total_runs}] Already complete; skipping {bag_name}")
                continue
            if bag_path.exists():
                print(
                    f"Error: incomplete output already exists: {bag_path}",
                    file=sys.stderr,
                )
                return 1

            counterfactual = make_counterfactual_scenario(
                scenario_cache[scenario_name],
                factual_planner,
                target_agent_id,
                baseline_planner,
            )

            print(
                f"[{run_number}/{total_runs}] {scenario_name} | seed={seed} | "
                f"planner={factual_planner} | robot={robot_policy} | "
                f"human={target_agent_id} -> {baseline_planner} | cause={cause}"
            )

            with tempfile.TemporaryDirectory(prefix="arena_counterfactual_") as temp_dir:
                scenario_path = Path(temp_dir) / f"{scenario_name}.yaml"
                with scenario_path.open("w", encoding="utf-8") as scenario_file:
                    yaml.safe_dump(
                        counterfactual,
                        scenario_file,
                        sort_keys=False,
                    )

                command = build_launch_command(
                    scenario_path,
                    factual_planner,
                    robot_policy,
                    seed,
                    bag_path,
                )
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
                    print(
                        f"Simulation failed for human {target_agent_id}.",
                        file=sys.stderr,
                    )
                    print(error.stderr[-2000:], file=sys.stderr)
                    return 1

    print(f"Counterfactual sweep complete. Bags saved to: {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one explicit human agent's planner at a time while "
            "preserving the robot and every other human."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"failure registry (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"new bag directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baseline-planner",
        default="straight",
        help="planner assigned only to the target human (default: straight)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="optional scenario-name allowlist",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="optional seed allowlist",
    )
    parser.add_argument(
        "--ped-planners",
        nargs="+",
        default=None,
        help="optional factual pedestrian-planner allowlist",
    )
    parser.add_argument(
        "--robot-policies",
        nargs="+",
        default=None,
        help="optional factual robot-policy allowlist",
    )
    parser.add_argument(
        "--trial-selection",
        choices=TRIAL_SELECTIONS,
        default="all",
        help=(
            "all: every failing configuration; scenario-seed: one deterministic "
            "configuration per scenario and seed; scenario: one per scenario"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and report the number of runs without launching ROS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_scenarios = set(args.scenarios) if args.scenarios else None
    return run_counterfactual_sweep(
        registry_path=args.registry.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        baseline_planner=args.baseline_planner,
        requested_scenarios=requested_scenarios,
        requested_seeds=set(args.seeds) if args.seeds else None,
        requested_ped_planners=(
            set(args.ped_planners) if args.ped_planners else None
        ),
        requested_robot_policies=(
            set(args.robot_policies) if args.robot_policies else None
        ),
        trial_selection=args.trial_selection,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
