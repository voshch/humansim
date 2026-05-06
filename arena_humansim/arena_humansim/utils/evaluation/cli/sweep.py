import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FINAL_RTF_RE = re.compile(r"sim=([0-9.]+)s, wall=([0-9.]+)s, compute=([0-9.]+)s")


def _scenario_has_robot(scenario_name: str) -> bool | None:
    """True if scenario yaml declares any kind=1 agent. None if the yaml can't be located/parsed."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("arena_humansim"))
    except Exception:
        return None
    p = Path(scenario_name)
    candidate = p if p.is_file() else share / "config" / "scenarios" / (p.name if p.suffix else f"{p.name}.yaml")
    if not candidate.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(candidate.read_text())
    except Exception:
        return None
    agents = data.get("agents", []) if isinstance(data, dict) else []
    return any(isinstance(a, dict) and a.get("kind") == 1 for a in agents)


def _write_progress(progress_file: Path | None, current: int, total: int) -> None:
    if progress_file is None:
        return
    tmp = progress_file.with_suffix(progress_file.suffix + ".tmp")
    tmp.write_text(f"{current}/{total}\n")
    tmp.replace(progress_file)


def _run_and_tee(cmd: list[str], env: dict[str, str]) -> tuple[float, float | None, float | None, float | None, int]:
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    sim_s: float | None = None
    wall_s: float | None = None
    compute_s: float | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        m = FINAL_RTF_RE.search(line)
        if m:
            sim_s = float(m.group(1))
            wall_s = float(m.group(2))
            compute_s = float(m.group(3))
    proc.wait()
    total_wall = time.monotonic() - t0
    return total_wall, sim_s, wall_s, compute_s, proc.returncode


def _try_claim(claims_dir: Path, trial_name: str) -> bool:
    # O_CREAT|O_EXCL atomically; the first worker to create the file wins.
    try:
        (claims_dir / f"{trial_name}.lock").touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def run_sweep(
    trials: list[tuple[str, str, str, int]],
    sim_duration: int,
    output_dir: Path,
    force_planner_scenarios: set[str] | frozenset[str] = frozenset(),
    worker_id: int = 0,
    num_workers: int = 1,
    progress_file: Path | None = None,
    robot_shutdown: str = "",
) -> None:
    if robot_shutdown not in ("", "true", "false"):
        raise ValueError(f"robot_shutdown must be '', 'true', or 'false'; got {robot_shutdown!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    claims_dir = output_dir / ".claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    if worker_id == 0:
        robot_scenarios = {scenario for scenario, _planner, robot_policy, _seed in trials if robot_policy}
        for scenario in sorted(robot_scenarios):
            has_robot = _scenario_has_robot(scenario)
            if has_robot is False:
                print(f"WARNING: scenario {scenario!r} has no kind=1 agent — robot_policy override will be a no-op and every robot metric will be NaN.")
            elif has_robot is None:
                print(f"WARNING: could not locate scenario yaml for {scenario!r}; cannot verify robot presence.")

    own = [t for i, t in enumerate(trials) if i % num_workers == worker_id]
    others = [t for i, t in enumerate(trials) if i % num_workers != worker_id]

    done_count = 0
    assigned = len(own)
    _write_progress(progress_file, done_count, assigned)

    overhead_csv = output_dir / "launch_overhead.csv"
    if not overhead_csv.exists():
        with overhead_csv.open("w") as f:
            f.write("scenario,planner,robot_policy,seed,total_wall_s,tick_wall_s,compute_s,overhead_s,overhead_pct\n")

    pos = 0

    def run_one(trial: tuple[str, str, str, int]) -> None:
        nonlocal pos
        pos += 1
        scenario, planner, robot_policy, seed = trial
        trial_dir = output_dir / _trial_dir_name(*trial)
        sentinel = trial_dir / ".done"

        if trial_dir.exists():
            shutil.rmtree(trial_dir)

        rp = f", RobotPolicy={robot_policy}" if robot_policy else ""
        print("=========================================")
        print(f"Worker {worker_id} #{pos}: Scenario={scenario}, Planner={planner}{rp}, Seed={seed}")
        print("=========================================")

        domain_id = worker_id * 2 + (pos % 2) + 2
        trial_env = {**os.environ, "ROS_DOMAIN_ID": str(domain_id)}
        cmd = [
            "ros2",
            "launch",
            "arena_humansim",
            "arena_humansim.launch.py",
            f"scenario:={scenario}",
            f"local_planner:={planner}",
            f"force_local_planner:={'true' if scenario in force_planner_scenarios else 'false'}",
            f"seed:={seed}",
            "rtf:=0",
            f"time:={sim_duration}",
            "record:=True",
            f"record_dir:={trial_dir}",
            "render:=False",
            "rviz:=false",
            "markers:=0",
        ]
        if robot_policy:
            cmd.append(f"robot_policy:={robot_policy}")
        if robot_shutdown:
            cmd.append(f"robot_shutdown:={robot_shutdown}")

        print(f"Simulation started at max speed (ROS_DOMAIN_ID={domain_id}). Waiting for {sim_duration} sim-seconds to elapse...")
        total_wall, _sim_s, wall_s, compute_s, rc = _run_and_tee(cmd, trial_env)
        if rc != 0:
            print(f"Trial failed: {trial_dir.name}: exit code {rc}")
        else:
            sentinel.touch()
            print("Simulation closed cleanly.")

        if wall_s is not None:
            overhead_s = total_wall - wall_s
            overhead_pct = 100.0 * overhead_s / total_wall if total_wall > 0 else 0.0
            print(f"  total={total_wall:.1f}s  tick_loop={wall_s:.1f}s  launch+teardown={overhead_s:.1f}s ({overhead_pct:.1f}%)")
            with overhead_csv.open("a") as f:
                f.write(f"{scenario},{planner},{robot_policy},{seed},{total_wall:.3f},{wall_s:.3f},{compute_s if compute_s is not None else ''},{overhead_s:.3f},{overhead_pct:.2f}\n")
        else:
            print(f"  total={total_wall:.1f}s  (final-rtf line not found; overhead unknown)")

    # Phase 1: own shard. Sibling-claimed and pre-existing-done trials drop out of `assigned`.
    for trial in own:
        name = _trial_dir_name(*trial)
        if (output_dir / name / ".done").exists():
            assigned -= 1
            _write_progress(progress_file, done_count, assigned)
            continue
        if not _try_claim(claims_dir, name):
            assigned -= 1
            _write_progress(progress_file, done_count, assigned)
            continue
        run_one(trial)
        done_count += 1
        _write_progress(progress_file, done_count, assigned)

    # Phase 2: steal anything siblings haven't claimed. Each successful claim grows `assigned`.
    for trial in others:
        name = _trial_dir_name(*trial)
        if (output_dir / name / ".done").exists():
            continue
        if not _try_claim(claims_dir, name):
            continue
        assigned += 1
        _write_progress(progress_file, done_count, assigned)
        run_one(trial)
        done_count += 1
        _write_progress(progress_file, done_count, assigned)

    print(f"Worker {worker_id} done: completed {done_count}/{assigned}.")
    print(f"Per-trial launch overhead: {overhead_csv}")


def _trial_dir_name(scenario: str, planner: str, robot_policy: str, seed: int) -> str:
    if robot_policy:
        return f"{scenario}__{planner}__{robot_policy}__{seed}"
    return f"{scenario}__{planner}__{seed}"


def _parse_trial_file(path: Path) -> list[tuple[str, str, str, int]]:
    trials: list[tuple[str, str, str, int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) == 3:
            scenario, planner, seed = parts
            robot_policy = ""
        elif len(parts) == 4:
            scenario, planner, robot_policy, seed = parts
        else:
            raise ValueError(f"Bad trial line (expect 'scenario:planner:seed' or 'scenario:planner:robot_policy:seed'): {line!r}")
        trials.append((scenario, planner, robot_policy, int(seed)))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Work-stealing sweep worker; iterates own modulo shard, then steals undone+unclaimed siblings.")
    parser.add_argument("--trial_file", type=str, required=True, help="Path to file with one 'scenario:planner:[robot_policy:]seed' per line. Same file for every worker.")
    parser.add_argument("--sim_duration", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="<output_dir>/<trial_name>/ per trial; <output_dir>/.claims/<trial_name>.lock for atomic claims")
    parser.add_argument("--force_planner_scenarios", nargs="*", default=[])
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Total cooperating workers; this worker owns trials with index % num_workers == worker_id.")
    parser.add_argument("--progress_file", type=str, default=None)
    parser.add_argument("--robot_shutdown", choices=("", "true", "false"), default="", help="end each trial when every robot reaches its goal; empty leaves the scenario value (default false).")
    args = parser.parse_args()

    trials = _parse_trial_file(Path(args.trial_file))
    run_sweep(
        trials=trials,
        sim_duration=args.sim_duration,
        output_dir=Path(args.output_dir).resolve(),
        force_planner_scenarios=set(args.force_planner_scenarios),
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        progress_file=Path(args.progress_file).resolve() if args.progress_file else None,
        robot_shutdown=args.robot_shutdown,
    )


if __name__ == "__main__":
    main()
