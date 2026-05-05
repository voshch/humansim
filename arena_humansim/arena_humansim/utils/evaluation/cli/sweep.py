import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FINAL_RTF_RE = re.compile(r"sim=([0-9.]+)s, wall=([0-9.]+)s, compute=([0-9.]+)s")


def _write_progress(progress_file: Path | None, current: int, total: int) -> None:
    if progress_file is None:
        return
    tmp = progress_file.with_suffix(progress_file.suffix + ".tmp")
    tmp.write_text(f"{current}/{total}\n")
    tmp.replace(progress_file)


def _run_and_tee(cmd: list[str], env: dict[str, str]) -> tuple[float, float | None, float | None, float | None, int]:
    """Run cmd with combined stdout/stderr teed to this process; parse final-rtf line."""
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


def run_sweep(
    trials: list[tuple[str, str, int]],
    sim_duration: int,
    output_dir: Path,
    force_planner_scenarios: set[str] | frozenset[str] = frozenset(),
    worker_id: int = 0,
    progress_file: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(trials)
    current_run = 0
    _write_progress(progress_file, 0, total_runs)

    overhead_csv = output_dir / "launch_overhead.csv"
    if not overhead_csv.exists():
        with overhead_csv.open("w") as f:
            f.write("scenario,planner,seed,total_wall_s,tick_wall_s,compute_s,overhead_s,overhead_pct\n")

    for scenario, planner, seed in trials:
        current_run += 1
        trial_dir = output_dir / f"{scenario}__{planner}__{seed}"
        sentinel = trial_dir / ".done"

        if sentinel.exists():
            print(f"Skipping ({current_run}/{total_runs}): {scenario}/{planner}/seed={seed} already complete")
            _write_progress(progress_file, current_run, total_runs)
            continue

        if trial_dir.exists():
            shutil.rmtree(trial_dir)

        print("=========================================")
        print(f"Running ablation ({current_run}/{total_runs}): Scenario={scenario}, Planner={planner}, Seed={seed}")
        print("=========================================")

        domain_id = worker_id * 2 + (current_run % 2) + 1
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

        print(f"Simulation started at max speed (ROS_DOMAIN_ID={domain_id}). Waiting for {sim_duration} sim-seconds to elapse...")
        total_wall, _sim_s, wall_s, compute_s, rc = _run_and_tee(cmd, trial_env)
        if rc != 0:
            print(f"An error occurred during {scenario} with {planner} (Seed {seed}): exit code {rc}")
        else:
            sentinel.touch()
            print("Simulation closed cleanly.")

        if wall_s is not None:
            overhead_s = total_wall - wall_s
            overhead_pct = 100.0 * overhead_s / total_wall if total_wall > 0 else 0.0
            print(f"  total={total_wall:.1f}s  tick_loop={wall_s:.1f}s  launch+teardown={overhead_s:.1f}s ({overhead_pct:.1f}%)")
            with overhead_csv.open("a") as f:
                f.write(f"{scenario},{planner},{seed},{total_wall:.3f},{wall_s:.3f},{compute_s if compute_s is not None else ''},{overhead_s:.3f},{overhead_pct:.2f}\n")
        else:
            print(f"  total={total_wall:.1f}s  (final-rtf line not found; overhead unknown)")

        _write_progress(progress_file, current_run, total_runs)

    print("All ablations completed!")
    print(f"Per-trial launch overhead: {overhead_csv}")


def _parse_trial_file(path: Path) -> list[tuple[str, str, int]]:
    trials: list[tuple[str, str, int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scenario, planner, seed = line.split(":")
        trials.append((scenario, planner, int(seed)))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-process serial sweep over an explicit trial list.")
    parser.add_argument("--trial_file", type=str, required=True,
                        help="Path to file with one 'scenario:planner:seed' per line.")
    parser.add_argument("--sim_duration", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="<output_dir>/<scenario>__<planner>__<seed>/bag/ per trial")
    parser.add_argument("--force_planner_scenarios", nargs="*", default=[],
                        help="Scenarios to run with force_local_planner:=true.")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Selects the worker's domain-id pair: {2*worker_id+1, 2*worker_id+2}.")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="Atomic 'current/total' write after each trial for the sweep monitor.")
    args = parser.parse_args()

    trials = _parse_trial_file(Path(args.trial_file))
    run_sweep(
        trials=trials,
        sim_duration=args.sim_duration,
        output_dir=Path(args.output_dir).resolve(),
        force_planner_scenarios=set(args.force_planner_scenarios),
        worker_id=args.worker_id,
        progress_file=Path(args.progress_file).resolve() if args.progress_file else None,
    )


if __name__ == "__main__":
    main()
