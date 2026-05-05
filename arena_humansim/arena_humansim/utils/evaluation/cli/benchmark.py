import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from arena_humansim.utils.evaluation.analyze import run_analysis

DEFAULT_NAV = ["simple_crossing", "corridor", "flow_coverage", "t_junction", "bottleneck", "l_corridor"]
DEFAULT_BT = ["escort", "queue", "bt_coverage"]
DEFAULT_HET = ["robot_test"]
DEFAULT_PLANNERS = ["sfm", "hsfm", "orca", "straight", "nsp", "socialgail"]

MAX_WORKERS = 50  # ROS_DOMAIN_ID portable range 0..101; ping-pong needs 2 ids per worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel sweep + monitor + analyze. Subsumes benchmark.sh.")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_NAV + DEFAULT_BT)
    parser.add_argument("--planners", nargs="+", default=DEFAULT_PLANNERS)
    parser.add_argument("--seeds", type=int, default=20,
                        help="Total seeds per (scenario, planner). Trials are sharded across workers.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sim_duration", type=int, default=60)
    parser.add_argument("--force_planner_scenarios", nargs="*", default=[])
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Output dir; defaults to $ARENA_DATA_DIR/peds/<ts>_sweep")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true",
                        help="Resume the latest incomplete sweep under $ARENA_DATA_DIR/peds; exit 1 if none found.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.resume:
        data_dir = os.environ.get("ARENA_DATA_DIR")
        if not data_dir:
            parser.error("--resume requires $ARENA_DATA_DIR (or pass --run_dir explicitly)")
        run_dir = None
        for cand in sorted(Path(data_dir, "peds").glob("*_sweep"), reverse=True):
            manifest = cand / "run_args.json"
            if not manifest.exists():
                continue
            stored = json.loads(manifest.read_text())
            expected = len(stored["scenarios"]) * len(stored["planners"]) * stored["seeds"]
            done = sum(1 for _ in cand.rglob(".done"))
            if done < expected:
                run_dir = cand
                print(f"Resuming {cand} ({done}/{expected} trials done).")
                break
        if run_dir is None:
            sys.exit(f"--resume: no incomplete sweep found under {Path(data_dir, 'peds')}")
    else:
        data_dir = os.environ.get("ARENA_DATA_DIR")
        if not data_dir:
            parser.error("--run_dir required when $ARENA_DATA_DIR is unset")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(data_dir) / "peds" / f"{ts}_sweep"

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_args.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        print(f"Resuming sweep from {run_dir}; using stored run_args.json (CLI args ignored).")
        for k, v in stored.items():
            setattr(args, k, v)
    else:
        stored = {k: v for k, v in vars(args).items() if k != "run_dir"}
        manifest_path.write_text(json.dumps(stored, indent=2, sort_keys=True))

    log_dir = run_dir / "logs"
    progress_dir = run_dir / "progress"
    trials_dir = run_dir / "trials"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recordings: {run_dir}")
    print(f"Logs: {log_dir}")

    all_trials = [(s, p, seed) for s in args.scenarios for p in args.planners for seed in range(1, args.seeds + 1)]
    if args.workers > MAX_WORKERS:
        print(f"Clamping workers from {args.workers} to {MAX_WORKERS} (ROS_DOMAIN_ID range).")
    workers = min(args.workers, len(all_trials), MAX_WORKERS)

    shards: list[list[tuple[str, str, int]]] = [[] for _ in range(workers)]
    for i, trial in enumerate(all_trials):
        shards[i % workers].append(trial)

    worker_procs: list[subprocess.Popen[bytes]] = []
    progress_files: list[str] = []
    log_files = []
    for i in range(workers):
        trial_file = trials_dir / f"worker_{i}.txt"
        trial_file.write_text("".join(f"{s}:{p}:{seed}\n" for s, p, seed in shards[i]))
        progress_file = progress_dir / f"worker_{i}.txt"
        progress_files.append(str(progress_file))
        log_file = log_dir / f"worker_{i}.log"

        cmd = [
            sys.executable, "-m", "arena_humansim.utils.evaluation.cli",
            "sweep",
            "--trial_file", str(trial_file),
            "--sim_duration", str(args.sim_duration),
            "--output_dir", str(run_dir),
            "--worker_id", str(i),
            "--progress_file", str(progress_file),
        ]
        if args.force_planner_scenarios:
            cmd.extend(["--force_planner_scenarios", *args.force_planner_scenarios])

        logf = open(log_file, "wb")
        log_files.append(logf)
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
        worker_procs.append(proc)
        d_lo, d_hi = i * 2 + 1, i * 2 + 2
        print(f"Worker {i} started (PID {proc.pid}): {len(shards[i])} trials, ROS_DOMAIN_ID in {{{d_lo},{d_hi}}}")

    print()
    monitor_cmd = [sys.executable, "-m", "arena_humansim.utils.evaluation.cli", "monitor", *progress_files]
    monitor_proc = subprocess.Popen(monitor_cmd)

    failed = 0
    for proc in worker_procs:
        if proc.wait() != 0:
            failed += 1

    try:
        monitor_proc.send_signal(signal.SIGTERM)
        monitor_proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        monitor_proc.kill()

    for logf in log_files:
        logf.close()

    print()
    print("=== Sweep complete ===")
    expected = len(args.scenarios) * len(args.planners) * args.seeds
    done = sum(1 for _ in run_dir.rglob(".done"))
    print(f"Trials complete: {done}/{expected}")
    if failed > 0:
        print(f"WARNING: {failed}/{len(worker_procs)} workers exited nonzero (see {log_dir}/worker_*.log)")
    print(f"Recordings: {run_dir}")
    print()

    if done < expected:
        print(f"Skipping analysis: {expected - done} trials remaining. Re-invoke `benchmark --run_dir {run_dir}` to continue.")
        return

    print("=== Pairwise divergence analysis ===")
    run_analysis(
        recordings_dir=run_dir,
        out_dir=run_dir,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
