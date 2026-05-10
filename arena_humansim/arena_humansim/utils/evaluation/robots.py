from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from arena_humansim.utils.evaluation.bag_cache import load_multi
from arena_humansim.utils.evaluation.buckets import SCENARIO_BUCKET

ROBOT_AGENT_ID = -1


def assert_run_mode(run_dir: Path, expected_mode: str, *, require_cop: bool = False) -> dict[str, object]:
    manifest_path = run_dir / "run_args.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{run_dir}: no run_args.json - not a benchmark sweep dir")
    manifest = json.loads(manifest_path.read_text())
    actual_mode = manifest.get("mode")
    if actual_mode != expected_mode:
        raise RuntimeError(f"{run_dir}: manifest mode={actual_mode!r}, expected {expected_mode!r}. This analyzer only runs on `{expected_mode}` sweeps.")
    if require_cop and not manifest.get("cop", False):
        raise RuntimeError(f"{run_dir}: manifest cop=false. This analyzer only runs on COP-flagged sweeps (invoke benchmark with --cop).")
    return manifest


GOAL_REACH_THRESHOLD_M = 0.5
PERSONAL_SPACE_M = 0.5
FREEZE_SPEED_THRESHOLD = 0.05
FREEZE_WINDOW_S = 5.0


def parse_robots_trial_dir(name: str) -> tuple[str, str, str, int] | None:
    parts = name.split("__")
    if len(parts) == 4:
        scenario, planner, robot_policy, seed = parts
    elif len(parts) == 3:
        scenario, planner, seed = parts
        robot_policy = ""
    else:
        return None
    try:
        return scenario, planner, robot_policy, int(seed)
    except ValueError:
        return None


def _name_and_goal(yaml_path: Path) -> tuple[str | None, tuple[float, float] | None]:
    try:
        from arena_humansim.utils.scenario import load_scenario

        scen = load_scenario(str(yaml_path))
    except Exception:
        return None, None
    name = scen.name or None
    for a in scen.agents:
        if int(a.kind) == 1 and int(a.agent_id) == ROBOT_AGENT_ID and a.goal_sequence:
            last = a.goal_sequence[-1]
            return name, (float(last.x), float(last.y))
    return name, None


_eval_goal_index: dict[str, tuple[float, float] | None] | None = None


def _eval_goal_for(canonical: str) -> tuple[float, float] | None:
    """Lazy index of {scenario name: robot goal} from the eval tree, used when a snapshot
    lacks a goal (older recordings predate goal_sequence on het scenarios)."""
    global _eval_goal_index
    if _eval_goal_index is None:
        from arena_humansim.utils.scenario_discovery import scenario_roots

        _eval_goal_index = {}
        for root in scenario_roots():
            if root.name != "evaluation":
                continue
            for path in root.glob("**/*.yaml"):
                name, goal = _name_and_goal(path)
                if name is not None:
                    _eval_goal_index[name] = goal
    return _eval_goal_index.get(canonical)


def _load_snapshot(trial_dir: Path) -> tuple[str | None, tuple[float, float] | None]:
    """Read the per-trial scenario.yaml snapshot, returning (canonical name, robot goal).
    Falls back to the matching eval-tree source yaml if the snapshot lacks a goal — older
    snapshots predate the het scenarios using `goal_sequence`. Returns (None, None) when
    no snapshot or canonical match is found."""
    snap = trial_dir / "scenario.yaml"
    if not snap.is_file():
        return None, None
    name, goal = _name_and_goal(snap)
    if goal is None and name is not None:
        goal = _eval_goal_for(name)
    return name, goal


def _trajectory_arc_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    diffs = np.diff(xy, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def compute_robot_metrics(
    trial_df: pd.DataFrame,
    goal: tuple[float, float] | None,
) -> dict[str, float]:
    robot = trial_df[trial_df["agent_id"] == ROBOT_AGENT_ID].sort_values("time")
    if robot.empty:
        return {
            "success": float("nan"),
            "time_to_goal_s": float("nan"),
            "ttg_ratio": float("nan"),
            "path_efficiency": float("nan"),
            "n_robot_collisions": float("nan"),
            "personal_space_violations": float("nan"),
            "psv_per_sec": float("nan"),
            "frozen_at_end": float("nan"),
            "final_goal_dist": float("nan"),
        }

    times = robot["time"].to_numpy()
    xy = robot[["x", "y"]].to_numpy()
    speeds = np.linalg.norm(robot[["vx", "vy"]].to_numpy(), axis=1)
    t0 = float(times[0])
    t_end = float(times[-1])
    duration_s = t_end - t0

    if goal is None:
        success = float("nan")
        ttg_s = float("nan")
        ttg_ratio = float("nan")
        path_eff = float("nan")
        final_goal_dist = float("nan")
    else:
        gx, gy = goal
        d_to_goal = np.hypot(xy[:, 0] - gx, xy[:, 1] - gy)
        reached = d_to_goal <= GOAL_REACH_THRESHOLD_M
        success = float(bool(reached.any()))
        if reached.any():
            first_reach_idx = int(np.argmax(reached))
            ttg_s = float(times[first_reach_idx] - t0)
        else:
            ttg_s = float("nan")
        start_xy = xy[0]
        straight_line = float(np.hypot(gx - start_xy[0], gy - start_xy[1]))
        v_pref = 1.1
        nominal_s = straight_line / max(v_pref, 1e-6)
        ttg_ratio = float(ttg_s / nominal_s) if not np.isnan(ttg_s) and nominal_s > 0 else float("nan")
        arc = _trajectory_arc_length(xy)
        path_eff = float(arc / straight_line) if straight_line > 1e-6 else float("nan")
        final_goal_dist = float(d_to_goal[-1])

    radii = robot["radius"].to_numpy()
    radius = float(radii[0]) if len(radii) > 0 else 0.3
    n_collisions = 0
    psv_count = 0
    for _t, frame in trial_df.groupby("time"):
        rb = frame[frame["agent_id"] == ROBOT_AGENT_ID]
        if rb.empty:
            continue
        rb_x = float(rb["x"].iloc[0])
        rb_y = float(rb["y"].iloc[0])
        others = frame[frame["agent_id"] != ROBOT_AGENT_ID]
        if others.empty:
            continue
        ox = others["x"].to_numpy()
        oy = others["y"].to_numpy()
        orad = others["radius"].to_numpy()
        d = np.hypot(ox - rb_x, oy - rb_y)
        collisions = d < (radius + orad)
        if collisions.any():
            n_collisions += 1
        if (d < PERSONAL_SPACE_M + radius).any():
            psv_count += 1

    psv_per_sec = float(psv_count / duration_s) if duration_s > 0 else float("nan")

    freeze_window_mask = times >= (t_end - FREEZE_WINDOW_S)
    end_speed = float(speeds[freeze_window_mask].mean()) if freeze_window_mask.any() else float("nan")
    frozen = float(end_speed < FREEZE_SPEED_THRESHOLD) if not np.isnan(end_speed) else float("nan")

    return {
        "success": success,
        "time_to_goal_s": ttg_s,
        "ttg_ratio": ttg_ratio,
        "path_efficiency": path_eff,
        "n_robot_collisions": float(n_collisions),
        "personal_space_violations": float(psv_count),
        "psv_per_sec": psv_per_sec,
        "frozen_at_end": frozen,
        "final_goal_dist": final_goal_dist,
    }


def load_robots_recordings(recordings_dirs: Path | list[Path]) -> pd.DataFrame:
    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    nested = load_multi(recordings_dirs)
    total_trials = sum(len(nested.get(d.name, {})) for d in recordings_dirs)
    bar = tqdm(total=total_trials, unit="trial", desc="robots")
    for recordings_dir in recordings_dirs:
        for trial_name, df in nested.get(recordings_dir.name, {}).items():
            bar.update(1)
            parsed = parse_robots_trial_dir(trial_name)
            if parsed is None:
                continue
            scenario, planner, robot_policy, seed = parsed
            key = (scenario, planner, robot_policy, seed)
            if key in seen:
                bar.write(f"skipping {recordings_dir.name}/{trial_name} (duplicate of trial already loaded from earlier sweep)")
                continue
            if df.empty:
                continue
            seen.add(key)
            canonical, goal = _load_snapshot(recordings_dir / trial_name)
            bucket_key = canonical or scenario
            metrics = compute_robot_metrics(df, goal)
            rows.append(
                {
                    "scenario": scenario,
                    "bucket": SCENARIO_BUCKET.get(bucket_key, "unknown"),
                    "ped_planner": planner,
                    "robot_policy": robot_policy,
                    "seed": seed,
                    "source_dir": recordings_dir.name,
                    **metrics,
                }
            )
    bar.close()

    return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, n_iter: int, rng: np.random.Generator, percentiles: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    n = len(values)
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo = float(np.percentile(means, percentiles[0]))
    hi = float(np.percentile(means, percentiles[1]))
    return lo, hi


def aggregate_robots(
    per_trial_df: pd.DataFrame,
    group_by: list[str],
    n_bootstrap: int = 1000,
    ci_seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(ci_seed)
    metric_cols = [
        "success",
        "ttg_ratio",
        "path_efficiency",
        "n_robot_collisions",
        "psv_per_sec",
        "frozen_at_end",
        "final_goal_dist",
    ]
    out_rows: list[dict[str, object]] = []
    for keys, group in per_trial_df.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(group_by, keys, strict=False))
        row["n_trials"] = int(len(group))
        for m in metric_cols:
            vals = group[m].to_numpy(dtype=np.float64)
            mean = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
            lo, hi = _bootstrap_mean_ci(vals, n_bootstrap, rng)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def run_robots_analysis(
    recordings_dirs: Path | list[Path],
    out_dir: Path,
    n_bootstrap: int = 1000,
    ci_seed: int = 0,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]
    print(f"Loading robots bags from {len(recordings_dirs)} dir(s): {[d.name for d in recordings_dirs]}")
    per_trial = load_robots_recordings(recordings_dirs)
    if per_trial.empty:
        raise RuntimeError(f"no usable robots trials under {[str(d) for d in recordings_dirs]}")
    print(f"  {len(per_trial)} trials | {per_trial['scenario'].nunique()} scenarios x {per_trial['ped_planner'].nunique()} ped planners x {per_trial['robot_policy'].nunique()} robot policies x {per_trial['seed'].nunique()} seeds")
    per_trial_path = out_dir / "robots_per_trial.csv"
    per_trial.to_csv(per_trial_path, index=False)
    print(f"  wrote {per_trial_path}")

    cell_path = out_dir / "robots_cell.csv"
    cell = aggregate_robots(per_trial, group_by=["bucket", "ped_planner", "robot_policy"], n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    cell.to_csv(cell_path, index=False)
    print(f"  wrote {cell_path}")

    by_policy_path = out_dir / "robots_by_policy.csv"
    by_policy = aggregate_robots(per_trial, group_by=["robot_policy"], n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    by_policy.to_csv(by_policy_path, index=False)
    print(f"  wrote {by_policy_path}")

    by_policy_bucket_path = out_dir / "robots_by_policy_bucket.csv"
    by_pb = aggregate_robots(per_trial, group_by=["robot_policy", "bucket"], n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    by_pb.to_csv(by_policy_bucket_path, index=False)
    print(f"  wrote {by_policy_bucket_path}")

    print()
    print("=== Robots (mean +/- 95% bootstrap CI) by robot_policy x bucket ===")
    summary_cols = ["robot_policy", "bucket", "n_trials", "success_mean", "success_ci_lo", "success_ci_hi", "ttg_ratio_mean", "psv_per_sec_mean"]
    print(by_pb[summary_cols].to_string(index=False, float_format="{:.3f}".format))

    return {"per_trial": per_trial, "cell": cell, "by_policy": by_policy, "by_policy_bucket": by_pb}
