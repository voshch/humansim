import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from arena_humansim.utils.bag_io import extract_agent_states
from arena_humansim.utils.evaluation.buckets import (
    DRIVER_CLASS,
    KNOWN_BUCKETS,
    SCENARIO_BUCKET,
)
from arena_humansim.utils.evaluation.framing import pick_framing, render_framing
from arena_humansim.utils.evaluation.metrics import (
    calculate_kinematic_metrics,
    calculate_run_collisions,
    pairwise_hausdorff,
)


def parse_trial_dir(name: str) -> tuple[str, str, int]:
    parts = name.split("__")
    if len(parts) != 3:
        raise ValueError(f"trial dir {name!r} not in <scenario>__<planner>__<seed> form")
    scenario, planner, seed = parts
    return scenario, planner, int(seed)


def load_recordings(recordings_dir: Path) -> pd.DataFrame:
    frames = []
    for trial_dir in sorted(recordings_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        bag_dir = trial_dir / "bag"
        if not bag_dir.exists():
            continue
        try:
            scenario, planner, seed = parse_trial_dir(trial_dir.name)
        except ValueError:
            print(f"skipping {trial_dir.name} (not a trial dir)")
            continue
        df = extract_agent_states(bag_dir)
        if df.empty:
            print(f"skipping {trial_dir.name} (empty bag)")
            continue
        df["scenario"] = scenario
        df["seed"] = seed
        df["planner"] = planner
        df["bucket"] = SCENARIO_BUCKET.get(scenario, "unknown")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no usable trials under {recordings_dir}")
    return pd.concat(frames, ignore_index=True)


def kinematics_per_trial(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, planner, seed), trial in df.groupby(["scenario", "planner", "seed"]):
        per_agent = (
            trial.sort_values(["agent_id", "time"])
            .groupby("agent_id")
            .apply(calculate_kinematic_metrics)
            .reset_index()
        )
        rows.append({
            "bucket": SCENARIO_BUCKET.get(scenario, "unknown"),
            "scenario": scenario,
            "planner": planner,
            "seed": seed,
            "jerk": per_agent["mean_jerk"].mean(),
            "curvature": per_agent["mean_curvature"].mean(),
            "collisions": calculate_run_collisions(trial),
        })
    return pd.DataFrame(rows)


def compute_pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, seed, agent_id), group in df.groupby(["scenario", "seed", "agent_id"]):
        planners = sorted(group["planner"].unique())
        if len(planners) < 2:
            continue
        traj = {p: group[group["planner"] == p][["x", "y"]].values for p in planners}
        bucket = SCENARIO_BUCKET.get(scenario, "unknown")
        for p1, p2 in itertools.combinations(planners, 2):
            t1, t2 = traj[p1], traj[p2]
            if len(t1) == 0 or len(t2) == 0:
                continue
            d = pairwise_hausdorff(t1, t2)
            rows.append({
                "bucket": bucket,
                "scenario": scenario,
                "seed": seed,
                "agent_id": agent_id,
                "p1": p1,
                "p2": p2,
                "class1": DRIVER_CLASS.get(p1, "unknown"),
                "class2": DRIVER_CLASS.get(p2, "unknown"),
                "same_class": DRIVER_CLASS.get(p1) == DRIVER_CLASS.get(p2),
                "hausdorff": d,
            })
    return pd.DataFrame(rows)


def _scenario_equal_weight(slice_df: pd.DataFrame) -> float:
    if slice_df.empty:
        return float("nan")
    return float(slice_df.groupby("scenario")["hausdorff"].mean().mean())


def _bootstrap_K(slice_df: pd.DataFrame, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    within = slice_df[slice_df["same_class"]]
    across = slice_df[~slice_df["same_class"]]
    scenarios = sorted(slice_df["scenario"].unique())
    if not scenarios or within.empty or across.empty:
        return float("nan"), float("nan")

    if len(scenarios) <= 1:
        w_arr = within["hausdorff"].to_numpy()
        a_arr = across["hausdorff"].to_numpy()
        if len(w_arr) == 0 or len(a_arr) == 0:
            return float("nan"), float("nan")
        ks = []
        for _ in range(n_iter):
            xb = w_arr[rng.integers(0, len(w_arr), len(w_arr))].mean()
            yb = a_arr[rng.integers(0, len(a_arr), len(a_arr))].mean()
            if xb > 0:
                ks.append(yb / xb)
    else:
        x_per = within.groupby("scenario")["hausdorff"].mean()
        y_per = across.groupby("scenario")["hausdorff"].mean()
        common = sorted(set(x_per.index) & set(y_per.index))
        if not common:
            return float("nan"), float("nan")
        x_arr = x_per.loc[common].to_numpy()
        y_arr = y_per.loc[common].to_numpy()
        n = len(common)
        ks = []
        for _ in range(n_iter):
            idx = rng.integers(0, n, n)
            xb = x_arr[idx].mean()
            yb = y_arr[idx].mean()
            if xb > 0:
                ks.append(yb / xb)

    if not ks:
        return float("nan"), float("nan")
    return float(np.percentile(ks, 5)), float(np.percentile(ks, 95))


def headline(pairwise_df: pd.DataFrame, n_bootstrap: int = 1000, ci_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(ci_seed)

    def _agg(slice_df: pd.DataFrame, label: str) -> dict:
        within = slice_df[slice_df["same_class"]]
        across = slice_df[~slice_df["same_class"]]
        x = _scenario_equal_weight(within)
        y = _scenario_equal_weight(across)
        k = y / x if x and not np.isnan(x) else float("nan")
        k_lo, k_hi = _bootstrap_K(slice_df, n_bootstrap, rng)
        return {
            "bucket": label,
            "within_class_mean": x,
            "across_class_mean": y,
            "ratio_K": k,
            "K_lo": k_lo,
            "K_hi": k_hi,
            "n_pairs": len(slice_df),
            "n_scenarios": slice_df["scenario"].nunique(),
        }

    rows = []
    for bucket in KNOWN_BUCKETS:
        slice_df = pairwise_df[pairwise_df["bucket"] == bucket]
        if slice_df.empty:
            continue
        rows.append(_agg(slice_df, bucket))

    pooled = pairwise_df[pairwise_df["bucket"].isin(KNOWN_BUCKETS)]
    if not pooled.empty:
        rows.append(_agg(pooled, "all"))

    return pd.DataFrame(rows)


def run_analysis(
    recordings_dir: Path,
    out_dir: Path,
    n_bootstrap: int = 1000,
    ci_seed: int = 0,
    framing_threshold: float = 1.2,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bags from {recordings_dir}...")
    df = load_recordings(recordings_dir)
    print(
        f"  {len(df)} rows | "
        f"{df['scenario'].nunique()} scenarios x "
        f"{df['planner'].nunique()} planners x "
        f"{df['seed'].nunique()} seeds | "
        f"buckets: {sorted(df['bucket'].unique())}"
    )

    print("Per-trial kinematics...")
    kin_df = kinematics_per_trial(df)
    kin_path = out_dir / "kinematics_per_trial.csv"
    kin_df.to_csv(kin_path, index=False)
    print(f"  wrote {kin_path}")

    print("Pairwise Hausdorff...")
    pair_df = compute_pairwise_table(df)
    pair_path = out_dir / "pairwise_distance.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"  wrote {pair_path} ({len(pair_df)} pairs)")

    print(f"Headline (n_bootstrap={n_bootstrap})...")
    head_df = headline(pair_df, n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    head_path = out_dir / "headline.csv"
    head_df.to_csv(head_path, index=False)
    print(f"  wrote {head_path}")

    print("Picking abstract framing...")
    framing = pick_framing(head_df, threshold=framing_threshold)
    framing_path = out_dir / "framing.md"
    framing_path.write_text(render_framing(framing))
    print(f"  wrote {framing_path}")

    print()
    print("=== Headline (X = within-class mean, Y = across-class mean, K = Y/X, [K_lo, K_hi] = 90% bootstrap CI) ===")
    print(head_df.to_string(index=False, float_format="{:.4f}".format))

    print()
    print(f"=== Recommended abstract framing: {framing['recommended']} ===")
    if framing["recommended"] != "incomplete":
        print(
            f"  X_nav={framing['X_nav']:.3f}m  K_nav={framing['K_nav']:.2f}x  "
            f"K_bt={framing['K_bt']:.2f}x  K_het={framing['K_het']:.2f}x  "
            f"K_pooled={framing['K_pooled']:.2f}x"
        )
        print(f"  Full sentence + alternatives in {framing_path}")

    return {
        "kinematics": kin_df,
        "pairwise": pair_df,
        "headline": head_df,
        "framing": framing,
    }
