from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from arena_humansim.utils.evaluation.variance import (
    decompose_scalar_variance,
    headline_seed_ratio,
    self_divergence_table,
)


def _trajectory(n: int, dx: float, dy: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(0.0, 1.0, n)[:, None] * np.array([dx, dy])
    return base + rng.normal(scale=jitter, size=(n, 2))


def _build_states(*, scenarios: list[str], planners: dict[str, tuple[float, float]], seeds: list[int], jitter: float, n_steps: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for scenario in scenarios:
        for planner, (dx, dy) in planners.items():
            for seed in seeds:
                traj = _trajectory(n_steps, dx, dy, jitter, rng)
                for t, (x, y) in enumerate(traj):
                    rows.append(
                        {
                            "time": float(t),
                            "agent_id": 0,
                            "x": float(x),
                            "y": float(y),
                            "vx": 0.0,
                            "vy": 0.0,
                            "radius": 0.3,
                            "scenario": scenario,
                            "seed": seed,
                            "planner": planner,
                            "bucket": "nav" if scenario in ("simple_crossing", "corridor") else "bt",
                        }
                    )
    return pd.DataFrame(rows)


def test_self_divergence_table_pairs_all_seed_combinations() -> None:
    df = _build_states(
        scenarios=["simple_crossing"],
        planners={"sfm": (10.0, 0.0)},
        seeds=[0, 1, 2, 3],
        jitter=0.01,
    )
    sdf = self_divergence_table(df)
    # 4 seeds -> C(4,2) = 6 pairs per (scenario, agent_id, planner) cell
    assert len(sdf) == 6
    assert set(sdf.columns) >= {"bucket", "scenario", "agent_id", "planner", "class", "seed_i", "seed_j", "hausdorff"}
    assert (sdf["seed_i"] < sdf["seed_j"]).all()
    assert (sdf["hausdorff"] >= 0).all()
    assert sdf["class"].iloc[0] == "classical"


def test_self_divergence_skips_single_seed_cells() -> None:
    df = _build_states(
        scenarios=["simple_crossing"],
        planners={"sfm": (10.0, 0.0), "orca": (10.0, 0.0)},
        seeds=[0],
        jitter=0.01,
    )
    sdf = self_divergence_table(df)
    assert sdf.empty


def test_headline_seed_ratio_shows_driver_dominance() -> None:
    # Drivers diverge a lot (different goals); seeds barely jitter the path.
    df = _build_states(
        scenarios=["simple_crossing", "corridor"],
        planners={"sfm": (10.0, 0.0), "orca": (0.0, 10.0), "nsp": (7.0, 7.0)},
        seeds=[0, 1, 2],
        jitter=0.01,
    )
    from arena_humansim.utils.evaluation.analyze import compute_pairwise_table

    pair_df = compute_pairwise_table(df)
    self_df = self_divergence_table(df)
    head = headline_seed_ratio(pair_df, self_df, n_bootstrap=100, ci_seed=0)
    nav_row = head[head["bucket"] == "nav"].iloc[0]
    assert nav_row["across_driver_mean"] > nav_row["within_driver_seed_mean"]
    assert nav_row["K_seed"] > 10.0
    assert nav_row["K_seed_lo"] <= nav_row["K_seed"] <= nav_row["K_seed_hi"] or np.isnan(nav_row["K_seed_lo"])


def test_headline_seed_ratio_handles_empty_inputs() -> None:
    empty = pd.DataFrame(columns=["bucket", "scenario", "hausdorff"])
    head = headline_seed_ratio(empty, empty, n_bootstrap=10, ci_seed=0)
    assert head.empty


def test_decompose_scalar_variance_attributes_to_factor() -> None:
    # Construct kinematics with metric driven entirely by planner: jerk = f(planner).
    rows = []
    planner_offset = {"sfm": 1.0, "orca": 2.0, "nsp": 3.0}
    for scenario in ("simple_crossing", "corridor"):
        for planner in ("sfm", "orca", "nsp"):
            for seed in (0, 1, 2, 3):
                rows.append(
                    {
                        "bucket": "nav",
                        "scenario": scenario,
                        "planner": planner,
                        "seed": seed,
                        "jerk": planner_offset[planner],
                        "curvature": float(seed),
                        "collisions": 0,
                    }
                )
    kin = pd.DataFrame(rows)
    out = decompose_scalar_variance(kin)
    nav = out[out["bucket"] == "nav"]
    jerk_row = nav[nav["metric"] == "jerk"].iloc[0]
    curv_row = nav[nav["metric"] == "curvature"].iloc[0]
    assert jerk_row["share_driver"] == pytest.approx(1.0, abs=1e-6)
    assert jerk_row["share_seed"] == pytest.approx(0.0, abs=1e-6)
    assert curv_row["share_seed"] == pytest.approx(1.0, abs=1e-6)
    assert curv_row["share_driver"] == pytest.approx(0.0, abs=1e-6)
    # Collisions are constant, so total SS = 0 -> row dropped entirely.
    assert nav[nav["metric"] == "collisions"].empty


def test_decompose_scalar_variance_shares_sum_to_one() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for scenario in ("simple_crossing", "corridor", "bottleneck"):
        for planner in ("sfm", "orca", "hsfm"):
            for seed in (0, 1, 2, 3):
                rows.append(
                    {
                        "bucket": "nav",
                        "scenario": scenario,
                        "planner": planner,
                        "seed": seed,
                        "jerk": float(rng.normal()),
                        "curvature": float(rng.normal()),
                        "collisions": int(rng.integers(0, 5)),
                    }
                )
    kin = pd.DataFrame(rows)
    out = decompose_scalar_variance(kin)
    share_cols = [c for c in out.columns if c.startswith("share_")]
    sums = out[share_cols].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert (out[share_cols] >= -1e-9).all().all()


def test_decompose_skips_under_sized_buckets() -> None:
    rows = [
        {"bucket": "nav", "scenario": "simple_crossing", "planner": "sfm", "seed": 0, "jerk": 1.0, "curvature": 0.0, "collisions": 0},
        {"bucket": "nav", "scenario": "simple_crossing", "planner": "sfm", "seed": 1, "jerk": 2.0, "curvature": 0.0, "collisions": 0},
    ]
    kin = pd.DataFrame(rows)
    out = decompose_scalar_variance(kin)
    # Only one planner in nav -> bucket skipped.
    assert out.empty
