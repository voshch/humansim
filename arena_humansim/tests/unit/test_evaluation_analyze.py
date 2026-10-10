from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import pandas as pd
from arena_humansim.utils.evaluation.analyze import run_analysis

from tests.unit._bags import NESTED, Agent, write_bag


def _record_trial(sweep: Path, name: str, seed: int) -> None:
    trial = sweep / name
    trial.mkdir(parents=True)
    (trial / "scenario.yaml").write_text("agents: []\n")
    frames = [(t * 100_000_000, [Agent(1, 0.1 * t + 0.01 * seed, 0.02 * t * t, vx=1.0)]) for t in range(20)]
    write_bag(trial / "bag", NESTED, frames)


def test_run_analysis_single_scenario_single_planner_pilot(tmp_path: Path) -> None:
    sweep = tmp_path / "pilot"
    for seed in (1, 2, 3, 4):
        _record_trial(sweep, f"nav_sparse_corridor__sfm__{seed}", seed)
    out = tmp_path / "out"

    result = run_analysis(recordings_dirs=sweep, out_dir=out, n_bootstrap=10)

    assert result["pairwise"].empty
    assert "bucket" in result["pairwise"].columns
    assert result["headline"].empty
    assert "bucket" in result["headline"].columns
    assert len(result["self_divergence"]) == 6
    assert result["framing"]["recommended"] == "incomplete"
    for csv in ("kinematics_per_trial.csv", "pairwise_distance.csv", "self_divergence.csv", "headline.csv", "variance_decomposition.csv", "within_class_pairs.csv"):
        assert (out / csv).is_file()
    assert len(pd.read_csv(out / "kinematics_per_trial.csv")) == 4
