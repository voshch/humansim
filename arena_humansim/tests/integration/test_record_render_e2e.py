from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("rosbag2_py")
pytest.importorskip("matplotlib")

import rclpy
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from tests.integration._helpers import build_manager


def test_record_and_autorender(tmp_path: Path) -> None:
    scenario = ScenarioConfig(
        name="record_e2e",
        simulation=SimulationParams(seed=1, dt=0.05, max_ticks=10),
        modules=ModuleConfig(),
    )
    record_dir = tmp_path / "run"

    mgr = build_manager(
        scenario,
        extra_params=[
            Parameter("record_bag", Parameter.Type.BOOL, True),
            Parameter("record_dir", Parameter.Type.STRING, str(record_dir)),
            Parameter("render_video", Parameter.Type.BOOL, True),
            Parameter("render_format", Parameter.Type.STRING, "gif"),
        ],
    )

    executor = SingleThreadedExecutor()
    executor.add_node(mgr)

    try:
        for _ in range(5):
            mgr.tick()
            executor.spin_once(timeout_sec=0.05)
        for _ in range(5):
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.remove_node(mgr)
        mgr.destroy_node()

    assert (record_dir / "bag").exists(), "bag directory not created"

    proc = mgr._render_proc
    assert proc is not None, "renderer subprocess was not spawned"

    try:
        rc = proc.wait(timeout=120)
    except Exception:
        proc.kill()
        raise

    log_path = record_dir / "render.log"
    log_text = log_path.read_text() if log_path.exists() else "<no log>"
    assert rc == 0, f"renderer exited {rc}. log:\n{log_text}"

    output = record_dir / "scenario.gif"
    assert output.exists() and output.stat().st_size > 0, f"no output; log:\n{log_text}"
