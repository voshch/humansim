from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.logger import SimulationLogger
from arena_humansim.core.replay import ReplayManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3
from rclpy.parameter import Parameter

from ._helpers import build_manager


def _spawn(mgr: AgentManager, specs: list[tuple[float, float, float, float]]) -> list[int]:
    req = SpawnAgents.Request()
    for x, y, gx, gy in specs:
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=x, y=y, theta=0.0)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = 1.3
        msg.radius = 0.0
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=gx, y=gy, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
        req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return list(resp.spawned_ids)


def test_replay_produces_no_divergence(manager_factory: Callable[..., AgentManager], rclpy_context: object, minimal_scenario: ScenarioConfig, tmp_path: Path) -> None:  # noqa: ARG001
    specs = [
        (0.0, 0.0, 5.0, 0.0),
        (0.0, 2.0, 5.0, 2.0),
    ]
    n_ticks = 30
    log_dir = tmp_path / "session"

    record_mgr = build_manager(
        minimal_scenario,
        node_name="test_replay_record",
        extra_params=[Parameter("log_dir", Parameter.Type.STRING, str(log_dir))],
    )
    try:
        assert record_mgr._sim_logger is not None
        _spawn(record_mgr, specs)
        for _ in range(n_ticks):
            record_mgr.tick()
        assert isinstance(record_mgr._sim_logger, SimulationLogger)
        record_mgr._sim_logger.close()
    finally:
        record_mgr.destroy_node()

    log_path = log_dir / "session.jsonl"
    assert log_path.is_file()

    replay = ReplayManager()
    replay.load(str(log_path))
    assert replay.tick_count == n_ticks

    replay_mgr = manager_factory(minimal_scenario, node_name="test_replay_playback")
    _spawn(replay_mgr, specs)

    result = replay.replay(replay_mgr)
    assert result.success, f"replay diverged: {result.first_divergence}"
    assert result.first_divergence is None
    assert result.total_ticks == n_ticks


def test_replay_get_tick_roundtrip(rclpy_context: object, minimal_scenario: ScenarioConfig, tmp_path: Path) -> None:  # noqa: ARG001
    specs = [
        (0.0, 0.0, 4.0, 0.0),
        (1.0, 1.0, 4.0, 1.0),
    ]
    n_ticks = 10
    log_dir = tmp_path / "roundtrip"

    record_mgr = build_manager(
        minimal_scenario,
        node_name="test_replay_rt_record",
        extra_params=[Parameter("log_dir", Parameter.Type.STRING, str(log_dir))],
    )
    recorded: dict[int, dict[int, tuple[float, float, float]]] = {}
    try:
        assert record_mgr._sim_logger is not None
        spawned = _spawn(record_mgr, specs)
        for _ in range(n_ticks):
            tick_n = record_mgr._tick_count
            record_mgr.tick()
            recorded[tick_n] = {
                aid: (agent.state.pose.x, agent.state.pose.y, agent.state.pose.theta)
                for aid, agent in record_mgr._agents.items()
            }
        record_mgr._sim_logger.close()
    finally:
        record_mgr.destroy_node()

    log_path = log_dir / "session.jsonl"
    assert log_path.is_file()

    replay = ReplayManager()
    replay.load(str(log_path))
    assert replay.tick_count == n_ticks
    assert sorted(replay.spawned_agent_ids) == sorted(spawned)

    tol = 1e-9
    for tick_n, expected in recorded.items():
        loaded = replay.get_agents_at_tick(tick_n)
        assert set(loaded.keys()) == set(expected.keys()), f"tick {tick_n}: ids differ"
        for aid, (ex, ey, eth) in expected.items():
            st = loaded[aid]
            assert abs(st.pose.x - ex) <= tol
            assert abs(st.pose.y - ey) <= tol
            assert abs(st.pose.theta - eth) <= tol
