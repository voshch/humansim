from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_humansim.utils.scenario import (
    ModuleConfig,
    ScenarioConfig,
    SimulationParams,
    load_scenario,
    resolve_extends,
)
from arena_humansim.utils.scenario_loader import converter


def test_load_scenario_round_trips_minimal_yaml(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    data = {
        "name": "from_disk",
        "description": "hello",
        "simulation": {"seed": 7, "dt": 0.1, "max_ticks": 50},
        "modules": {"local_planner": "sfm", "global_planner": "dijkstra"},
    }
    path.write_text(yaml.safe_dump(data))
    scn = load_scenario(str(path))
    assert scn.name == "from_disk"
    assert scn.description == "hello"
    assert scn.simulation.seed == 7
    assert scn.simulation.dt == 0.1
    assert scn.simulation.max_ticks == 50
    assert scn.modules.local_planner == "sfm"


def test_load_scenario_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario(str(tmp_path / "nope.yaml"))


def test_load_empty_yaml_yields_defaults(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("")
    scn = load_scenario(str(path))
    assert scn.name == "unnamed"
    assert isinstance(scn.simulation, SimulationParams)
    assert isinstance(scn.modules, ModuleConfig)


def test_missing_required_field_in_structuring_raises() -> None:
    from arena_humansim.core.agents.types import AgentType

    with pytest.raises(Exception):
        converter.structure({}, AgentType)


def test_resolve_extends_merges_scalar_from_parent(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    data = {
        "name": "ext",
        "simulation": {"seed": 1, "dt": 0.1, "max_ticks": 1},
        "modules": {},
        "agent_types": {
            "parent": {"desired_velocity": 2.0},
            "child": {"extends": "parent", "agent_radius": 0.5},
        },
    }
    path.write_text(yaml.safe_dump(data))
    scn = load_scenario(str(path))
    resolved_child = scn.agent_types["child"]
    assert resolved_child.desired_velocity.mean == 2.0
    assert resolved_child.agent_radius.mean == 0.5
    assert resolved_child.extends is None


def test_resolve_extends_merges_dict_fields_from_parent(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    data = {
        "name": "ext",
        "simulation": {"seed": 1, "dt": 0.1, "max_ticks": 1},
        "modules": {},
        "agent_types": {
            "parent": {"needs": {"hunger": {"decay_rate": 0.1}}},
            "child": {"extends": "parent", "needs": {"thirst": {"decay_rate": 0.2}}},
        },
    }
    path.write_text(yaml.safe_dump(data))
    scn = load_scenario(str(path))
    resolved_child = scn.agent_types["child"]
    assert "hunger" in resolved_child.needs
    assert "thirst" in resolved_child.needs
    assert resolved_child.extends is None


def test_resolve_extends_detects_cycle() -> None:
    a = {"name": "a", "extends": "b"}
    b = {"name": "b", "extends": "a"}
    with pytest.raises(ValueError, match="Circular"):
        resolve_extends({"a": a, "b": b}, {})


def test_resolve_extends_unknown_parent() -> None:
    child = {"name": "child", "extends": "ghost"}
    with pytest.raises(ValueError, match="not defined"):
        resolve_extends({"child": child}, {})


def test_minimal_scenario_fixture_valid(minimal_scenario: ScenarioConfig) -> None:
    assert minimal_scenario.name == "minimal"
    assert minimal_scenario.simulation.seed == 42
    assert minimal_scenario.simulation.dt == 0.05
    assert minimal_scenario.simulation.max_ticks == 10
    assert isinstance(minimal_scenario.modules, ModuleConfig)
    assert minimal_scenario.agents == []


def test_scenario_agent_kind_and_policy_parse(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    data = {
        "name": "robots",
        "simulation": {"seed": 1, "dt": 0.1, "max_ticks": 1},
        "modules": {},
        "agents": [
            {
                "agent_id": 0,
                "spawn_pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "goal_sequence": [{"x": 1.0, "y": 0.0}],
            },
            {
                "agent_id": 1,
                "kind": 1,
                "policy": "straight",
                "policy_params": "{}",
                "spawn_pose": {"x": 0.0, "y": 1.0, "theta": 0.0},
                "goal_sequence": [{"x": 2.0, "y": 1.0}],
            },
        ],
    }
    path.write_text(yaml.safe_dump(data))
    scn = load_scenario(str(path))
    human, robot = scn.agents
    assert human.kind == 0
    assert human.policy == ""
    assert robot.kind == 1
    assert robot.policy == "straight"
    assert robot.policy_params == "{}"


def test_robot_test_scenario_file_valid() -> None:
    scenario_path = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "robot_test.yaml"
    scn = load_scenario(str(scenario_path))
    assert scn.name == "robot_test"
    robots = [a for a in scn.agents if a.kind == 1]
    humans = [a for a in scn.agents if a.kind == 0]
    assert len(humans) >= 1
    assert len(robots) >= 2
    policies = {r.policy for r in robots}
    assert "straight" in policies
    assert "sfm" in policies
