from __future__ import annotations

from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.srv import ResetSimulation, SpawnAgents
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.parameter import Parameter

_HALF = '{"kind_gains": {"human_robot": {"strength_scale": 0.75, "range_scale": 0.65}}}'
_DEFAULT = (1.5, 1.3)


def _gain(planner: object, key: tuple[int, int] = (0, 1)) -> tuple[float, float]:
    assert isinstance(planner, SFMPlanner)
    return float(planner._gain_strength_scale[key]), float(planner._gain_range_scale[key])


def _param(value: str) -> list[Parameter]:
    return [Parameter("local_planner_params", Parameter.Type.STRING, value)]


def test_empty_param_keeps_planner_defaults(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    unset = manager_factory(minimal_scenario, node_name="lpp_unset")
    empty = manager_factory(minimal_scenario, node_name="lpp_empty", extra_params=_param(""))
    assert _gain(unset._local_planner) == _gain(empty._local_planner) == _DEFAULT


def test_param_sets_default_planner_gains(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="lpp_half", extra_params=_param(_HALF))
    assert _gain(mgr._local_planner) == (0.75, 0.65)
    assert _gain(mgr._local_planner, (0, 0)) == (1.0, 1.0)


def test_gains_survive_hard_and_soft_reset(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="lpp_reset", extra_params=_param(_HALF))
    for soft in (True, False):
        mgr._reset_callback(ResetSimulation.Request(soft=soft), ResetSimulation.Response())
        assert _gain(mgr._local_planner) == (0.75, 0.65)


def test_planner_selected_at_reset_gets_the_gains(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="lpp_switch", extra_params=_param(_HALF))
    (result,) = mgr.set_parameters([Parameter("local_planner", Parameter.Type.STRING, "hsfm")])
    assert result.successful
    mgr._reset_callback(ResetSimulation.Request(soft=False), ResetSimulation.Response())
    assert isinstance(mgr._local_planner, HSFMPlanner)
    assert _gain(mgr._local_planner) == (0.75, 0.65)


@pytest.mark.parametrize("bad", ["not json", '{"kind_gain": {}}', '{"kind_gains": {"human_robot": {"strength_scale": -1}}}'])
def test_bad_param_fails_startup(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig, bad: str) -> None:
    with pytest.raises(ValueError, match="local_planner_params for sfm"):
        manager_factory(minimal_scenario, node_name="lpp_bad", extra_params=_param(bad))


def test_planner_without_params_fails_startup(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    with pytest.raises(ValueError, match="takes no policy params"):
        manager_factory(
            minimal_scenario,
            node_name="lpp_orca",
            extra_params=[Parameter("local_planner", Parameter.Type.STRING, "orca"), *_param(_HALF)],
        )


def test_bad_agent_policy_params_keep_gains_and_spawn(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="lpp_agent")
    msg = AgentStateMsg(agent_id=0, pose=Pose2DMsg(x=0.0, y=0.0, theta=0.0), agent_type="adult", policy="sfm", policy_params="not json")
    response = mgr._spawn_agents_callback(SpawnAgents.Request(agents=[msg]), SpawnAgents.Response())
    assert response.success
    assert _gain(mgr._local_planner) == _DEFAULT
