from __future__ import annotations

import pytest

from arena_humansim.agents import resolve_agent_type
from arena_humansim.agents.types import AgentType, LocalPlannerDist, ParamDist, PerceptionDist


def test_resolve_no_extends_returns_same_object() -> None:
    agent = AgentType(name="plain")
    result = resolve_agent_type(agent, registry={})
    assert result is agent


def test_resolve_unknown_parent_raises_keyerror() -> None:
    child = AgentType(name="child", extends="missing")
    with pytest.raises(KeyError):
        resolve_agent_type(child, registry={})


def test_resolve_child_overrides_scalar_fields() -> None:
    parent = AgentType(
        name="parent",
        desired_velocity=ParamDist(2.0),
        agent_radius=ParamDist(0.5),
    )
    child = AgentType(
        name="child",
        extends="parent",
        desired_velocity=ParamDist(3.3),
    )
    registry = {"parent": parent}
    resolved = resolve_agent_type(child, registry=registry)

    assert resolved.name == "child"
    assert resolved.extends is None
    assert resolved.desired_velocity == ParamDist(3.3)
    assert resolved.agent_radius == ParamDist(0.5)


def test_resolve_nested_perception_merges_field_by_field() -> None:
    parent = AgentType(
        name="parent",
        perception=PerceptionDist(
            vision_range=ParamDist(9.0),
            vision_fov=ParamDist(270.0),
        ),
    )
    child = AgentType(
        name="child",
        extends="parent",
        perception=PerceptionDist(vision_range=ParamDist(12.5)),
    )
    resolved = resolve_agent_type(child, registry={"parent": parent})

    assert resolved.perception.vision_range == ParamDist(12.5)
    assert resolved.perception.vision_fov == ParamDist(270.0)


def test_resolve_nested_local_planner_params_merges() -> None:
    parent = AgentType(
        name="parent",
        local_planner_params=LocalPlannerDist(
            relaxation_time=ParamDist(1.0),
            repulsion_strength=ParamDist(4.0),
            repulsion_range=ParamDist(0.9),
            anisotropy=ParamDist(0.8),
        ),
    )
    child = AgentType(
        name="child",
        extends="parent",
        local_planner_params=LocalPlannerDist(repulsion_strength=ParamDist(7.7)),
    )
    resolved = resolve_agent_type(child, registry={"parent": parent})

    assert resolved.local_planner_params.repulsion_strength == ParamDist(7.7)
    assert resolved.local_planner_params.relaxation_time == ParamDist(1.0)
    assert resolved.local_planner_params.repulsion_range == ParamDist(0.9)
    assert resolved.local_planner_params.anisotropy == ParamDist(0.8)


def test_resolve_multilevel_extends_chain() -> None:
    grandparent = AgentType(
        name="gp",
        desired_velocity=ParamDist(0.5),
        agent_radius=ParamDist(0.2),
        max_velocity=ParamDist(1.0),
    )
    parent = AgentType(
        name="parent",
        extends="gp",
        agent_radius=ParamDist(0.9),
    )
    child = AgentType(
        name="child",
        extends="parent",
        max_velocity=ParamDist(5.5),
    )
    registry = {"gp": grandparent, "parent": parent}
    resolved = resolve_agent_type(child, registry=registry)

    assert resolved.desired_velocity == ParamDist(0.5)
    assert resolved.agent_radius == ParamDist(0.9)
    assert resolved.max_velocity == ParamDist(5.5)
    assert resolved.extends is None


def test_resolve_respects_custom_registry_arg() -> None:
    custom_parent = AgentType(name="builtin_name", desired_velocity=ParamDist(42.0))
    child = AgentType(name="child", extends="builtin_name")
    resolved = resolve_agent_type(child, registry={"builtin_name": custom_parent})
    assert resolved.desired_velocity == ParamDist(42.0)

    with pytest.raises(KeyError):
        resolve_agent_type(child, registry={})
