from __future__ import annotations

import pytest
from arena_humansim_msgs.msg import WorldObjectInfo
from arena_humansim_msgs.srv import AddWorldObjects, RemoveWorldObjects
from geometry_msgs.msg import Pose2D as Pose2DMsg

from tests.ros._helpers import RosTestSystem

pytestmark = pytest.mark.ros


def _info(
    object_id: str,
    type_: str = "desk",
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    capacity: int = 1,
    satisfies: dict[str, float] | None = None,
    interaction_radius: float = 0.0,
    formation_type: str = "",
    formation_params: dict[str, float] | None = None,
) -> WorldObjectInfo:
    info = WorldObjectInfo()
    info.object_id = object_id
    info.type = type_
    info.pose = Pose2DMsg(x=x, y=y, theta=theta)
    info.capacity = capacity
    if satisfies:
        info.satisfies_keys = list(satisfies.keys())
        info.satisfies_values = [float(v) for v in satisfies.values()]
    info.interaction_radius = interaction_radius
    info.formation_type = formation_type
    if formation_params:
        info.formation_param_keys = list(formation_params.keys())
        info.formation_param_values = [float(v) for v in formation_params.values()]
    return info


def _add_request(infos: list[WorldObjectInfo]) -> AddWorldObjects.Request:
    req = AddWorldObjects.Request()
    req.objects = infos
    return req


def _remove_request(object_ids: list[str]) -> RemoveWorldObjects.Request:
    req = RemoveWorldObjects.Request()
    req.object_ids = list(object_ids)
    return req


@pytest.fixture()
def system(ros_system: RosTestSystem) -> RosTestSystem:
    # Clean slate for each test
    ros_system.call(RemoveWorldObjects, "remove_world_objects", _remove_request([]))
    return ros_system


def test_add_populates_world_knowledge(system: RosTestSystem) -> None:
    resp = system.call(
        AddWorldObjects,
        "add_world_objects",
        _add_request(
            [
                _info("desk_1", type_="desk", x=1.0, y=2.0, capacity=3, satisfies={"checkin": 80.0}, interaction_radius=1.0),
                _info("chair_1", type_="chair", x=5.0, y=6.0, capacity=1),
            ]
        ),
    )
    assert resp.success is True
    wk = system.manager._world_knowledge
    assert len(wk) == 2
    desk = wk.get("desk_1")
    assert desk is not None
    assert desk.type == "desk"
    assert desk.capacity == 3
    assert desk.satisfies == {"checkin": 80.0}
    assert desk.interaction_radius == pytest.approx(1.0)
    assert desk.pose.x == pytest.approx(1.0)
    assert desk.pose.y == pytest.approx(2.0)
    chair = wk.get("chair_1")
    assert chair is not None
    assert chair.interaction_radius is None  # zero -> None


def test_nearest_object_lookup_after_add(system: RosTestSystem) -> None:
    system.call(
        AddWorldObjects,
        "add_world_objects",
        _add_request([_info("desk_1", type_="desk", x=0.0, y=0.0)]),
    )
    wk = system.manager._world_knowledge
    from arena_humansim.utils.types import Pose2D

    obj = wk.nearest_object("desk", Pose2D(x=3.0, y=4.0))
    assert obj is not None
    assert obj.object_id == "desk_1"


def test_remove_by_id(system: RosTestSystem) -> None:
    system.call(
        AddWorldObjects,
        "add_world_objects",
        _add_request([_info("a"), _info("b"), _info("c")]),
    )
    resp = system.call(RemoveWorldObjects, "remove_world_objects", _remove_request(["a", "c"]))
    assert resp.success is True
    wk = system.manager._world_knowledge
    assert len(wk) == 1
    assert wk.get("a") is None
    assert wk.get("b") is not None
    assert wk.get("c") is None


def test_remove_all_with_empty_list(system: RosTestSystem) -> None:
    system.call(
        AddWorldObjects,
        "add_world_objects",
        _add_request([_info("a"), _info("b")]),
    )
    resp = system.call(RemoveWorldObjects, "remove_world_objects", _remove_request([]))
    assert resp.success is True
    assert len(system.manager._world_knowledge) == 0


def test_formation_roundtrips_into_world_knowledge(system: RosTestSystem) -> None:
    system.call(
        AddWorldObjects,
        "add_world_objects",
        _add_request(
            [
                _info(
                    "desk_1",
                    type_="desk",
                    capacity=3,
                    formation_type="line",
                    formation_params={"base_step": 0.8, "front_offset": 0.6},
                ),
                _info("chair_1", type_="chair"),  # no formation
            ]
        ),
    )
    wk = system.manager._world_knowledge
    desk = wk.get("desk_1")
    assert desk is not None
    assert desk.formation is not None
    assert desk.formation.type == "line"
    assert desk.formation.params == {"base_step": pytest.approx(0.8), "front_offset": pytest.approx(0.6)}

    chair = wk.get("chair_1")
    assert chair is not None
    assert chair.formation is None


def test_reset_clears_world_objects(system: RosTestSystem) -> None:
    from arena_humansim_msgs.srv import ResetSimulation

    system.call(
        AddWorldObjects,
        "add_world_objects",
        _add_request([_info("desk_1", x=1.0)]),
    )
    assert len(system.manager._world_knowledge) == 1
    system.call(ResetSimulation, "reset", ResetSimulation.Request())
    assert len(system.manager._world_knowledge) == 0
