from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation import (
    AgentAnchor,
    CentroidAnchor,
    ObjectAnchor,
    PoseAnchor,
)
from arena_humansim.core.interaction_manager import (
    CommandType,
    InteractionManager,
    _make_contract,
)
from arena_humansim.core.world_knowledge import FormationSpec, WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionType,
    Pose2D,
)


@dataclass
class _FakeMovement:
    command: HighLevelCommand | None = None
    last_outcome: int | None = None


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


@dataclass
class _FakeState:
    agent_id: int = 0
    pose: Pose2D = field(default_factory=Pose2D)
    desired_velocity: float = 1.2


@dataclass
class _FakeAgent:
    state: _FakeState
    params: _FakeParams = field(default_factory=_FakeParams)
    movement: _FakeMovement = field(default_factory=_FakeMovement)


def _mk_manager(agents: dict[int, _FakeAgent], world: WorldKnowledge | None = None) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world or WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid))
    return mgr


def _advertise(mgr: InteractionManager, agent_id: int, itype: int, object_id: str | None = None) -> int:
    if object_id is not None:
        iid = mgr.next_interaction_id
        mgr._create_interaction(int(itype), agent_id, object_id=object_id)
        return iid
    cmd = HighLevelCommand(
        agent_id=agent_id,
        type=int(CommandType.ADVERTISE),
        interaction_type=int(itype),
    )
    mgr.update({agent_id: cmd})
    ads = mgr._advertisements.get(agent_id, [])
    assert ads and ads[-1].interaction_id is not None
    return ads[-1].interaction_id


def test_resolve_formation_uses_object_metadata_override() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(
        object_id="counter",
        type="counter",
        pose=Pose2D(x=10.0, y=0.0, theta=0.0),
        formation=FormationSpec(type="line", params={"base_step": 0.5}),
    ))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents, world=wk)
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, object_id="counter")
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "LineFormation"
    assert formation.base_step == 0.5


def test_resolve_formation_falls_back_to_type_default() -> None:
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents)
    iid = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "FFormation"


def test_anchor_from_spec_object_kind() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=3.0, y=4.0)))
    wk.add_object(WorldObject(
        object_id="counter",
        type="counter",
        pose=Pose2D(x=10.0, y=0.0),
        formation=FormationSpec(type="line", anchor_kind="object", anchor_ref="atm"),
    ))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents, world=wk)
    contract = _make_contract(int(InteractionType.QUEUE_USE))
    from arena_humansim.utils.types import InteractionState
    inter = InteractionState(contract=contract, object_id="counter", participants=[1])
    counter = wk.get("counter")
    assert counter is not None and counter.formation is not None
    anchor = mgr._anchor_from_spec(counter.formation, inter)
    assert isinstance(anchor, ObjectAnchor)
    assert anchor.pose() == Pose2D(x=3.0, y=4.0)


def test_anchor_from_spec_pose_kind() -> None:
    pose = Pose2D(x=2.0, y=2.0, theta=0.3)
    spec = FormationSpec(type="line", anchor_kind="pose", anchor_pose=pose)
    agents: dict[int, _FakeAgent] = {}
    mgr = _mk_manager(agents)
    from arena_humansim.utils.types import InteractionState
    inter = InteractionState(contract=_make_contract(int(InteractionType.USE)))
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, PoseAnchor)
    assert anchor.pose() == pose


def test_anchor_from_spec_agent_kind() -> None:
    agents = {7: _FakeAgent(state=_FakeState(agent_id=7, pose=Pose2D(x=9.0, y=9.0)))}
    spec = FormationSpec(type="line", anchor_kind="agent", anchor_ref="7")
    mgr = _mk_manager(agents)
    from arena_humansim.utils.types import InteractionState
    inter = InteractionState(contract=_make_contract(int(InteractionType.USE)))
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, AgentAnchor)
    assert anchor.pose() == Pose2D(x=9.0, y=9.0)


def test_anchor_from_spec_centroid_kind() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=4.0, y=0.0))),
    }
    mgr = _mk_manager(agents)
    spec = FormationSpec(type="f_formation", anchor_kind="centroid")
    from arena_humansim.utils.types import InteractionState
    inter = InteractionState(
        contract=_make_contract(int(InteractionType.GROUP_CONVERSATION)),
        participants=[1, 2],
    )
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, CentroidAnchor)
    assert anchor.pose().x == pytest.approx(2.0)


def test_anchor_from_spec_object_with_no_world_knowledge_returns_none() -> None:
    mgr = InteractionManager(RNG(0))  # no world_knowledge wired
    spec = FormationSpec(type="line", anchor_kind="object", anchor_ref="atm")
    from arena_humansim.utils.types import InteractionState
    inter = InteractionState(contract=_make_contract(int(InteractionType.QUEUE_USE)))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_anchor_from_spec_agent_with_invalid_ref_returns_none() -> None:
    mgr = _mk_manager({})
    spec = FormationSpec(type="line", anchor_kind="agent", anchor_ref="not_an_int")
    from arena_humansim.utils.types import InteractionState
    inter = InteractionState(contract=_make_contract(int(InteractionType.USE)))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_tick_formations_writes_navigate_to_member() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=5.0, y=0.0, theta=0.0)))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D()))}
    mgr = _mk_manager(agents, world=wk)
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, object_id="atm")
    # Run another update to tick formations after creation
    mgr.update({}, dt=0.05)
    cmd = agents[1].movement.command
    assert cmd is not None
    assert cmd.type == int(CommandType.NAVIGATE)
    assert cmd.target_pose == Pose2D(x=5.0, y=0.0, theta=0.0)
    assert cmd.desired_velocity == agents[1].state.desired_velocity
    assert mgr.interactions[iid].contract.formation is not None


def test_tick_formations_skips_when_agent_lookup_returns_none() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    agents: dict[int, _FakeAgent] = {}  # empty
    mgr = _mk_manager(agents, world=wk)
    mgr._create_interaction(int(InteractionType.QUEUE_USE), 99, object_id="atm")
    mgr.update({}, dt=0.05)
    assert 99 in mgr._formation_targets
