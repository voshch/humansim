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
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import (
    CommandType,
    InteractionManager,
    _make_contract,
)
from arena_humansim.core.world_knowledge import AnchorKind, FormationSpec, WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    BehaviorTreeMovement,
    InteractionOutcome,
    InteractionState,
    Pose2D,
    SeekSpec,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


@dataclass
class _FakeState:
    agent_id: int = 0
    pose: Pose2D = field(default_factory=Pose2D)
    desired_velocity: float = 1.2
    kind: int = 0


@dataclass
class _FakeAgent:
    state: _FakeState
    params: _FakeParams = field(default_factory=_FakeParams)
    movement: BehaviorTreeMovement = field(default_factory=BehaviorTreeMovement)


def _mk_manager(agents: dict[int, _FakeAgent], world: WorldKnowledge | None = None) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world or WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid))
    return mgr


def _create(
    mgr: InteractionManager,
    agent_id: int,
    itype: InteractionType,
    *,
    target: str | int | None = None,
    offer: bool = False,
    duration: float | None = None,
    formation_spec: FormationSpec | None = None,
    min_participants: int | None = None,
    max_participants: int | None = None,
    queueable: bool | None = None,
) -> int:
    spec = SeekSpec(
        interaction_type=itype,
        target=target,
        offer=offer,
        duration=duration,
        formation_spec=formation_spec,
        min_participants=min_participants,
        max_participants=max_participants,
        queueable=queueable,
    )
    interaction = mgr._create_interaction(creator_id=agent_id, spec=spec)
    return interaction.id


def test_resolve_formation_uses_object_metadata_override() -> None:
    wk = WorldKnowledge()
    wk.add_object(
        WorldObject(
            object_id="counter",
            type="counter",
            pose=Pose2D(x=10.0, y=0.0, theta=0.0),
            formation=FormationSpec(type="line", params={"base_step": 0.5}),
        )
    )
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="counter")
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "LineFormation"
    assert formation.base_step == 0.5


def test_resolve_formation_falls_back_to_type_default() -> None:
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents)
    iid = _create(mgr, 1, InteractionType.GROUP_CONVERSATION)
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "FFormation"


def test_anchor_from_spec_object_kind() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=3.0, y=4.0)))
    wk.add_object(
        WorldObject(
            object_id="counter",
            type="counter",
            pose=Pose2D(x=10.0, y=0.0),
            formation=FormationSpec(type="line", anchor_kind=AnchorKind.OBJECT, anchor_ref="atm"),
        )
    )
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1))}
    mgr = _mk_manager(agents, world=wk)
    contract = _make_contract(InteractionType.QUEUE_USE)
    inter = InteractionState(contract=contract, object_id="counter", participants=[1])
    counter = wk.get("counter")
    assert counter is not None and counter.formation is not None
    anchor = mgr._anchor_from_spec(counter.formation, inter)
    assert isinstance(anchor, ObjectAnchor)
    assert anchor.pose() == Pose2D(x=3.0, y=4.0)


def test_anchor_from_spec_pose_kind() -> None:
    pose = Pose2D(x=2.0, y=2.0, theta=0.3)
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.POSE, anchor_pose=pose)
    agents: dict[int, _FakeAgent] = {}
    mgr = _mk_manager(agents)
    inter = InteractionState(contract=_make_contract(InteractionType.USE))
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, PoseAnchor)
    assert anchor.pose() == pose


def test_anchor_from_spec_agent_kind() -> None:
    agents = {7: _FakeAgent(state=_FakeState(agent_id=7, pose=Pose2D(x=9.0, y=9.0)))}
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.AGENT, anchor_ref="7")
    mgr = _mk_manager(agents)
    inter = InteractionState(contract=_make_contract(InteractionType.USE))
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, AgentAnchor)
    assert anchor.pose() == Pose2D(x=9.0, y=9.0)


def test_anchor_from_spec_centroid_kind() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=4.0, y=0.0))),
    }
    mgr = _mk_manager(agents)
    spec = FormationSpec(type="f_formation", anchor_kind=AnchorKind.CENTROID)
    inter = InteractionState(
        contract=_make_contract(InteractionType.GROUP_CONVERSATION),
        participants=[1, 2],
    )
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, CentroidAnchor)
    assert anchor.pose().x == pytest.approx(2.0)


def test_anchor_from_spec_provider_kind() -> None:
    agents = {5: _FakeAgent(state=_FakeState(agent_id=5, pose=Pose2D(x=7.0, y=0.0)))}
    mgr = _mk_manager(agents)
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.PROVIDER, params={"base_step": 0.8})
    inter = InteractionState(
        contract=_make_contract(InteractionType.SERVICE),
        participants=[5],
        state={"provider": 5},
    )
    anchor = mgr._anchor_from_spec(spec, inter)
    assert isinstance(anchor, AgentAnchor)
    assert anchor.pose() == Pose2D(x=7.0, y=0.0)


def test_anchor_from_spec_object_with_no_world_knowledge_returns_none() -> None:
    mgr = InteractionManager(RNG(0))  # no world_knowledge wired
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.OBJECT, anchor_ref="atm")
    inter = InteractionState(contract=_make_contract(InteractionType.QUEUE_USE))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_anchor_from_spec_agent_with_invalid_ref_returns_none() -> None:
    mgr = _mk_manager({})
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.AGENT, anchor_ref="not_an_int")
    inter = InteractionState(contract=_make_contract(InteractionType.USE))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_anchor_from_spec_provider_without_provider_returns_none() -> None:
    mgr = _mk_manager({})
    spec = FormationSpec(type="line", anchor_kind=AnchorKind.PROVIDER)
    inter = InteractionState(contract=_make_contract(InteractionType.SERVICE))
    assert mgr._anchor_from_spec(spec, inter) is None


def test_tick_formations_writes_navigate_to_member() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=5.0, y=0.0, theta=0.0)))
    agents = {1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D()))}
    mgr = _mk_manager(agents, world=wk)
    iid = _create(mgr, 1, InteractionType.QUEUE_USE, target="atm")
    _, formation_targets, _ = mgr.update({}, dt=0.05)
    assert 1 in formation_targets
    assert formation_targets[1] == Pose2D(x=5.0, y=0.0, theta=0.0)
    cmd = agents[1].movement.command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose == Pose2D(x=5.0, y=0.0, theta=0.0)
    assert cmd.desired_velocity == agents[1].state.desired_velocity
    assert mgr.interactions[iid].contract.formation is not None


def test_tick_formations_skips_when_agent_lookup_returns_none() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    agents: dict[int, _FakeAgent] = {}  # empty
    mgr = _mk_manager(agents, world=wk)
    _create(mgr, 99, InteractionType.QUEUE_USE, target="atm")
    mgr.update({}, dt=0.05)
    assert 99 in mgr._formation_targets


def test_service_provider_anchored_formation_via_spec() -> None:
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=1.0, y=0.0))),
    }
    mgr = _mk_manager(agents)
    fs = FormationSpec(type="line", anchor_kind=AnchorKind.PROVIDER, params={"base_step": 0.8})
    iid = _create(mgr, 1, InteractionType.SERVICE, target="water", offer=True, formation_spec=fs)
    formation = mgr.interactions[iid].contract.formation
    assert formation is not None
    assert type(formation).__name__ == "LineFormation"
    assert isinstance(formation.anchor, AgentAnchor)
    assert formation.anchor.agent_id == 1


def test_group_conversation_releases_formation_on_duration_expiry() -> None:
    # For 2-member F-formation with defaults: radius = base_radius + radius_per_member*(n-1) = 0.82.
    # Placing agents on that circle makes arrived() true so _tick_durations actually ticks.
    wk = WorldKnowledge()
    wk.add_object(
        WorldObject(
            object_id="gathering",
            type="gathering_area",
            pose=Pose2D(x=0.0, y=0.0),
            formation=FormationSpec(type="f_formation", anchor_kind=AnchorKind.POSE, anchor_pose=Pose2D(x=0.0, y=0.0)),
        )
    )
    agents = {
        1: _FakeAgent(state=_FakeState(agent_id=1, pose=Pose2D(x=0.82, y=0.0))),
        2: _FakeAgent(state=_FakeState(agent_id=2, pose=Pose2D(x=-0.82, y=0.0))),
    }
    mgr = _mk_manager(agents, world=wk)
    iid = _create(
        mgr,
        1,
        InteractionType.GROUP_CONVERSATION,
        duration=0.2,
        formation_spec=FormationSpec(type="f_formation", anchor_kind=AnchorKind.POSE, anchor_pose=Pose2D()),
    )
    assert mgr.accept(2, iid) is True
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE

    _, targets, departed = mgr.update({}, dt=0.15)
    assert iid in mgr.interactions
    assert set(targets.keys()) == {1, 2}
    assert departed == set()

    _, targets, departed = mgr.update({}, dt=0.15)
    assert iid not in mgr.interactions
    assert departed == {1, 2}
    assert targets == {}
    assert mgr._formation_targets == {}

    _, targets, departed = mgr.update({}, dt=0.05)
    assert targets == {}
    assert departed == set()
