import dataclasses
import enum
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arena_humansim.core.access import AcceptResult, AccessPolicy
from arena_humansim.core.access.fifo_queue import FIFOQueue
from arena_humansim.core.formation import (
    AgentAnchor,
    Anchor,
    CentroidAnchor,
    Formation,
    ObjectAnchor,
    PoseAnchor,
)
from arena_humansim.core.world_knowledge import FormationSpec
from arena_humansim.utils import RNG
from arena_humansim.utils.const import DISTANCE_TOLERANCE
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import (
    AgentKind,
    BehaviorTreeMovement,
    HighLevelCommand,
    InteractionContract,
    InteractionOutcome,
    InteractionState,
    InteractionType,
    Pose2D,
)

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.world_knowledge import WorldKnowledge


AgentLookup = Callable[[int], "BaseAgent | None"]
VisibilityLookup = Callable[[int], "set[int]"]

# Formation defaults per interaction type: (strategy_name, default_params).
# Object metadata on WorldObject overrides these when present.
DEFAULT_FORMATION_BY_INTERACTION: dict[InteractionType, tuple[str, dict[str, Any]]] = {
    InteractionType.QUEUE_USE: ("line", {"base_step": 1.0}),
    InteractionType.SIT_ON: ("cluster", {}),
    InteractionType.LIE_ON: ("cluster", {}),
    InteractionType.GROUP_CONVERSATION: ("f_formation", {}),
    InteractionType.TALK_TO: ("dyad", {}),
    InteractionType.SERVICE: ("f_formation", {}),
    InteractionType.FOLLOW: ("line", {"base_step": 0.8}),
}


# How close a BT-driven agent must be to its target before it may advertise.
# Cascade: StepDef override > WorldObject override > this type default > DISTANCE_TOLERANCE.
DEFAULT_INTERACTION_RADIUS: dict[InteractionType, float] = {
    InteractionType.USE: DISTANCE_TOLERANCE,
    InteractionType.QUEUE_USE: DISTANCE_TOLERANCE,
    InteractionType.SIT_ON: DISTANCE_TOLERANCE,
    InteractionType.LIE_ON: DISTANCE_TOLERANCE,
    InteractionType.GROUP_CONVERSATION: 3.0,
    InteractionType.TALK_TO: 2.0,
    InteractionType.FOLLOW: 5.0,
    InteractionType.SERVICE: 3.0,
}


def interaction_radius_for(interaction_type: "InteractionType | int") -> float:
    key = interaction_type if isinstance(interaction_type, InteractionType) else InteractionType(interaction_type)
    return DEFAULT_INTERACTION_RADIUS.get(key, DISTANCE_TOLERANCE)


def _make_contract(interaction_type: int, max_participants: int | None = None) -> InteractionContract:
    it = InteractionType(interaction_type)
    if it == InteractionType.TALK_TO:
        contract = InteractionContract(type=interaction_type, min_participants=2, max_participants=2)
    elif it == InteractionType.GROUP_CONVERSATION:
        contract = InteractionContract(type=interaction_type, min_participants=2, max_participants=-1)
    elif it == InteractionType.FOLLOW:
        contract = InteractionContract(type=interaction_type, min_participants=2, max_participants=2)
    elif it in (InteractionType.SIT_ON, InteractionType.LIE_ON):
        contract = InteractionContract(type=interaction_type, min_participants=1, max_participants=1, queueable=True)
        contract.access = FIFOQueue()
    elif it == InteractionType.USE:
        contract = InteractionContract(type=interaction_type, min_participants=1, max_participants=1, queueable=True)
        contract.access = FIFOQueue()
    elif it == InteractionType.QUEUE_USE:
        contract = InteractionContract(type=interaction_type, min_participants=1, max_participants=1, queueable=True)
        contract.access = FIFOQueue()
    elif it == InteractionType.SERVICE:
        contract = InteractionContract(type=interaction_type, min_participants=1, max_participants=-1, queueable=True)
        contract.access = FIFOQueue()
    else:
        contract = InteractionContract(type=interaction_type, min_participants=2, max_participants=2)
    if max_participants is not None:
        contract.max_participants = max_participants
    return contract


class CommandType(enum.IntEnum):
    NAVIGATE = 0
    ADVERTISE = 1
    STOP = 5
    IDLE = 6


@dataclasses.dataclass
class _Advertisement:
    """An agent's declaration of intent to participate in an interaction of a given type.

    Unbound (interaction_id is None): the matcher will resolve this ad into an interaction,
    either by joining an existing one or pairing with another ad.

    Bound (interaction_id set): the agent already participates in that interaction; the ad
    is kept around so others can join via the matcher.
    """

    agent_id: int
    interaction_type: int
    interaction_id: int | None = None
    object_id: str | None = None
    target_agent: int = -1
    interaction_target: int = -1
    duration: float | None = None
    service_tag: str | None = None
    max_participants: int | None = None


class InteractionManager(Loggable):
    def __init__(
        self,
        rng_manager: RNG,
        world_knowledge: "WorldKnowledge | None" = None,
        agent_lookup: AgentLookup | None = None,
        formation_scale: float = 1.0,
    ):
        self.rng_manager = rng_manager
        self.interactions: dict[int, InteractionState] = {}
        self.next_interaction_id: int = 0
        self._advertisements: dict[int, list[_Advertisement]] = {}
        self._ads_by_type: dict[int, list[_Advertisement]] = {}  # interaction_type -> ads
        self._agent_to_interactions: dict[int, set[int]] = {}  # agent_id -> set of interaction_ids
        self._agent_to_queues: dict[int, set[int]] = {}  # agent_id -> set of interaction_ids they're queued in
        self._interaction_by_object_type: dict[tuple[str, int], int] = {}  # (object_id, type) -> interaction_id
        self._rng = rng_manager.get_substream("interaction_manager")
        self._world_knowledge = world_knowledge
        self._agent_lookup: AgentLookup = agent_lookup or (lambda _aid: None)
        self._visibility_lookup: VisibilityLookup | None = None
        self._formation_scale = formation_scale
        self._formation_targets: dict[int, Pose2D] = {}  # agent_id -> target_pose from last formation tick
        self._current_departed: set[int] = set()

    def _pose_lookup(self, agent_id: int) -> Pose2D | None:
        agent = self._agent_lookup(agent_id)
        return agent.state.pose if agent is not None else None

    def set_context(
        self,
        world_knowledge: "WorldKnowledge | None" = None,
        agent_lookup: AgentLookup | None = None,
        formation_scale: float | None = None,
        visibility_lookup: VisibilityLookup | None = None,
    ) -> None:
        """Wire world knowledge, agent lookup, and perception-visibility callback after construction."""
        if world_knowledge is not None:
            self._world_knowledge = world_knowledge
        if agent_lookup is not None:
            self._agent_lookup = agent_lookup
        if formation_scale is not None:
            self._formation_scale = formation_scale
        if visibility_lookup is not None:
            self._visibility_lookup = visibility_lookup

    def _post_ad(self, cmd: HighLevelCommand) -> _Advertisement:
        """Create or refresh an unbound ad for the advertising agent.

        Idempotent: if the agent already has an unbound ad with matching targeting
        fields, that ad is returned (with duration refreshed). Otherwise a fresh
        unbound ad is appended.
        """
        agent_id = cmd.agent_id
        itype = cmd.interaction_type
        object_id = cmd.object_id
        target_agent = cmd.target_agent
        interaction_target = cmd.interaction_target
        duration = cmd.interaction_duration
        service_tag = cmd.service_tag
        max_participants = cmd.max_participants

        for ad in self._advertisements.get(agent_id, []):
            if ad.interaction_id is not None:
                continue
            if ad.interaction_type != itype:
                continue
            if ad.object_id != object_id:
                continue
            if ad.target_agent != target_agent:
                continue
            if ad.interaction_target != interaction_target:
                continue
            if ad.service_tag != service_tag:
                continue
            ad.duration = duration
            ad.max_participants = max_participants
            return ad

        ad = _Advertisement(
            agent_id=agent_id,
            interaction_type=itype,
            object_id=object_id,
            target_agent=target_agent,
            interaction_target=interaction_target,
            duration=duration,
            service_tag=service_tag,
            max_participants=max_participants,
        )
        self._advertisements.setdefault(agent_id, []).append(ad)
        self._ads_by_type.setdefault(itype, []).append(ad)
        return ad

    def accept(self, agent_id: int, interaction_id: int) -> bool:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return False
        target = interaction.state.get("target_agent", -1)
        if target >= 0 and target != agent_id:
            return False
        if agent_id in interaction.participants:
            return True
        if agent_id in interaction.contract.queue:
            return True

        contract = interaction.contract
        access: AccessPolicy | None = contract.access
        if access is not None:
            result = access.on_accept(interaction, agent_id)
            if result == AcceptResult.REJECTED:
                return False
            if result == AcceptResult.QUEUED:
                self._agent_to_queues.setdefault(agent_id, set()).add(interaction_id)
                self._on_formation_join(interaction, agent_id)
                return True
            # BECAME_PARTICIPANT
            self._agent_to_interactions.setdefault(agent_id, set()).add(interaction_id)
            self._maybe_activate(interaction)
            self._readvertise_for_participant(agent_id, interaction)
            self._on_formation_join(interaction, agent_id)
            return True

        # Backward-compat path: no access policy set
        if contract.is_full:
            if contract.queueable:
                if contract.max_queue == -1 or len(contract.queue) < contract.max_queue:
                    contract.queue.append(agent_id)
                    self._agent_to_queues.setdefault(agent_id, set()).add(interaction_id)
                    self._on_formation_join(interaction, agent_id)
                    return True
            return False

        interaction.participants.append(agent_id)
        contract.current_participants.append(agent_id)
        self._agent_to_interactions.setdefault(agent_id, set()).add(interaction_id)
        self._maybe_activate(interaction)
        self._readvertise_for_participant(agent_id, interaction)
        self._on_formation_join(interaction, agent_id)
        return True

    def _on_formation_join(self, interaction: InteractionState, agent_id: int) -> None:
        formation: Formation | None = interaction.contract.formation
        if formation is None:
            return
        formation.on_join(agent_id)

    def _on_formation_leave(self, interaction: InteractionState, agent_id: int) -> None:
        formation: Formation | None = interaction.contract.formation
        if formation is None:
            return
        formation.on_leave(agent_id)
        self._formation_targets.pop(agent_id, None)

    def stop(self, agent_id: int, interaction_id: int) -> InteractionState | None:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return None

        contract = interaction.contract
        was_involved = agent_id in contract.queue or agent_id in interaction.participants
        if was_involved:
            self._current_departed.add(agent_id)

        access: AccessPolicy | None = contract.access
        if access is not None:
            access.on_stop(interaction, agent_id)
        elif agent_id in contract.queue:
            contract.queue.remove(agent_id)

        queues = self._agent_to_queues.get(agent_id)
        if queues is not None:
            queues.discard(interaction_id)
            if not queues:
                del self._agent_to_queues[agent_id]

        if agent_id in interaction.participants:
            interaction.participants.remove(agent_id)
            interactions_set = self._agent_to_interactions.get(agent_id)
            if interactions_set is not None:
                interactions_set.discard(interaction_id)
                if not interactions_set:
                    del self._agent_to_interactions[agent_id]
        if agent_id in contract.current_participants:
            contract.current_participants.remove(agent_id)

        if was_involved:
            interaction.member_durations.pop(agent_id, None)
            self._on_formation_leave(interaction, agent_id)

        self._remove_ads_for_interaction(agent_id, interaction_id)

        if len(interaction.participants) < contract.min_participants:
            self._teardown(interaction_id, InteractionOutcome.INTERRUPTED)
            return None

        self._readvertise_all_participants(interaction)
        return interaction

    def update(
        self,
        high_level_commands: dict[int, HighLevelCommand],
        dt: float = 0.0,
        extra_commands: list[HighLevelCommand] | None = None,
    ) -> tuple[dict[int, InteractionState], dict[int, Pose2D], set[int]]:
        self._current_departed = set()
        self._prune_ended_interactions()
        self._tick_access(dt)
        self._tick_durations(dt)

        interaction_cmds: list[HighLevelCommand] = []
        for agent_id in sorted(high_level_commands.keys()):
            cmd = high_level_commands[agent_id]
            if not isinstance(cmd, HighLevelCommand):
                continue
            if cmd.type == CommandType.NAVIGATE:
                continue
            interaction_cmds.append(cmd)

        if extra_commands:
            for cmd in extra_commands:
                if isinstance(cmd, HighLevelCommand) and cmd.type != CommandType.NAVIGATE:
                    interaction_cmds.append(cmd)

        self._rng.shuffle(interaction_cmds)  # type: ignore[arg-type]

        for cmd in interaction_cmds:
            self._process_command(cmd)

        self._match_ads()

        self._tick_formations(dt)
        self._prune_dead_interactions()
        self._prune_ended_interactions()
        return self.interactions, dict(self._formation_targets), set(self._current_departed)

    def _tick_formations(self, dt: float) -> dict[int, Pose2D]:
        """Tick every interaction's formation; return the flat agent_id -> target_pose mapping.

        For BT agents we also overwrite movement.command so the BT's persisted ADVERTISE
        doesn't stomp the formation target on the next BT tick. AgentManager separately
        applies `_formation_targets` to `_high_level_cmds` for non-BT agents.
        """
        targets: dict[int, Pose2D] = {}
        for interaction in self.interactions.values():
            formation: Formation | None = interaction.contract.formation
            if formation is None:
                continue
            per_formation = formation.tick(dt)
            for aid, pose in per_formation.items():
                targets[aid] = pose
                agent = self._agent_lookup(aid)
                if agent is None or not isinstance(agent.movement, BehaviorTreeMovement):
                    continue
                agent.movement.command = HighLevelCommand(
                    agent_id=aid,
                    type=int(CommandType.NAVIGATE),
                    target_pose=pose,
                    desired_velocity=agent.state.desired_velocity,
                )
        self._formation_targets = targets
        return targets

    def formation_target(self, agent_id: int) -> Pose2D | None:
        return self._formation_targets.get(agent_id)

    def is_in_interaction(self, agent_id: int) -> bool:
        return bool(self._agent_to_interactions.get(agent_id))

    def is_in_queue(self, agent_id: int) -> bool:
        return bool(self._agent_to_queues.get(agent_id))

    def force_stop(self, agent_id: int) -> None:
        for iid in list(self._agent_to_interactions.get(agent_id, ())):
            self.stop(agent_id, iid)
        for iid in list(self._agent_to_queues.get(agent_id, ())):
            interaction = self.interactions.get(iid)
            if interaction is None:
                continue
            contract = interaction.contract
            access: AccessPolicy | None = contract.access
            if access is not None:
                access.on_stop(interaction, agent_id)
            elif agent_id in contract.queue:
                contract.queue.remove(agent_id)
            self._on_formation_leave(interaction, agent_id)
        self._agent_to_interactions.pop(agent_id, None)
        self._agent_to_queues.pop(agent_id, None)
        # Drop any surviving ads (unbound ones that self.stop wouldn't have touched).
        for ad in self._advertisements.pop(agent_id, []):
            type_list = self._ads_by_type.get(ad.interaction_type)
            if type_list is None:
                continue
            try:
                type_list.remove(ad)
            except ValueError:
                pass
            if not type_list:
                del self._ads_by_type[ad.interaction_type]

    def queue_length_for_object(self, object_id: str) -> int:
        total = 0
        for interaction in self.interactions.values():
            if interaction.object_id == object_id:
                total += interaction.contract.queue_length
        return total

    def _tick_access(self, dt: float) -> None:
        for interaction in self.interactions.values():
            contract = interaction.contract
            access: AccessPolicy | None = contract.access

            if access is not None:
                promoted = access.tick(interaction, dt)
                for next_agent in promoted:
                    queues = self._agent_to_queues.get(next_agent)
                    if queues is not None:
                        queues.discard(interaction.id)
                        if not queues:
                            del self._agent_to_queues[next_agent]
                    self._agent_to_interactions.setdefault(next_agent, set()).add(interaction.id)
                if promoted:
                    self._maybe_activate(interaction)
                continue

            # Backward-compat: legacy queueable path for manually constructed contracts
            if not contract.queueable or not contract.queue:
                continue
            while not contract.is_full and contract.queue:
                next_agent = contract.queue.pop(0)
                queues = self._agent_to_queues.get(next_agent)
                if queues is not None:
                    queues.discard(interaction.id)
                    if not queues:
                        del self._agent_to_queues[next_agent]
                interaction.participants.append(next_agent)
                contract.current_participants.append(next_agent)
                self._agent_to_interactions.setdefault(next_agent, set()).add(interaction.id)
                self._maybe_activate(interaction)

    def _process_command(self, cmd: HighLevelCommand) -> None:
        ctype = cmd.type

        if ctype == CommandType.ADVERTISE:
            self._post_ad(cmd)
        elif ctype == CommandType.STOP:
            target_id = cmd.interaction_target
            if target_id >= 0:
                self.stop(cmd.agent_id, target_id)
            else:
                self.force_stop(cmd.agent_id)

    def _match_ads(self) -> None:
        """Resolve unbound ads against existing interactions or other pending ads.

        One tick, one pass. Unbound ads left unmatched persist for next tick. Pairing
        (targeted-agent and open branches) may bind two ads in one iteration; the guard
        in the loop skips ads that were bound as the partner of an earlier ad.
        """
        unbound = [ad for ads in self._advertisements.values() for ad in ads if ad.interaction_id is None]
        unbound.sort(key=lambda a: a.agent_id)
        for ad in unbound:
            if ad.interaction_id is not None:
                continue
            if ad not in self._advertisements.get(ad.agent_id, []):
                continue
            self._bind_ad(ad)

    def _bind_ad(self, ad: _Advertisement) -> bool:
        """Dispatch an unbound ad to the most specific binding branch that applies.

        Specificity descends: a named interaction > a named object > a named peer > open.
        SERVICE ads take their own branch regardless of target_agent/object_id.
        """
        if ad.interaction_type == InteractionType.SERVICE:
            return self._bind_service(ad)
        if ad.interaction_target >= 0:
            return self._bind_targeted_interaction(ad)
        if ad.object_id is not None:
            return self._bind_object_anchored(ad)
        if ad.target_agent >= 0:
            return self._bind_targeted_agent(ad)
        return self._bind_open(ad)

    def _bind_service(self, ad: _Advertisement) -> bool:
        """Pair SERVICE ads by matching service_tag; robot is anchor/initiator."""
        peer = self._find_visible_service_peer(ad)
        if peer is None:
            return False
        if peer.interaction_id is not None:
            interaction = self.interactions.get(peer.interaction_id)
            if interaction is None:
                return False
            return self._join_ad_to_interaction(ad, interaction)
        robot_ad, seeker_ad = self._order_service_pair(ad, peer)
        merged_cap = robot_ad.max_participants if robot_ad.max_participants is not None else seeker_ad.max_participants
        interaction = self._create_interaction(
            ad.interaction_type,
            robot_ad.agent_id,
            object_id=None,
            target_agent=-1,
            duration=robot_ad.duration if robot_ad.duration is not None else seeker_ad.duration,
            max_participants=merged_cap,
        )
        if robot_ad.duration is not None and robot_ad.duration > 0:
            interaction.member_durations[robot_ad.agent_id] = robot_ad.duration
        robot_ad.interaction_id = interaction.id
        if ad is robot_ad:
            return self._join_ad_to_interaction(seeker_ad, interaction)
        return self._join_ad_to_interaction(ad, interaction)

    def _order_service_pair(self, ad: _Advertisement, peer: _Advertisement) -> tuple[_Advertisement, _Advertisement]:
        """Return (robot_ad, seeker_ad); robot_ad wins anchor + max_participants."""
        ad_agent = self._agent_lookup(ad.agent_id)
        peer_agent = self._agent_lookup(peer.agent_id)
        ad_is_robot = ad_agent is not None and ad_agent.state.kind == AgentKind.ROBOT
        peer_is_robot = peer_agent is not None and peer_agent.state.kind == AgentKind.ROBOT
        if peer_is_robot and not ad_is_robot:
            return peer, ad
        return ad, peer

    def _find_visible_service_peer(self, self_ad: _Advertisement) -> _Advertisement | None:
        if self._visibility_lookup is None:
            return None
        visible = self._visibility_lookup(self_ad.agent_id)
        if not visible:
            return None
        agent_pose = self._pose_lookup(self_ad.agent_id)
        if agent_pose is None:
            return None
        self_agent = self._agent_lookup(self_ad.agent_id)
        self_is_robot = self_agent is not None and self_agent.state.kind == AgentKind.ROBOT
        best: _Advertisement | None = None
        best_d = float("inf")
        for candidate in self._ads_by_type.get(self_ad.interaction_type, []):
            if candidate is self_ad:
                continue
            if candidate.agent_id == self_ad.agent_id:
                continue
            if candidate.service_tag != self_ad.service_tag:
                continue
            if candidate.agent_id not in visible:
                continue
            candidate_agent = self._agent_lookup(candidate.agent_id)
            candidate_is_robot = candidate_agent is not None and candidate_agent.state.kind == AgentKind.ROBOT
            if self_is_robot == candidate_is_robot:
                continue
            if candidate.interaction_id is not None:
                interaction = self.interactions.get(candidate.interaction_id)
                if interaction is None:
                    continue
                if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED):
                    continue
                if interaction.contract.is_full and not interaction.contract.queueable:
                    continue
            candidate_pose = self._pose_lookup(candidate.agent_id)
            if candidate_pose is None:
                continue
            d = math.hypot(agent_pose.x - candidate_pose.x, agent_pose.y - candidate_pose.y)
            if d < best_d:
                best_d = d
                best = candidate
        return best

    def _bind_targeted_interaction(self, ad: _Advertisement) -> bool:
        """Ad names a specific interaction id — join it if still alive."""
        interaction = self.interactions.get(ad.interaction_target)
        if interaction is None:
            return False
        if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED):
            return False
        return self._join_ad_to_interaction(ad, interaction)

    def _bind_object_anchored(self, ad: _Advertisement) -> bool:
        """Ad names a world object — join the existing (object, type) interaction or create one."""
        object_id = ad.object_id
        assert object_id is not None  # guaranteed by dispatcher
        existing = self._find_interaction_for_object(ad.interaction_type, object_id)
        if existing is not None:
            return self._join_ad_to_interaction(ad, existing)
        interaction = self._create_interaction(
            ad.interaction_type,
            ad.agent_id,
            object_id=object_id,
            target_agent=ad.target_agent,
            duration=ad.duration,
        )
        if ad.duration is not None and ad.duration > 0:
            interaction.member_durations[ad.agent_id] = ad.duration
        ad.interaction_id = interaction.id
        return True

    def _bind_targeted_agent(self, ad: _Advertisement) -> bool:
        """Ad names a specific peer — join their interaction or pair with their ad."""
        other = self._find_ad_from_agent(ad.interaction_type, ad.target_agent)
        if other is None:
            return False
        return self._join_via_partner(ad, other)

    def _bind_open(self, ad: _Advertisement) -> bool:
        """Ad names nothing — match against any visible same-type peer (perception gate)."""
        candidate = self._find_visible_open_ad(ad)
        if candidate is None:
            return False
        return self._join_via_partner(ad, candidate)

    def _join_via_partner(self, ad: _Advertisement, partner: _Advertisement) -> bool:
        """Join `ad` to the interaction `partner` is in — creating one if needed."""
        if partner.interaction_id is not None:
            interaction = self.interactions.get(partner.interaction_id)
            if interaction is None:
                return False
            return self._join_ad_to_interaction(ad, interaction)
        # Both unbound: create a new interaction with `partner` as creator, then join `ad`.
        interaction = self._create_interaction(
            ad.interaction_type,
            partner.agent_id,
            object_id=None,
            target_agent=partner.target_agent,
            duration=partner.duration if partner.duration is not None else ad.duration,
        )
        if partner.duration is not None and partner.duration > 0:
            interaction.member_durations[partner.agent_id] = partner.duration
        partner.interaction_id = interaction.id
        return self._join_ad_to_interaction(ad, interaction)

    def _join_ad_to_interaction(
        self,
        ad: _Advertisement,
        interaction: InteractionState,
    ) -> bool:
        if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED):
            return False
        if not self.accept(ad.agent_id, interaction.id):
            return False
        if ad.duration is not None and ad.duration > 0:
            interaction.member_durations[ad.agent_id] = ad.duration
        # accept() ran _readvertise_for_participant, which upgraded this ad's
        # interaction_id in place. Nothing more to do.
        return True

    def _find_ad_from_agent(self, interaction_type: int, agent_id: int) -> _Advertisement | None:
        for ad in self._ads_by_type.get(interaction_type, []):
            if ad.agent_id != agent_id:
                continue
            if ad.interaction_id is not None:
                interaction = self.interactions.get(ad.interaction_id)
                if interaction is None:
                    continue
                if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED):
                    continue
                if interaction.contract.is_full:
                    continue
            return ad
        return None

    def _find_visible_open_ad(self, self_ad: _Advertisement) -> _Advertisement | None:
        if self._visibility_lookup is None:
            return None
        visible = self._visibility_lookup(self_ad.agent_id)
        if not visible:
            return None
        agent_pose = self._pose_lookup(self_ad.agent_id)
        if agent_pose is None:
            return None
        best: _Advertisement | None = None
        best_d = float("inf")
        for candidate in self._ads_by_type.get(self_ad.interaction_type, []):
            if candidate is self_ad:
                continue
            if candidate.agent_id == self_ad.agent_id:
                continue
            if candidate.agent_id not in visible:
                continue
            # Targeted ads aimed at someone else are off-limits.
            if candidate.target_agent >= 0 and candidate.target_agent != self_ad.agent_id:
                continue
            if candidate.interaction_id is None:
                # Unbound candidate: only pair if it's also "open" — specific
                # targeting (object_id / interaction_target) means it belongs to
                # a higher-priority rule, not this one.
                if candidate.object_id is not None:
                    continue
                if candidate.interaction_target >= 0:
                    continue
            if candidate.interaction_id is not None:
                interaction = self.interactions.get(candidate.interaction_id)
                if interaction is None:
                    continue
                if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED):
                    continue
                if interaction.contract.is_full:
                    continue
                target = interaction.state.get("target_agent", -1)
                if target >= 0 and target != self_ad.agent_id:
                    continue
            candidate_pose = self._pose_lookup(candidate.agent_id)
            if candidate_pose is None:
                continue
            d = math.hypot(agent_pose.x - candidate_pose.x, agent_pose.y - candidate_pose.y)
            if d < best_d:
                best_d = d
                best = candidate
        return best

    def _remove_ad(self, ad: _Advertisement) -> None:
        ads = self._advertisements.get(ad.agent_id)
        if ads is not None:
            try:
                ads.remove(ad)
            except ValueError:
                pass
            if not ads:
                del self._advertisements[ad.agent_id]
        type_list = self._ads_by_type.get(ad.interaction_type)
        if type_list is not None:
            try:
                type_list.remove(ad)
            except ValueError:
                pass
            if not type_list:
                del self._ads_by_type[ad.interaction_type]

    def _create_interaction(
        self,
        interaction_type: int,
        creator_id: int,
        object_id: str | None = None,
        target_agent: int = -1,
        duration: float | None = None,
        max_participants: int | None = None,
    ) -> InteractionState:
        contract = _make_contract(interaction_type, max_participants=max_participants)
        if duration is not None and duration > 0:
            contract.duration = duration
        iid = self.next_interaction_id
        self.next_interaction_id += 1

        state_dict: dict[str, Any] = {}
        if target_agent >= 0:
            state_dict["target_agent"] = target_agent
        if interaction_type == InteractionType.FOLLOW:
            state_dict["asymmetric_roles"] = True
            state_dict["leader"] = creator_id

        interaction = InteractionState(
            id=iid,
            type=interaction_type,
            contract=contract,
            participants=[creator_id],
            state=state_dict,
            object_id=object_id,
        )
        contract.current_participants.append(creator_id)
        self._agent_to_interactions.setdefault(creator_id, set()).add(iid)
        self.interactions[iid] = interaction
        if object_id is not None:
            self._interaction_by_object_type[(object_id, interaction_type)] = iid

        contract.formation = self._resolve_formation(interaction)
        if contract.formation is not None:
            contract.formation.on_join(creator_id)

        # Ensure the creator has a bound ad so the matcher can discover this interaction
        # via rules 3 (target_agent) and 4 (nearby) — consistent with how joiners get
        # readvertised in `accept()`.
        self._readvertise_for_participant(creator_id, interaction)

        self._maybe_activate(interaction)
        self._logger.debug(f"Interaction {iid} created: type={InteractionType(interaction_type).name}, creator={creator_id}")
        return interaction

    def _resolve_formation(self, interaction: InteractionState) -> Formation | None:
        spec = self._object_formation_spec(interaction.object_id)
        if spec is not None:
            name = spec.type
            params = dict(spec.params or {})
            anchor = self._anchor_from_spec(spec, interaction)
        else:
            default = DEFAULT_FORMATION_BY_INTERACTION.get(InteractionType(interaction.type))
            if default is None:
                return None
            name = default[0]
            params = dict(default[1])
            anchor = self._build_anchor_for(interaction)

        if anchor is None:
            return None

        try:
            return Formation.create(
                name,
                anchor=anchor,
                agent_lookup=self._agent_lookup,
                formation_scale=self._formation_scale,
                **params,
            )
        except (KeyError, TypeError) as e:
            self._logger.warning(f"Formation '{name}' instantiation failed for interaction {interaction.id}: {e}")
            return None

    def _object_formation_spec(self, object_id: str | None) -> FormationSpec | None:
        if object_id is None or self._world_knowledge is None:
            return None
        obj = self._world_knowledge.get(object_id)
        if obj is None:
            return None
        return obj.formation

    def _anchor_from_spec(self, spec: FormationSpec, interaction: InteractionState) -> Anchor | None:
        kind = spec.anchor_kind
        if kind == "object":
            ref = spec.anchor_ref or interaction.object_id
            if ref and self._world_knowledge is not None:
                return ObjectAnchor(world_knowledge=self._world_knowledge, object_id=ref)
            return None
        if kind == "agent":
            try:
                aid = int(spec.anchor_ref) if spec.anchor_ref is not None else -1
            except (TypeError, ValueError):
                aid = -1
            if aid >= 0:
                return AgentAnchor(pose_lookup=self._pose_lookup, agent_id=aid)
            return None
        if kind == "pose":
            return PoseAnchor(fixed=spec.anchor_pose or Pose2D())
        if kind == "centroid":
            members_ref = interaction.participants
            return CentroidAnchor(
                pose_lookup=self._pose_lookup,
                members_fn=lambda: list(members_ref),
            )
        return None

    def _build_anchor_for(self, interaction: InteractionState) -> Anchor | None:
        it = InteractionType(interaction.type)
        if it in (InteractionType.QUEUE_USE, InteractionType.USE, InteractionType.SIT_ON, InteractionType.LIE_ON):
            if interaction.object_id and self._world_knowledge is not None:
                return ObjectAnchor(world_knowledge=self._world_knowledge, object_id=interaction.object_id)
            creator_pose = self._pose_lookup(interaction.participants[0]) if interaction.participants else None
            return PoseAnchor(fixed=creator_pose or Pose2D())
        if it in (InteractionType.GROUP_CONVERSATION, InteractionType.TALK_TO):
            members_ref = interaction.participants
            return CentroidAnchor(
                pose_lookup=self._pose_lookup,
                members_fn=lambda: list(members_ref),
            )
        if it == InteractionType.FOLLOW:
            leader = interaction.state.get("leader", -1)
            if leader >= 0:
                return AgentAnchor(pose_lookup=self._pose_lookup, agent_id=leader)
        return None

    def _readvertise_for_participant(self, agent_id: int, interaction: InteractionState) -> None:
        ads = self._advertisements.get(agent_id, [])
        for ad in ads:
            if ad.interaction_id == interaction.id:
                return
        # Upgrade an existing unbound ad of matching type rather than duplicating it.
        # We do this even when the interaction is full so stale unbound ads can't be
        # rematched by the matcher after the agent is already a participant.
        for ad in ads:
            if ad.interaction_id is None and ad.interaction_type == interaction.type:
                ad.interaction_id = interaction.id
                return
        # No unbound ad to upgrade. Only post a fresh bound ad when the interaction
        # still has room for more — full interactions don't recruit.
        if interaction.contract.is_full:
            return
        new_ad = _Advertisement(
            agent_id=agent_id,
            interaction_type=interaction.type,
            interaction_id=interaction.id,
        )
        self._advertisements.setdefault(agent_id, []).append(new_ad)
        self._ads_by_type.setdefault(interaction.type, []).append(new_ad)

    def _readvertise_all_participants(self, interaction: InteractionState) -> None:
        if interaction.contract.is_full:
            for pid in interaction.participants:
                self._remove_ads_for_interaction(pid, interaction.id)
            return
        for pid in interaction.participants:
            self._readvertise_for_participant(pid, interaction)

    def _remove_ads_for_interaction(self, agent_id: int, interaction_id: int) -> None:
        ads = self._advertisements.get(agent_id)
        if ads is None:
            return
        removed = [ad for ad in ads if ad.interaction_id == interaction_id]
        self._advertisements[agent_id] = [ad for ad in ads if ad.interaction_id != interaction_id]
        if not self._advertisements[agent_id]:
            del self._advertisements[agent_id]
        for ad in removed:
            type_list = self._ads_by_type.get(ad.interaction_type)
            if type_list is not None:
                try:
                    type_list.remove(ad)
                except ValueError:
                    pass
                if not type_list:
                    del self._ads_by_type[ad.interaction_type]

    def _teardown(
        self,
        interaction_id: int,
        outcome: int = InteractionOutcome.INTERRUPTED,
    ) -> None:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return
        interaction.outcome = outcome
        if interaction.object_id is not None:
            self._interaction_by_object_type.pop((interaction.object_id, interaction.type), None)
        self._logger.debug(f"Interaction {interaction_id} torn down (outcome={InteractionOutcome(outcome).name}, participants={interaction.participants})")
        for agent_id in list(interaction.participants):
            self._current_departed.add(agent_id)
            interactions_set = self._agent_to_interactions.get(agent_id)
            if interactions_set is not None:
                interactions_set.discard(interaction_id)
                if not interactions_set:
                    del self._agent_to_interactions[agent_id]
            self._on_formation_leave(interaction, agent_id)
        for agent_id in list(interaction.contract.queue):
            self._current_departed.add(agent_id)
            queues = self._agent_to_queues.get(agent_id)
            if queues is not None:
                queues.discard(interaction_id)
                if not queues:
                    del self._agent_to_queues[agent_id]
            self._on_formation_leave(interaction, agent_id)
            self._clear_bt_movement(agent_id, outcome=outcome)
        for agent_id in list(interaction.participants):
            self._clear_bt_movement(agent_id, outcome=outcome)
        interaction.member_durations.clear()
        for agent_id in list(self._advertisements.keys()):
            self._remove_ads_for_interaction(agent_id, interaction_id)

    def _clear_bt_movement(self, agent_id: int, outcome: int | None = None) -> None:
        agent = self._agent_lookup(agent_id)
        if agent is None or not isinstance(agent.movement, BehaviorTreeMovement):
            return
        agent.movement.command = None
        if outcome is not None:
            agent.movement.last_outcome = outcome

    def _tick_durations(self, dt: float) -> None:
        for iid, interaction in list(self.interactions.items()):
            contract = interaction.contract
            if contract.duration is None or interaction.outcome != InteractionOutcome.ACTIVE:
                continue
            formation = contract.formation
            if formation is not None and interaction.participants:
                if not all(formation.arrived(pid) for pid in interaction.participants):
                    continue
            contract.elapsed += dt
            if contract.elapsed < contract.duration:
                continue
            if contract.access is not None and contract.queue:
                for pid in list(interaction.participants):
                    self._release_participant(interaction, pid)
                promoted = contract.access.tick(interaction, 0.0)
                for next_agent in promoted:
                    queues = self._agent_to_queues.get(next_agent)
                    if queues is not None:
                        queues.discard(iid)
                        if not queues:
                            del self._agent_to_queues[next_agent]
                    self._agent_to_interactions.setdefault(next_agent, set()).add(iid)
                if interaction.participants:
                    active_id = interaction.participants[0]
                    stored = interaction.member_durations.get(active_id)
                    if stored is not None and stored > 0:
                        contract.duration = stored
                contract.elapsed = 0.0
                continue
            self._teardown(iid, InteractionOutcome.COMPLETED)

    def _release_participant(self, interaction: InteractionState, agent_id: int) -> None:
        self._current_departed.add(agent_id)
        contract = interaction.contract
        if agent_id in interaction.participants:
            interaction.participants.remove(agent_id)
        if agent_id in contract.current_participants:
            contract.current_participants.remove(agent_id)
        interactions_set = self._agent_to_interactions.get(agent_id)
        if interactions_set is not None:
            interactions_set.discard(interaction.id)
            if not interactions_set:
                del self._agent_to_interactions[agent_id]
        interaction.member_durations.pop(agent_id, None)
        self._remove_ads_for_interaction(agent_id, interaction.id)
        self._on_formation_leave(interaction, agent_id)
        self._clear_bt_movement(agent_id, outcome=InteractionOutcome.COMPLETED)

    def _find_interaction_for_object(self, interaction_type: int, object_id: str) -> InteractionState | None:
        iid = self._interaction_by_object_type.get((object_id, interaction_type))
        if iid is None:
            return None
        interaction = self.interactions.get(iid)
        if interaction is None:
            return None
        if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED):
            return None
        return interaction

    def _maybe_activate(self, interaction: InteractionState) -> None:
        if interaction.outcome != InteractionOutcome.FORMING:
            return
        if len(interaction.participants) >= interaction.contract.min_participants:
            interaction.outcome = InteractionOutcome.ACTIVE

    def _prune_ended_interactions(self) -> None:
        to_remove = [iid for iid, interaction in self.interactions.items() if interaction.outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.INTERRUPTED)]
        for iid in to_remove:
            del self.interactions[iid]

    def _prune_dead_interactions(self) -> None:
        to_remove = [iid for iid, interaction in self.interactions.items() if interaction.outcome == InteractionOutcome.ACTIVE and len(interaction.participants) < interaction.contract.min_participants]
        for iid in to_remove:
            self._teardown(iid)
