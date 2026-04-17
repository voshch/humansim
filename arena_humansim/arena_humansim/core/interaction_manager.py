import dataclasses
import enum
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
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import (
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

# Formation defaults per interaction type: (strategy_name, default_params).
# Object metadata on WorldObject overrides these when present.
DEFAULT_FORMATION_BY_INTERACTION: dict[int, tuple[str, dict[str, Any]]] = {
    int(InteractionType.QUEUE_USE): ("line", {"base_step": 1.0}),
    int(InteractionType.SIT_ON): ("cluster", {}),
    int(InteractionType.LIE_ON): ("cluster", {}),
    int(InteractionType.GROUP_CONVERSATION): ("f_formation", {}),
    int(InteractionType.TALK_TO): ("dyad", {}),
}


def _make_contract(interaction_type: int) -> InteractionContract:
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
    else:
        contract = InteractionContract(type=interaction_type, min_participants=2, max_participants=2)
    return contract


class CommandType(enum.IntEnum):
    NAVIGATE = 0
    ADVERTISE = 1
    SEARCH = 2
    ACCEPT = 3
    DECLINE = 4
    STOP = 5
    IDLE = 6


@dataclasses.dataclass
class _Advertisement:
    agent_id: int
    interaction_type: int
    interaction_id: int | None = None


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
        self._rng = rng_manager.get_substream("interaction_manager")
        self._world_knowledge = world_knowledge
        self._agent_lookup: AgentLookup = agent_lookup or (lambda _aid: None)
        self._formation_scale = formation_scale
        self._formation_targets: dict[int, Pose2D] = {}  # agent_id -> target_pose from last formation tick

    def _pose_lookup(self, agent_id: int) -> Pose2D | None:
        agent = self._agent_lookup(agent_id)
        return agent.state.pose if agent is not None else None

    def set_context(
        self,
        world_knowledge: "WorldKnowledge | None" = None,
        agent_lookup: AgentLookup | None = None,
        formation_scale: float | None = None,
    ) -> None:
        """Wire world knowledge and agent lookup after construction (agent_manager init order)."""
        if world_knowledge is not None:
            self._world_knowledge = world_knowledge
        if agent_lookup is not None:
            self._agent_lookup = agent_lookup
        if formation_scale is not None:
            self._formation_scale = formation_scale

    def advertise(self, agent_id: int, interaction_type: int) -> _Advertisement:
        ad = _Advertisement(agent_id=agent_id, interaction_type=interaction_type)
        self._advertisements.setdefault(agent_id, []).append(ad)
        self._ads_by_type.setdefault(interaction_type, []).append(ad)
        return ad

    def search(self, agent_id: int, interaction_type: int) -> list[_Advertisement]:
        results: list[_Advertisement] = []
        for ad in self._ads_by_type.get(interaction_type, []):
            if ad.agent_id == agent_id:
                continue
            if ad.interaction_id is not None:
                interaction = self.interactions.get(ad.interaction_id)
                if interaction is None or interaction.contract.is_full:
                    continue
                target = interaction.state.get("target_agent", -1)
                if target >= 0 and target != agent_id:
                    continue
            results.append(ad)
        return results

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

    def decline(self, agent_id: int, interaction_id: int) -> None:
        self._remove_ads_for_interaction(agent_id, interaction_id)

    def stop(self, agent_id: int, interaction_id: int) -> InteractionState | None:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return None

        contract = interaction.contract
        was_involved = agent_id in contract.queue or agent_id in interaction.participants

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
    ) -> dict[int, InteractionState]:
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

        self._rng.shuffle(interaction_cmds)  # type: ignore[arg-type]

        for cmd in interaction_cmds:
            self._process_command(cmd)

        self._tick_formations(dt)
        self._prune_dead_interactions()
        return self.interactions

    def _tick_formations(self, dt: float) -> dict[int, Pose2D]:
        """Tick every interaction's formation; write NAVIGATE commands to member agents.

        Returns the flat agent_id -> target_pose mapping (also cached on self).
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
                if agent is None:
                    continue
                movement = getattr(agent, "movement", None)
                if movement is None:
                    continue
                movement.command = HighLevelCommand(
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
            ad = self.advertise(cmd.agent_id, cmd.interaction_type)
            object_id = getattr(cmd, "_object_id", None)
            interaction = self._create_interaction(
                cmd.interaction_type,
                cmd.agent_id,
                object_id=object_id,
                target_agent=cmd.target_agent,
                duration=cmd.interaction_duration,
            )
            ad.interaction_id = interaction.id

        elif ctype == CommandType.SEARCH:
            self.search(cmd.agent_id, cmd.interaction_type)

        elif ctype == CommandType.ACCEPT:
            target_id = cmd.interaction_target
            if target_id >= 0:
                self.accept(cmd.agent_id, target_id)
            else:
                ads = self.search(cmd.agent_id, cmd.interaction_type)
                from_agent = cmd.target_agent
                for ad in ads:
                    if from_agent >= 0 and ad.agent_id != from_agent:
                        continue
                    if ad.interaction_id is not None:
                        if self.accept(cmd.agent_id, ad.interaction_id):
                            break

        elif ctype == CommandType.DECLINE:
            target_id = cmd.interaction_target
            if target_id >= 0:
                self.decline(cmd.agent_id, target_id)

        elif ctype == CommandType.STOP:
            target_id = cmd.interaction_target
            if target_id >= 0:
                self.stop(cmd.agent_id, target_id)

    def _create_interaction(
        self,
        interaction_type: int,
        creator_id: int,
        object_id: str | None = None,
        target_agent: int = -1,
        duration: float = -1.0,
    ) -> InteractionState:
        contract = _make_contract(interaction_type)
        if duration > 0:
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

        contract.formation = self._resolve_formation(interaction)
        if contract.formation is not None:
            contract.formation.on_join(creator_id)

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
            default = DEFAULT_FORMATION_BY_INTERACTION.get(interaction.type)
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
        return getattr(obj, "formation", None)

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
            return PoseAnchor(fixed=Pose2D())
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
        if interaction.contract.is_full:
            return
        ads = self._advertisements.get(agent_id, [])
        for ad in ads:
            if ad.interaction_id == interaction.id:
                return
        ad = _Advertisement(
            agent_id=agent_id,
            interaction_type=interaction.type,
            interaction_id=interaction.id,
        )
        self._advertisements.setdefault(agent_id, []).append(ad)
        self._ads_by_type.setdefault(interaction.type, []).append(ad)

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
        self._logger.debug(f"Interaction {interaction_id} torn down (outcome={InteractionOutcome(outcome).name}, participants={interaction.participants})")
        for agent_id in list(interaction.participants):
            interactions_set = self._agent_to_interactions.get(agent_id)
            if interactions_set is not None:
                interactions_set.discard(interaction_id)
                if not interactions_set:
                    del self._agent_to_interactions[agent_id]
            self._on_formation_leave(interaction, agent_id)
        for agent_id in list(interaction.contract.queue):
            queues = self._agent_to_queues.get(agent_id)
            if queues is not None:
                queues.discard(interaction_id)
                if not queues:
                    del self._agent_to_queues[agent_id]
            self._on_formation_leave(interaction, agent_id)
        for agent_id in list(self._advertisements.keys()):
            self._remove_ads_for_interaction(agent_id, interaction_id)

    def _tick_durations(self, dt: float) -> None:
        for iid, interaction in list(self.interactions.items()):
            contract = interaction.contract
            if contract.duration is None or interaction.outcome != InteractionOutcome.ACTIVE:
                continue
            contract.elapsed += dt
            if contract.elapsed >= contract.duration:
                self._teardown(iid, InteractionOutcome.COMPLETED)

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
