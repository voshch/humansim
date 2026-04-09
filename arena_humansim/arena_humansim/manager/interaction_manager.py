import dataclasses
import enum
from typing import Any

from arena_humansim.utils import RNG
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionContract,
    InteractionOutcome,
    InteractionState,
    InteractionType,
)


def _make_contract(interaction_type: int) -> InteractionContract:
    it = InteractionType(interaction_type)
    if it == InteractionType.TALK_TO:
        return InteractionContract(
            type=interaction_type, min_participants=2, max_participants=2
        )
    elif it == InteractionType.GROUP_CONVERSATION:
        return InteractionContract(
            type=interaction_type, min_participants=2, max_participants=-1
        )
    elif it == InteractionType.FOLLOW:
        return InteractionContract(
            type=interaction_type, min_participants=2, max_participants=2
        )
    elif it in (InteractionType.SIT_ON, InteractionType.LIE_ON):
        return InteractionContract(
            type=interaction_type, min_participants=1, max_participants=1
        )
    elif it == InteractionType.USE:
        return InteractionContract(
            type=interaction_type, min_participants=1, max_participants=1
        )
    elif it == InteractionType.QUEUE_USE:
        return InteractionContract(
            type=interaction_type,
            min_participants=1,
            max_participants=1,
            queueable=True,
        )
    else:
        return InteractionContract(
            type=interaction_type, min_participants=2, max_participants=2
        )


class CommandType(enum.IntEnum):
    NAVIGATE = 0
    ADVERTISE = 1
    SEARCH = 2
    ACCEPT = 3
    DECLINE = 4
    STOP = 5


@dataclasses.dataclass
class _Advertisement:
    agent_id: int
    interaction_type: int
    interaction_id: int | None = None


class InteractionManager(Loggable):
    def __init__(self, rng_manager: RNG):
        self.rng_manager = rng_manager
        self.interactions: dict[int, InteractionState] = {}
        self.next_interaction_id: int = 0
        self._advertisements: dict[int, list[_Advertisement]] = {}
        self._ads_by_type: dict[
            int, list[_Advertisement]
        ] = {}  # interaction_type -> ads
        self._agent_to_interactions: dict[
            int, set[int]
        ] = {}  # agent_id -> set of interaction_ids
        self._agent_to_queues: dict[
            int, set[int]
        ] = {}  # agent_id -> set of interaction_ids they're queued in
        self._rng = rng_manager.get_substream("interaction_manager")

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

        if interaction.contract.is_full:
            contract = interaction.contract
            if contract.queueable:
                if contract.max_queue == -1 or len(contract.queue) < contract.max_queue:
                    contract.queue.append(agent_id)
                    self._agent_to_queues.setdefault(agent_id, set()).add(
                        interaction_id
                    )
                    return True
            return False

        interaction.participants.append(agent_id)
        interaction.contract.current_participants.append(agent_id)
        self._agent_to_interactions.setdefault(agent_id, set()).add(interaction_id)
        self._readvertise_for_participant(agent_id, interaction)
        return True

    def decline(self, agent_id: int, interaction_id: int) -> None:
        self._remove_ads_for_interaction(agent_id, interaction_id)

    def stop(self, agent_id: int, interaction_id: int) -> InteractionState | None:
        interaction = self.interactions.get(interaction_id)
        if interaction is None:
            return None

        if agent_id in interaction.contract.queue:
            interaction.contract.queue.remove(agent_id)
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
        if agent_id in interaction.contract.current_participants:
            interaction.contract.current_participants.remove(agent_id)

        self._remove_ads_for_interaction(agent_id, interaction_id)

        if len(interaction.participants) < interaction.contract.min_participants:
            self._teardown(interaction_id, InteractionOutcome.INTERRUPTED)
            return None

        self._readvertise_all_participants(interaction)
        return interaction

    def update(
        self,
        high_level_commands: dict[int, Any],
        dt: float = 0.0,
    ) -> dict[int, InteractionState]:
        self._prune_ended_interactions()
        self._tick_queues()
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

        self._prune_dead_interactions()
        return self.interactions

    def is_in_interaction(self, agent_id: int) -> bool:
        return bool(self._agent_to_interactions.get(agent_id))

    def is_in_queue(self, agent_id: int) -> bool:
        return bool(self._agent_to_queues.get(agent_id))

    def force_stop(self, agent_id: int) -> None:
        for iid in list(self._agent_to_interactions.get(agent_id, ())):
            self.stop(agent_id, iid)
        for iid in list(self._agent_to_queues.get(agent_id, ())):
            interaction = self.interactions.get(iid)
            if interaction and agent_id in interaction.contract.queue:
                interaction.contract.queue.remove(agent_id)
        self._agent_to_interactions.pop(agent_id, None)
        self._agent_to_queues.pop(agent_id, None)

    def queue_length_for_object(self, object_id: str) -> int:
        total = 0
        for interaction in self.interactions.values():
            if interaction.object_id == object_id:
                total += interaction.contract.queue_length
        return total

    def _tick_queues(self) -> None:
        for interaction in self.interactions.values():
            contract = interaction.contract
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
                self._agent_to_interactions.setdefault(next_agent, set()).add(
                    interaction.id
                )

    def _process_command(self, cmd: HighLevelCommand) -> None:
        ctype = cmd.type

        if ctype == CommandType.ADVERTISE:
            ad = self.advertise(cmd.agent_id, cmd.interaction_type)
            object_id = cmd.__dict__.get("_object_id")
            interaction = self._create_interaction(
                cmd.interaction_type,
                cmd.agent_id,
                object_id=object_id,
                target_agent=cmd.target_agent,
                duration=cmd.interaction_duration,
            )
            ad.interaction_id = interaction.id

        elif ctype == CommandType.SEARCH:
            results = self.search(cmd.agent_id, cmd.interaction_type)
            cmd.__dict__["_search_results"] = [
                ad.interaction_id for ad in results if ad.interaction_id is not None
            ]

        elif ctype == CommandType.ACCEPT:
            target_id = cmd.interaction_target
            if target_id >= 0:
                self.accept(cmd.agent_id, target_id)
            else:
                ads = self.search(cmd.agent_id, cmd.interaction_type)
                for ad in ads:
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

        state_dict: dict = {}
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
        self._logger.debug(
            f"Interaction {iid} created: type={InteractionType(interaction_type).name}, creator={creator_id}"
        )
        return interaction

    def _readvertise_for_participant(
        self, agent_id: int, interaction: InteractionState
    ) -> None:
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
        self._advertisements[agent_id] = [
            ad for ad in ads if ad.interaction_id != interaction_id
        ]
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
        self._logger.debug(
            f"Interaction {interaction_id} torn down (outcome={InteractionOutcome(outcome).name}, participants={interaction.participants})"
        )
        for agent_id in list(interaction.participants):
            interactions_set = self._agent_to_interactions.get(agent_id)
            if interactions_set is not None:
                interactions_set.discard(interaction_id)
                if not interactions_set:
                    del self._agent_to_interactions[agent_id]
        for agent_id in list(interaction.contract.queue):
            queues = self._agent_to_queues.get(agent_id)
            if queues is not None:
                queues.discard(interaction_id)
                if not queues:
                    del self._agent_to_queues[agent_id]
        for agent_id in list(self._advertisements.keys()):
            self._remove_ads_for_interaction(agent_id, interaction_id)

    def _tick_durations(self, dt: float) -> None:
        for iid, interaction in list(self.interactions.items()):
            contract = interaction.contract
            if (
                contract.duration is None
                or interaction.outcome != InteractionOutcome.ACTIVE
            ):
                continue
            contract.elapsed += dt
            if contract.elapsed >= contract.duration:
                self._teardown(iid, InteractionOutcome.COMPLETED)

    def _prune_ended_interactions(self) -> None:
        to_remove = [
            iid
            for iid, interaction in self.interactions.items()
            if interaction.outcome != InteractionOutcome.ACTIVE
        ]
        for iid in to_remove:
            del self.interactions[iid]

    def _prune_dead_interactions(self) -> None:
        to_remove = [
            iid
            for iid, interaction in self.interactions.items()
            if interaction.outcome == InteractionOutcome.ACTIVE
            and len(interaction.participants) < interaction.contract.min_participants
        ]
        for iid in to_remove:
            self._teardown(iid)
