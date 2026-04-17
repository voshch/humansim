from __future__ import annotations

from typing import TYPE_CHECKING

from . import AcceptResult, AccessPolicy

if TYPE_CHECKING:
    from arena_humansim.utils.types import InteractionState


class FIFOQueue(AccessPolicy):
    """First-in-first-out overflow queue for capacity-limited interactions.

    On accept while full: appends to contract.queue (up to max_queue).
    On tick: drains contract.queue into participants until capacity reached.
    On stop: removes from contract.queue.
    """

    def __init__(self, max_queue: int = -1) -> None:
        self.max_queue = max_queue

    def on_accept(self, interaction: InteractionState, agent_id: int) -> AcceptResult:
        contract = interaction.contract
        if agent_id in contract.queue:
            return AcceptResult.QUEUED
        if contract.is_full:
            cap = self.max_queue if self.max_queue != -1 else contract.max_queue
            if cap == -1 or len(contract.queue) < cap:
                contract.queue.append(agent_id)
                return AcceptResult.QUEUED
            return AcceptResult.REJECTED
        interaction.participants.append(agent_id)
        contract.current_participants.append(agent_id)
        return AcceptResult.BECAME_PARTICIPANT

    def tick(self, interaction: InteractionState, dt: float) -> list[int]:
        contract = interaction.contract
        promoted: list[int] = []
        while not contract.is_full and contract.queue:
            next_id = contract.queue.pop(0)
            interaction.participants.append(next_id)
            contract.current_participants.append(next_id)
            promoted.append(next_id)
        return promoted

    def on_stop(self, interaction: InteractionState, agent_id: int) -> None:
        contract = interaction.contract
        if agent_id in contract.queue:
            contract.queue.remove(agent_id)
