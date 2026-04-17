from __future__ import annotations

from typing import TYPE_CHECKING

from . import AcceptResult, AccessPolicy

if TYPE_CHECKING:
    from arena_humansim.utils.types import InteractionState


class NoAccess(AccessPolicy):
    """Capacity gate without overflow queue. Rejects when full."""

    def on_accept(self, interaction: InteractionState, agent_id: int) -> AcceptResult:
        contract = interaction.contract
        if contract.is_full:
            return AcceptResult.REJECTED
        interaction.participants.append(agent_id)
        contract.current_participants.append(agent_id)
        return AcceptResult.BECAME_PARTICIPANT

    def tick(self, interaction: InteractionState, dt: float) -> list[int]:
        return []

    def on_stop(self, interaction: InteractionState, agent_id: int) -> None:
        pass
