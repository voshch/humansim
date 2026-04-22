from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import ServiceSpec
from arena_humansim.utils.types import CommandType, HighLevelCommand, SeekSpec


class RobotServiceAdvertiser:
    """Synthesizes SEEK HighLevelCommands (with offer=True) for robots declaring `services`."""

    def __init__(self) -> None:
        self._robot_services: dict[int, list[ServiceSpec]] = {}

    def register(self, agent_id: int, services: list[ServiceSpec]) -> None:
        if services:
            self._robot_services[agent_id] = list(services)

    def unregister(self, agent_id: int) -> None:
        self._robot_services.pop(agent_id, None)

    def emit(self, agents: dict[int, BaseAgent]) -> list[HighLevelCommand]:
        out: list[HighLevelCommand] = []
        for aid, services in self._robot_services.items():
            agent = agents.get(aid)
            if agent is None:
                continue
            for svc in services:
                if not svc.tag:
                    continue
                spec = SeekSpec(
                    interaction_type=InteractionType.SERVICE,
                    target=svc.tag,
                    offer=True,
                    max_participants=svc.max_participants if svc.max_participants >= 0 else None,
                )
                out.append(
                    HighLevelCommand(
                        agent_id=aid,
                        type=CommandType.SEEK,
                        spec=spec,
                    )
                )
        return out
