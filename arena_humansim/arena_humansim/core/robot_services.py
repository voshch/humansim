from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.utils.scenario import ServiceSpec
from arena_humansim.utils.types import AgentKind, HighLevelCommand, InteractionType


class RobotServiceAdvertiser:
    """Synthesizes ADVERTISE HighLevelCommands for robots declaring `services`."""

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
            if agent is None or agent.state.kind != AgentKind.ROBOT:
                continue
            for svc in services:
                if not svc.tag:
                    continue
                out.append(
                    HighLevelCommand(
                        agent_id=aid,
                        type=CommandType.ADVERTISE,
                        interaction_type=int(InteractionType.SERVICE),
                        service_tag=svc.tag,
                        max_participants=svc.max_participants if svc.max_participants >= 0 else None,
                    )
                )
        return out
