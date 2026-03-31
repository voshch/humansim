class EventBus:
    def __init__(self) -> None:
        self._events: dict[str, set[int] | None] = {}

    def fire(self, event_name: str, agent_id: int) -> None:
        agents = self._events.setdefault(event_name, set())
        if agents is not None:
            agents.add(agent_id)

    def fire_broadcast(self, event_name: str, agent_ids: set[int] | None = None) -> None:
        if agent_ids is None:
            self._events[event_name] = None
        else:
            agents = self._events.setdefault(event_name, set())
            if agents is not None:
                agents.update(agent_ids)

    def has(self, event_name: str, agent_id: int) -> bool:
        agents = self._events.get(event_name)
        if agents is None:
            return event_name in self._events
        return agent_id in agents

    def consume(self, event_name: str, agent_id: int) -> bool:
        agents = self._events.get(event_name)
        if agents is None:
            return event_name in self._events
        if agent_id in agents:
            agents.discard(agent_id)
            if not agents:
                del self._events[event_name]
            return True
        return False

    def clear(self) -> None:
        self._events.clear()

    def clear_agent(self, agent_id: int) -> None:
        for event_name in list(self._events.keys()):
            agents = self._events[event_name]
            if agents is None:
                continue
            agents.discard(agent_id)
            if not agents:
                del self._events[event_name]

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return bool(self._events)
