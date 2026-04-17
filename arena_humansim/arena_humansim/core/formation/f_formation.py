from __future__ import annotations

import math

from arena_humansim.utils.types import Pose2D

from . import AgentLookup, Formation
from .anchor import Anchor


class FFormation(Formation):
    """Kendon F-formation: circle of participants facing inward (o-space at centroid).

    Radius scales with member count so personal space is preserved as the circle grows.
    Per-member pose recomputes each tick from the current member list; no shuffle
    propagation (group reorganization is discrete).
    """

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        base_radius: float = 0.7,
        radius_per_member: float = 0.12,
        formation_scale: float = 1.0,
    ) -> None:
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.base_radius = base_radius
        self.radius_per_member = radius_per_member
        self.formation_scale = formation_scale
        self._members: list[int] = []

    def on_join(self, agent_id: int) -> None:
        if agent_id in self._members:
            return
        self._members.append(agent_id)

    def on_leave(self, agent_id: int) -> None:
        if agent_id in self._members:
            self._members.remove(agent_id)

    def _radius(self) -> float:
        n = max(len(self._members), 1)
        return (self.base_radius + self.radius_per_member * (n - 1)) * self.formation_scale

    def tick(self, dt: float) -> dict[int, Pose2D]:
        if not self._members:
            return {}
        center = self.anchor.pose()
        radius = self._radius()
        n = len(self._members)
        out: dict[int, Pose2D] = {}
        for i, aid in enumerate(self._members):
            angle = 2.0 * math.pi * i / n
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            inward = angle + math.pi
            out[aid] = Pose2D(x=x, y=y, theta=inward)
        return out
