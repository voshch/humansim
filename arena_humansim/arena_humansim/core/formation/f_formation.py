from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D, pose_distance

from . import AgentLookup, Formation
from .anchor import Anchor


class FFormation(Formation):
    """Kendon F-formation: circle of participants facing inward (o-space at centroid).

    Radius scales with member count so personal space is preserved as the circle grows.
    Slots (angular positions) are Hungarian-matched to current member poses on join/leave
    to minimize total travel and avoid path crossings during approach.
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
        self._slot_by_member: dict[int, int] = {}

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        if agent_id in self._members:
            return
        self._members.append(agent_id)
        self._reassign()

    def on_leave(self, agent_id: int) -> None:
        if agent_id in self._members:
            self._members.remove(agent_id)
            self._reassign()

    def _radius(self) -> float:
        n = max(len(self._members), 1)
        return (self.base_radius + self.radius_per_member * (n - 1)) * self.formation_scale

    def _slot_pose(self, slot_index: int, n: int, center: Pose2D, radius: float) -> Pose2D:
        angle = 2.0 * math.pi * slot_index / n
        return Pose2D(x=center.x + radius * math.cos(angle), y=center.y + radius * math.sin(angle), theta=angle + math.pi)

    def _reassign(self) -> None:
        n = len(self._members)
        if n == 0:
            self._slot_by_member = {}
            return
        poses: list[Pose2D] = []
        for aid in self._members:
            agent = self.agent_lookup(aid)
            if agent is None:
                self._slot_by_member = {mid: i for i, mid in enumerate(self._members)}
                return
            poses.append(agent.state.pose)
        center = self.anchor.pose()
        radius = self._radius()
        slot_xy = [(center.x + radius * math.cos(2.0 * math.pi * i / n), center.y + radius * math.sin(2.0 * math.pi * i / n)) for i in range(n)]
        cost = np.array([[math.hypot(p.x - sx, p.y - sy) for sx, sy in slot_xy] for p in poses])
        row_ind, col_ind = linear_sum_assignment(cost)
        self._slot_by_member = {self._members[int(r)]: int(c) for r, c in zip(row_ind, col_ind, strict=True)}

    def _slot_for(self, agent_id: int) -> Pose2D | None:
        if agent_id not in self._members:
            return None
        slot_index = self._slot_by_member.get(agent_id, self._members.index(agent_id))
        n = len(self._members)
        return self._slot_pose(slot_index, n, self.anchor.pose(), self._radius())

    def tick(self, dt: float) -> dict[int, Pose2D]:
        if not self._members:
            return {}
        out: dict[int, Pose2D] = {}
        for aid in self._members:
            slot = self._slot_for(aid)
            if slot is not None:
                out[aid] = slot
        return out

    def arrived(self, agent_id: int) -> bool:
        slot = self._slot_for(agent_id)
        if slot is None:
            return True
        agent = self.agent_lookup(agent_id)
        if agent is None:
            return True
        pose = agent.state.pose
        return pose_distance(pose, slot) < DISTANCE_TOLERANCE
