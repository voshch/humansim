from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import attrs

from arena_humansim.utils.types import Pose2D

if TYPE_CHECKING:
    from arena_humansim.core.world_knowledge import WorldKnowledge


PoseLookup = Callable[[int], Pose2D | None]
MembersFn = Callable[[], list[int]]


class Anchor(Protocol):
    def pose(self) -> Pose2D: ...


@attrs.define
class PoseAnchor:
    fixed: Pose2D

    def pose(self) -> Pose2D:
        return self.fixed


@attrs.define
class ObjectAnchor:
    world_knowledge: WorldKnowledge
    object_id: str

    def pose(self) -> Pose2D:
        p = self.world_knowledge.object_pose(self.object_id)
        if p is None:
            return Pose2D()
        return p


@attrs.define
class AgentAnchor:
    pose_lookup: PoseLookup
    agent_id: int

    def pose(self) -> Pose2D:
        p = self.pose_lookup(self.agent_id)
        if p is None:
            return Pose2D()
        return p


@attrs.define
class CentroidAnchor:
    pose_lookup: PoseLookup
    members_fn: MembersFn
    fallback_yaw: float = 0.0

    def pose(self) -> Pose2D:
        ids = self.members_fn()
        if not ids:
            return Pose2D(theta=self.fallback_yaw)
        xs: list[float] = []
        ys: list[float] = []
        for aid in ids:
            p = self.pose_lookup(aid)
            if p is None:
                continue
            xs.append(p.x)
            ys.append(p.y)
        if not xs:
            return Pose2D(theta=self.fallback_yaw)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        # Yaw is undefined for centroid; keep fallback. Formations that need
        # orientation (inward facing) compute per-member yaw from member pose -> centroid.
        return Pose2D(x=cx, y=cy, theta=self.fallback_yaw)


def angle_toward(from_p: Pose2D, to_p: Pose2D) -> float:
    return math.atan2(to_p.y - from_p.y, to_p.x - from_p.x)
