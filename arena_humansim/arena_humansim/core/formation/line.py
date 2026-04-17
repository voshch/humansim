from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs

from arena_humansim.utils.types import Pose2D

from . import AgentLookup, Formation
from .anchor import Anchor

if TYPE_CHECKING:
    pass


@attrs.define
class _Slot:
    agent_id: int
    target: Pose2D
    prev_target: Pose2D
    reaction_left: float = 0.0
    pending_target: Pose2D | None = None  # set on leave; applied after reaction delay
    target_changed_tick: bool = False


# Coefficient applied to agent.reaction_time for shuffle wave propagation.
# Queue shuffles are motor-adjacent; ~0.7x of the base reaction trait.
SITE_COEF_SHUFFLE = 0.7

_POSE_EPS = 1e-4


def _pose_differs(a: Pose2D, b: Pose2D) -> bool:
    return abs(a.x - b.x) > _POSE_EPS or abs(a.y - b.y) > _POSE_EPS or abs(a.theta - b.theta) > _POSE_EPS


class LineFormation(Formation):
    """Ordered line with target-propagation shuffle.

    Invariant: slot k's target is, after a reaction delay, a delayed echo of
    slot k-1's target. Slot 0 is anchored to the formation's anchor pose.
    The line extends backward along (anchor.theta + pi).
    """

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        base_step: float = 1.0,
        formation_scale: float = 1.0,
    ) -> None:
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.base_step = base_step
        self.formation_scale = formation_scale
        self._slots: list[_Slot] = []

    def _spacing_for(self, agent_id: int) -> float:
        agent = self.agent_lookup(agent_id)
        if agent is None:
            floor = self.base_step
        else:
            floor = max(self.base_step, agent.params.personal_space_min)
        return floor * self.formation_scale

    def _backward_offset(self, step: float, yaw: float) -> tuple[float, float]:
        backward = yaw + math.pi
        return step * math.cos(backward), step * math.sin(backward)

    def on_join(self, agent_id: int) -> None:
        if any(s.agent_id == agent_id for s in self._slots):
            return
        anchor_pose = self.anchor.pose()
        if not self._slots:
            target = Pose2D(x=anchor_pose.x, y=anchor_pose.y, theta=anchor_pose.theta)
        else:
            tail = self._slots[-1].target
            dx, dy = self._backward_offset(self._spacing_for(agent_id), tail.theta)
            target = Pose2D(x=tail.x + dx, y=tail.y + dy, theta=tail.theta)
        self._slots.append(_Slot(agent_id=agent_id, target=target, prev_target=target))

    def on_leave(self, agent_id: int) -> None:
        idx = next((i for i, s in enumerate(self._slots) if s.agent_id == agent_id), -1)
        if idx < 0:
            return
        removed = self._slots.pop(idx)
        if idx < len(self._slots):
            successor = self._slots[idx]
            agent = self.agent_lookup(successor.agent_id)
            reaction = agent.params.reaction_time if agent is not None else 0.4
            successor.reaction_left = SITE_COEF_SHUFFLE * reaction
            successor.pending_target = removed.target

    def tick(self, dt: float) -> dict[int, Pose2D]:
        anchor_pose = self.anchor.pose()

        for s in self._slots:
            s.target_changed_tick = False

        if self._slots:
            front = self._slots[0]
            if _pose_differs(front.target, anchor_pose):
                front.prev_target = front.target
                front.target = Pose2D(x=anchor_pose.x, y=anchor_pose.y, theta=anchor_pose.theta)
                front.target_changed_tick = True
                front.reaction_left = 0.0
                front.pending_target = None

        for i in range(1, len(self._slots)):
            pred = self._slots[i - 1]
            cur = self._slots[i]
            if pred.target_changed_tick and cur.reaction_left <= 0.0 and cur.pending_target is None:
                agent = self.agent_lookup(cur.agent_id)
                reaction = agent.params.reaction_time if agent is not None else 0.4
                cur.reaction_left = SITE_COEF_SHUFFLE * reaction
            if cur.reaction_left > 0.0:
                cur.reaction_left -= dt
                if cur.reaction_left <= 0.0:
                    cur.prev_target = cur.target
                    if cur.pending_target is not None:
                        cur.target = cur.pending_target
                        cur.pending_target = None
                    else:
                        cur.target = Pose2D(
                            x=pred.prev_target.x,
                            y=pred.prev_target.y,
                            theta=pred.prev_target.theta,
                        )
                    cur.target_changed_tick = True
                    cur.reaction_left = 0.0

        return {s.agent_id: s.target for s in self._slots}
