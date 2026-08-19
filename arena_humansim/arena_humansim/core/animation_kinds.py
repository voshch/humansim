"""Unified AnimationType enum and helper logic to translate agent interaction and physical states into animation_state values."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING
from arena_humansim.core.interaction_kinds import InteractionType

if TYPE_CHECKING:
    from arena_humansim.core.agents.base import BaseAgent


class AnimationType(enum.IntEnum):
    # Legacy / Locomotion / Reaction states (clash-free, matching Pedestrian.msg and task_generator mapping)
    IDLE = 0
    WALKING = 1
    RUNNING = 2
    PANIC = 3
    SURPRISED = 4
    CURIOUS = 5
    THREATENING = 6

    # New / Interaction & Special states (Non-clashing, >= 7)
    HUG = 7
    JUMP = 8
    POINT = 9
    SHAKE_HAND = 10
    SIT = 11
    TALK = 12
    WAVE = 13
    WAVE_HIGH = 14
    FALL = 15


# Map InteractionType directly to the corresponding AnimationType
INTERACTION_TO_ANIMATION = {
    InteractionType.TALK_TO: AnimationType.TALK,
    InteractionType.GROUP_CONVERSATION: AnimationType.TALK,
    InteractionType.SIT_ON: AnimationType.SIT,
    InteractionType.LIE_ON: AnimationType.SIT,
    InteractionType.WAVE_AT: AnimationType.WAVE,
    InteractionType.HUG: AnimationType.HUG,
    InteractionType.JUMP: AnimationType.JUMP,
    InteractionType.POINT: AnimationType.POINT,
    InteractionType.SHAKE_HAND: AnimationType.SHAKE_HAND,
    InteractionType.FALL: AnimationType.FALL,
}


def get_animation_for_agent(
    agent: BaseAgent,
    active_interaction_type: int | None,
    speed: float,
    has_goal: bool = False,
) -> int:
    """
    Resolves the animation state of an agent based on its active interaction,
    physical/movement characteristics (e.g. speed), or non-interaction special states (like falling).
    """
    # 1. Check for non-interaction-based special states (like falling)
    # This can be set via attributes on the agent, its state, or behavior/movement commands
    if getattr(agent, "falling", False) or getattr(agent.state, "falling", False):
        return int(AnimationType.FALL)

    # 2. Check active interaction-based animation
    if active_interaction_type is not None:
        try:
            itype = InteractionType(active_interaction_type)
            if itype in INTERACTION_TO_ANIMATION:
                return int(INTERACTION_TO_ANIMATION[itype])
        except ValueError:
            pass

    # 3. Fallback to standard speed-based locomotion states
    # If speed is virtually 0 but the agent has an active goal/intent to move,
    # use desired_velocity as the fallback speed to prevent a 1-frame startup idle.
    effective_speed = speed
    if speed <= 0.05 and has_goal:
        effective_speed = agent.state.desired_velocity

    if effective_speed > 1.5:
        return int(AnimationType.RUNNING)
    elif effective_speed > 0.05:
        return int(AnimationType.WALKING)
    else:
        return int(AnimationType.IDLE)
