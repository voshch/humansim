import math

import numpy as np
from rclpy.logging import get_logger

from arena_humansim.core.agents import BaseAgent, ParamDist
from arena_humansim.core.interaction_manager import CommandType, interaction_radius_for
from arena_humansim.core.world_knowledge import WorldObject
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import HighLevelCommand, InteractionType, Pose2D

_bt_logger = get_logger("behavior_tree")


def _sample_param_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    value = rng.normal(dist.mean, dist.std) if dist.std > 0 else dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def _nav_command(agent: BaseAgent, target_pose: Pose2D) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.NAVIGATE,
        target_pose=target_pose,
        desired_velocity=agent.state.desired_velocity,
    )


def _interaction_command(
    agent: BaseAgent,
    interaction_name: str,
    target_agent: int = -1,
    interaction_target: int = -1,
    duration: float | None = None,
    object_id: str | None = None,
    target_pose: Pose2D | None = None,
    service_tag: str | None = None,
) -> HighLevelCommand:
    cmd = HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.ADVERTISE,
        desired_velocity=agent.state.desired_velocity,
        interaction_type=InteractionType[interaction_name].value,
        target_agent=target_agent,
        interaction_target=interaction_target,
        interaction_duration=duration,
        object_id=object_id,
        service_tag=service_tag,
    )
    if target_pose is not None:
        cmd.target_pose = target_pose
    return cmd


def _at_target(agent: BaseAgent, target_pose: Pose2D, tolerance: float = DISTANCE_TOLERANCE) -> bool:
    dx = agent.state.pose.x - target_pose.x
    dy = agent.state.pose.y - target_pose.y
    return math.hypot(dx, dy) < tolerance


def _resolve_interaction_radius(obj: WorldObject, step_override: float | None, interaction_name: str | None) -> float:
    if step_override is not None:
        return step_override
    obj_radius = getattr(obj, "interaction_radius", None)
    if obj_radius is not None:
        return float(obj_radius)
    if interaction_name is not None:
        return interaction_radius_for(InteractionType[interaction_name])
    return DISTANCE_TOLERANCE
