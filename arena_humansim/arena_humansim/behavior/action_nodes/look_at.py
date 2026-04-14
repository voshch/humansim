import numpy as np
import py_trees
from pydantic import BaseModel, Field

from arena_humansim.agents import BaseAgent
from arena_humansim.behavior.nodes import _nav_command
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.utils.const import ANGLE_TOLERANCE
from arena_humansim.utils.types import HighLevelCommand, InteractionOutcome, Pose2D


class LookAtNodeSchema(BaseModel):
    """
    Command agent to look at a target position in a duration (second)
    """

    agent: BaseAgent = Field(description="The agent will look")
    target_look_pose: Pose2D = Field(description="The target position for the agent to look at")
    duration: float = Field(description="The duration of time of the look")


class LookAtNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: LookAtNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.target_look_pose = config.target_look_pose
        self.duration = config.duration
        self.target_heading_angle = 0.0
        self.emitted_idle_command = False
        self.target_pose = Pose2D()

    def initialise(self) -> None:
        self.emitted_idle_command = False

        # Calculate direction vector
        dx = self.target_look_pose.x - self.agent.state.pose.x
        dy = self.target_look_pose.y - self.agent.state.pose.y

        self.target_heading_angle = np.arctan2(dy, dx).item()

        # Command agent to rotate in place (keep same X, Y)
        self.target_pose = Pose2D(
            x=self.agent.state.pose.x,
            y=self.agent.state.pose.y,
            theta=self.target_heading_angle,
        )

    def update(self) -> py_trees.common.Status:
        # 1. Check Rotation Progress
        angle_diff = abs(self.agent.state.pose.theta - self.target_heading_angle)
        # Normalize angle diff to [0, 180]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        angle_diff = angle_diff / np.pi * 180

        if abs(angle_diff) > ANGLE_TOLERANCE:
            self.agent.movement.command = _nav_command(agent=self.agent, target_pose=self.target_pose)
            return py_trees.common.Status.RUNNING

        # 2. Rotation complete, emit IDLE command
        if not self.emitted_idle_command:
            self.agent.movement.command = HighLevelCommand(
                agent_id=self.agent.state.agent_id,
                type=CommandType.IDLE,
                interaction_duration=self.duration,
            )
            self.emitted_idle_command = True
            return py_trees.common.Status.RUNNING

        # 3. Monitor outcome from InteractionManager
        outcome = self.agent.movement.last_outcome

        if outcome in [InteractionOutcome.ACTIVE, InteractionOutcome.FORMING]:
            return py_trees.common.Status.RUNNING

        elif outcome == InteractionOutcome.COMPLETED:
            return py_trees.common.Status.SUCCESS

        elif outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE

        else:
            raise ValueError(f"Invalid of InteractionOutcome for {self.__class__.__name__} of agent {self.agent.state.agent_id}. Received {outcome}")
