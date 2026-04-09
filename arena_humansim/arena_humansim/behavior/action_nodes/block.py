import math
from os import stat_result

import py_trees

from pydantic import BaseModel, Field
from arena_humansim import agents
from arena_humansim.agents import BaseAgent
from arena_humansim.behavior.nodes import _nav_command
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.utils.const import DISTANCE_TOLERANCE
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionOutcome,
    InteractionType,
    Pose2D,
)


class BlockNodeSchema(BaseModel):
    """
    Command agent to block a target agent in a duration (second)
    """

    agent: BaseAgent = Field(description="The blocker")
    target_agent: BaseAgent = Field(description="The agent will be blocked")
    duration: float = Field(description="The duration of time of the block")


class BlockNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: BlockNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.target_agent = config.target_agent
        self.duration = config.duration
        self.reached_target = False  # Flag for reaching the target for the first time
        self.prev_vel = (
            self.agent.state.desired_velocity
        )  # Store the velocity of agent before blocking target agent

    def initialise(self) -> None:
        self.reached_target = False

        # Store the velocity of agent before blocking target agent
        self.prev_vel = self.agent.state.velocity

        # Make agent to move faster than target agent
        self.agent.state.desired_velocity = (
            self.target_agent.state.desired_velocity * 1.5
        )

    def predict_target_future_pos(self) -> tuple[float, float, float]:
        """
        Get the future position of the target agent base on its current state
        """
        state = self.target_agent.state
        x = state.pose.x + state.velocity[0]
        y = state.pose.y + state.velocity[0]
        theta = state.pose.theta

        return (x, y, theta)

    def at_target(self):
        target = self.predict_target_future_pos()
        dx = self.agent.state.pose.x - target[0]
        dy = self.agent.state.pose.y - target[1]

        return math.hypot(dx, dy) < DISTANCE_TOLERANCE

    def update(self) -> py_trees.common.Status:
        if self.at_target():
            # Reached the target for the first time: Emit high level command, start tracking block duration
            if not self.reached_target:
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.target_agent.state.agent_id,
                    type=CommandType.ADVERTISE,
                    target_agent=self.target_agent.state.agent_id,
                    interaction_type=InteractionType.BLOCK,
                    interaction_duration=self.duration,
                )

        # Navigate towards target agent's heading position
        else:
            target_pose = self.predict_target_future_pos()
            target_pose = Pose2D(
                x=target_pose[0],
                y=target_pose[1],
                theta=target_pose[2],
            )
            self.agent.movement.command = _nav_command(
                agent=self.agent, target_pose=target_pose
            )

        # Monitor duration via InteractionOutcome
        outcome = self.agent.movement.last_outcome
        if outcome in [InteractionOutcome.ACTIVE, InteractionOutcome.FORMING]:
            return py_trees.common.Status.RUNNING

        elif outcome == InteractionOutcome.COMPLETED:
            return py_trees.common.Status.SUCCESS

        elif outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE

        else:
            raise ValueError(
                f"Invalid InteractionOutcome for node {self.__class__.__name__}. Received {outcome}"
            )
