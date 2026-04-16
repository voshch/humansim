import math

import py_trees
from pydantic import BaseModel, Field

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.behavior.nodes import _nav_command
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D


class GoToNodeSchema(BaseModel):
    """
    Navigate the agent to a specific coordinate.
    """

    agent: BaseAgent = Field(description="Agent that is commanded to navigate to the target position")
    target_pose: Pose2D = Field(description="Position of the target point")


class GoToNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: GoToNodeSchema):
        super().__init__(name)
        self.target_pose = config.target_pose
        self.agent = config.agent

    def at_target(self) -> bool:
        dx = self.agent.state.pose.x - self.target_pose.x
        dy = self.agent.state.pose.y - self.target_pose.y
        return math.hypot(dx, dy) < DISTANCE_TOLERANCE

    def update(self) -> py_trees.common.Status:
        if self.at_target():
            return py_trees.common.Status.SUCCESS

        else:
            self.agent.movement.command = _nav_command(agent=self.agent, target_pose=self.target_pose)

            return py_trees.common.Status.RUNNING
