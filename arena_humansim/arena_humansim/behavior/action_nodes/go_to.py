import math

from pydantic import BaseModel, Field

import py_trees

from arena_humansim.agents import BaseAgent
from arena_humansim.utils.types import Pose2D
from arena_humansim.behavior.nodes import _nav_command
from arena_humansim.utils import DISTANCE_TOLERANCE


class GoToNodeSchema(BaseModel):
    """
    Navigate the agent to a specific coordinate.
    """

    agent: BaseAgent = Field(
        description="Agent that is commanded to navigate to the target position"
    )
    target_pose: Pose2D = Field(description="Position of the target point")


class GoToNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: GoToNodeSchema):
        super().__init__(name)
        self.target_pose = config.target_pose
        self.agent = config.agent

    def at_target(self):
        dx = self.agent.state.pose.x - self.target_pose.x
        dy = self.agent.state.pose.y - self.target_pose.y
        return math.hypot(dx, dy) < DISTANCE_TOLERANCE

    def update(self) -> py_trees.common.Status:
        if self.at_target():
            return py_trees.common.Status.SUCCESS

        else:
            self.agent.movement.command = _nav_command(
                agent=self.agent, target_pose=self.target_pose
            )

            return py_trees.common.Status.RUNNING
