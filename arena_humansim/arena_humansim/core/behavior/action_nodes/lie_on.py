import math

import py_trees
from pydantic import BaseModel, Field
from rclpy.logging import get_logger

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.behavior.nodes import _nav_command
from arena_humansim.utils.const import DISTANCE_TOLERANCE
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionOutcome,
    InteractionType,
    Pose2D,
)

_bt_logger = get_logger("behavior_tree")


class LieOnNodeSchema(BaseModel):
    """
    Command agent to navigate to an object and lie on it for a specific duration of time
    """

    agent: BaseAgent = Field(description="Agent that will lie on the object")
    object_name: str = Field(description="Object to be lied on")
    duration: float = Field(description="The duration of time the agent will lie on the object")


class LieOnNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: LieOnNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.object_name = config.object_name
        self.duration = config.duration
        self.object_position: Pose2D | None = None
        self.emitted_lie_on_cmd = False
        self.target_pose = Pose2D()

    def initialise(self) -> None:
        """
        Emit HighLevelCommand only once, when node first starts to tick for InteractionManager to manage elapsed time via `InteractionContract.elapsed`
        """
        # TODO: Search for object

    def at_target(self) -> bool:
        assert self.object_position is not None
        dx = self.agent.state.pose.x - self.object_position.x
        dy = self.agent.state.pose.y - self.object_position.y

        return math.hypot(dx, dy) < DISTANCE_TOLERANCE

    def update(self) -> py_trees.common.Status:
        # Could not find object to lie on
        if self.object_position is None:
            _bt_logger.info(f"Could not find object {self.object_name} for agent {self.agent.state.agent_id} to lie on it.")
            return py_trees.common.Status.FAILURE

        # Object found, navigating toward it
        elif not self.at_target():
            _bt_logger.info(f"Agent {self.agent.state.agent_id} is navigating to {self.object_name} to lie on it.")
            self.agent.movement.command = _nav_command(agent=self.agent, target_pose=self.target_pose)
            return py_trees.common.Status.RUNNING

        # Navigated to object, emit lie on command
        else:
            if not self.emitted_lie_on_cmd:
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.agent.state.agent_id,
                    interaction_type=InteractionType.LIE_ON,
                    interaction_duration=self.duration,
                )

            else:
                outcome = self.agent.movement.last_outcome

                if outcome == InteractionOutcome.ACTIVE:
                    return py_trees.common.Status.RUNNING

                elif outcome == InteractionOutcome.COMPLETED:
                    return py_trees.common.Status.SUCCESS

                elif outcome == InteractionOutcome.INTERRUPTED:
                    return py_trees.common.Status.FAILURE

                else:
                    raise ValueError(f"Invalid of InteractionOutcome for {self.__class__.__name__} of agent {self.agent.state.agent_id}. Received {outcome}")
