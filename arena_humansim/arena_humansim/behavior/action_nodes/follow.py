import py_trees
from pydantic import BaseModel, Field

from arena_humansim.agents import BaseAgent
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionOutcome,
    InteractionType,
)


class FollowNodeSchema(BaseModel):
    """
    Command agent to follow a target agent in a duration (second)
    """

    agent: BaseAgent = Field(description="The follower")
    target_agent: BaseAgent = Field(description="The agent will be followed")
    duration: float = Field(description="The duration of time of the follow")


class FollowNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: FollowNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.target_agent = config.target_agent
        self.duration = config.duration

    def initialise(self) -> None:
        """
        Emit HighLevelCommand only once, when node first starts to tick for InteractionManager to manage elapsed time via `InteractionContract.elapsed`
        """
        self.agent.movement.command = HighLevelCommand(
            agent_id=self.agent.state.agent_id,
            type=CommandType.ADVERTISE,
            target_agent=self.target_agent.state.agent_id,
            interaction_type=InteractionType.FOLLOW,
            interaction_duration=self.duration,
        )

    def update(self) -> py_trees.common.Status:
        outcome = self.agent.movement.last_outcome
        if outcome in [InteractionOutcome.ACTIVE, InteractionOutcome.FORMING]:
            return py_trees.common.Status.RUNNING

        elif outcome == InteractionOutcome.COMPLETED:
            return py_trees.common.Status.SUCCESS

        elif outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE

        else:
            raise ValueError(f"Invalid InteractionOutcome for node {self.__class__.__name__}. Received {outcome}")
