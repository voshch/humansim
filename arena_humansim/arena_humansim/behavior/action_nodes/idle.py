import py_trees
from pydantic import BaseModel, Field

from arena_humansim.agents import BaseAgent
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.utils.types import HighLevelCommand, InteractionOutcome


class IdleNodeSchema(BaseModel):
    """
    Command agent to idle for a specific amount of time.
    """

    agent: BaseAgent = Field(description="Agent that is commanded to idle")
    duration: float = Field(description="Duration to idle")


class IdleNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: IdleNodeSchema):
        super().__init__(name)
        self.duration = config.duration
        self.agent = config.agent

    def initialise(self) -> None:
        """
        Emit HighLevelCommand only once, when node first starts to tick for InteractionManager to manage elapsed time via `InteractionContract.elapsed`
        """
        self.agent.movement.command = HighLevelCommand(
            agent_id=self.agent.state.agent_id,
            type=CommandType.IDLE,
            interaction_duration=self.duration,
        )

    def update(self) -> py_trees.common.Status:
        outcome = self.agent.movement.last_outcome

        if outcome == InteractionOutcome.ACTIVE:
            return py_trees.common.Status.RUNNING

        elif outcome == InteractionOutcome.COMPLETED:
            return py_trees.common.Status.SUCCESS

        elif outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE

        else:
            raise ValueError(f"Invalid of InteractionOutcome for node {self.__class__.__name__} of agent {self.agent.state.agent_id}. Received {outcome}")
