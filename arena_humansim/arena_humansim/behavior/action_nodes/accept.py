import py_trees
from pydantic import BaseModel, ConfigDict, Field

from arena_humansim.agents import BaseAgent
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.utils.types import HighLevelCommand, InteractionOutcome, InteractionType


class AcceptNodeSchema(BaseModel):
    """
    Wait for and accept an incoming interaction of a given type. Parks in RUNNING
    until a matching advertisement exists; optionally restricts to ads emitted by
    a specific agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: BaseAgent = Field(description="The agent that will accept the incoming interaction")
    interaction_type: InteractionType = Field(description="The type of interaction to accept")
    from_agent: BaseAgent | None = Field(default=None, description="If set, only accept ads from this specific agent; otherwise accept from anyone")


class AcceptNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: AcceptNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.interaction_type = config.interaction_type
        self.from_agent = config.from_agent

    def initialise(self) -> None:
        self.agent.movement.last_outcome = None
        self.agent.movement.command = HighLevelCommand(
            agent_id=self.agent.state.agent_id,
            type=CommandType.ACCEPT,
            interaction_type=int(self.interaction_type),
            interaction_target=-1,
            target_agent=self.from_agent.state.agent_id if self.from_agent is not None else -1,
        )

    def update(self) -> py_trees.common.Status:
        outcome = self.agent.movement.last_outcome
        if outcome is None or outcome == InteractionOutcome.FORMING:
            return py_trees.common.Status.RUNNING

        if outcome == InteractionOutcome.ACTIVE:
            return py_trees.common.Status.RUNNING

        if outcome == InteractionOutcome.COMPLETED:
            return py_trees.common.Status.SUCCESS

        if outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE

        raise ValueError(f"Invalid InteractionOutcome for node {self.__class__.__name__}. Received {outcome}")
