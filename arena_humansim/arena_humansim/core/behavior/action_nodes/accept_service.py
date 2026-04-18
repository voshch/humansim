import py_trees
from pydantic import BaseModel, ConfigDict, Field

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.utils.types import HighLevelCommand, InteractionOutcome, InteractionType


class AcceptServiceNodeSchema(BaseModel):
    """Wait to be matched into a SERVICE interaction advertising `service_tag`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: BaseAgent = Field(description="The agent that will accept the service")
    service_tag: str = Field(description="The service tag to match (e.g. 'water')")
    from_agent: BaseAgent | None = Field(default=None, description="If set, only match with this specific agent's ad")


class AcceptServiceNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: AcceptServiceNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.service_tag = config.service_tag
        self.from_agent = config.from_agent

    def initialise(self) -> None:
        self.agent.movement.last_outcome = None
        self.agent.movement.command = HighLevelCommand(
            agent_id=self.agent.state.agent_id,
            type=CommandType.ADVERTISE,
            interaction_type=int(InteractionType.SERVICE),
            service_tag=self.service_tag,
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
