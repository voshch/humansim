import py_trees
from pydantic import BaseModel, Field

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.utils.types import (
    AgentKind,
    HighLevelCommand,
    InteractionOutcome,
    InteractionType,
    Pose2D,
)


class GroupTalkNodeSchema(BaseModel):
    """
    Orchestrates a group conversation. Positioning (circle formation) is handled by
    the InteractionManager's FFormation; this node only drives the policy: leader
    advertises, followers accept, and all monitor outcome.
    """

    agent: BaseAgent = Field(description="The local agent instance controlled by this specific behavior tree node.")
    participant_agents: list[BaseAgent] = Field(description="The full roster of agents participating in the talk. The first agent in this list is designated as the 'initiator' or 'leader' for synchronization purposes.")
    group_center: Pose2D = Field(description="Legacy hint; ignored when FFormation is active. Kept for backward-compat with scenarios that still supply it.")
    duration: float = Field(description="The total time in seconds the group should remain in the ACTIVE conversation state.")


class GroupTalkNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: GroupTalkNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.agents = [a for a in config.participant_agents if a.state.kind == AgentKind.HUMAN]
        self.duration = config.duration
        self.emitted_cmd = False
        self.is_leading_agent = False

    def initialise(self) -> None:
        self.emitted_cmd = False
        if not self.agents:
            self.is_leading_agent = False
            return
        self.is_leading_agent = self.agent.state.agent_id == self.agents[0].state.agent_id

    def update(self) -> py_trees.common.Status:
        if not self.agents:
            return py_trees.common.Status.FAILURE

        if not self.emitted_cmd:
            if self.is_leading_agent:
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.agent.state.agent_id,
                    type=CommandType.ADVERTISE,
                    interaction_type=InteractionType.GROUP_CONVERSATION,
                    interaction_duration=self.duration,
                )
            else:
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.agent.state.agent_id,
                    type=CommandType.ADVERTISE,
                    interaction_type=InteractionType.GROUP_CONVERSATION,
                    target_agent=self.agents[0].state.agent_id,
                )
            self.emitted_cmd = True
            return py_trees.common.Status.RUNNING

        outcome = self.agent.movement.last_outcome
        if outcome in (InteractionOutcome.FORMING, InteractionOutcome.ACTIVE, None):
            return py_trees.common.Status.RUNNING
        if outcome == InteractionOutcome.COMPLETED:
            return py_trees.common.Status.SUCCESS
        if outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE
        raise ValueError(f"Invalid InteractionOutcome. Received {outcome}")
