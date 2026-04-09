import math

from typing import List

from pydantic import BaseModel, Field

import py_trees

from arena_humansim.agents import BaseAgent
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionOutcome,
    InteractionType,
    Pose2D,
)
from arena_humansim.behavior.nodes import _nav_command
from arena_humansim.utils import DISTANCE_TOLERANCE


class GroupTalkNodeSchema(BaseModel):
    """
    Orchestrates a social gathering by first navigating the agent to a focal point and then transitioning into a synchronized group conversation state.
    """

    agent: BaseAgent = Field(
        description="The local agent instance controlled by this specific behavior tree node."
    )
    participant_agents: List[BaseAgent] = Field(
        description="The full roster of agents participating in the talk. "
        "The first agent in this list is designated as the 'initiator' or 'leader' for synchronization purposes."
    )
    group_center: Pose2D = Field(
        description="The spatial coordinates (target) where the agents gather to form the conversation circle."
    )
    duration: float = Field(
        description="The total time in seconds the group should remain in the ACTIVE conversation state."
    )


class GroupTalkNode(py_trees.behaviour.Behaviour):
    """
    A state-dependent behavior node that manages the 'Gather-then-Talk' workflow:
    1. RUNNING: Navigates 'agent' to 'group_center' until within DISTANCE_TOLERANCE.
    2. RUNNING: Waits until all 'participant_agents' have arrived at the center.
    3. RUNNING: Triggers a GROUP_CONVERSATION HighLevelCommand once.
    4. SUCCESS: Returns when the InteractionOutcome is COMPLETED.
    5. FAILURE: Returns if the interaction is INTERRUPTED.
    """

    def __init__(self, name: str, config: GroupTalkNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.agents = config.participant_agents
        self.group_center = config.group_center
        self.duration = config.duration
        self.emited_cmd = False
        self.is_leading_agent = False

    def initialise(self) -> None:
        if self.agent.state.agent_id == self.agents[0].state.agent_id:
            self.is_leading_agent = True

    def at_target(self, agent: BaseAgent):
        dx = agent.state.pose.x - self.group_center.x
        dy = agent.state.pose.y - self.group_center.y
        return math.hypot(dx, dy) < DISTANCE_TOLERANCE

    def update(self) -> py_trees.common.Status:
        # 1. Check if all agents are gathered around the group center
        all_gathered = all(self.at_target(a) for a in self.agents)

        if not all_gathered:
            # Only command the local agent to move if it isn't there yet
            self.agent.movement.command = _nav_command(self.agent, self.group_center)

            return py_trees.common.Status.RUNNING

        # 2. All gathered, start group conversation
        # Emit HighLevelCommand only once, when agents gathered for InteractionManager to manage elapsed time via `InteractionContract.elapsed`
        elif all_gathered and not self.emited_cmd:
            # TODO: Test for race condition: Leading agent emits advertisement after participant agents accept
            if self.is_leading_agent:
                # Leading agent advertises to create the interaction
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.agent.state.agent_id,
                    type=CommandType.ADVERTISE,
                    interaction_type=InteractionType.GROUP_CONVERSATION,
                    interaction_duration=self.duration,
                )

            else:
                # Participant joins interaction by searching
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.agent.state.agent_id,
                    type=CommandType.ACCEPT,
                    interaction_type=InteractionType.GROUP_CONVERSATION,
                    interaction_target=-1,
                    target_agent=self.agents[0].state.agent_id,  # Leading agent ID
                )

            self.emited_cmd = True

            return py_trees.common.Status.RUNNING

        # 3. Monitor outcome
        # if all_gathered and self.emited_cmd:
        else:
            outcome = self.agent.movement.last_outcome
            if outcome in [InteractionOutcome.FORMING, InteractionOutcome.ACTIVE]:
                return py_trees.common.Status.RUNNING

            if outcome == InteractionOutcome.COMPLETED:
                return py_trees.common.Status.SUCCESS

            if outcome == InteractionOutcome.INTERRUPTED:
                return py_trees.common.Status.FAILURE

            else:
                raise ValueError(
                    f"Invalid InteractionOutcome. Received {self.agent.movement.last_outcome}"
                )
