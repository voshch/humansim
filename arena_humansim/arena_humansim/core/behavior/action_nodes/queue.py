import math

import py_trees
from pydantic import BaseModel, Field

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.behavior.nodes import _nav_command
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    HighLevelCommand,
    InteractionOutcome,
    Pose2D,
)


class QueueNodeSchema(BaseModel):
    """
    Commands an agent to join a queue and progress to the front.
    """

    agent: BaseAgent = Field(description="The agent joining the queue")
    service_duration: float = Field(description="Time spent at the very front of the queue (e.g., at the counter)")
    front_pose: Pose2D = Field(description="The head of the queue. Yaw determines the direction the line forms.")
    post_queue_pose: Pose2D = Field(description="Where the agent goes after finishing the service.")
    step_distance: float = Field(default=1.0, description="Distance between people in the queue")
    chaos_mode: bool = Field(
        default=False,
        description="If True, agent ignores strict ordering and rushes for the gap",
    )


class QueueNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, config: QueueNodeSchema):
        super().__init__(name)
        self.agent = config.agent
        self.service_duration = config.service_duration
        self.front_pose = config.front_pose
        self.post_queue_pose = config.post_queue_pose
        self.step_distance = config.step_distance
        self.chaos_mode = config.chaos_mode

        # Internal State
        self.current_index: int = -1
        self.assigned_pose: Pose2D | None = None
        self.service_started = False

    def calculate_queue_pose(self, index: int) -> Pose2D:
        """
        Calculates world position for spot `index` in the line
        Line extends backwards from the `front_pose yaw
        E.g: If yaw is 0 (facing East), the line grows towards the West
        """
        backward_angle = self.front_pose.theta + math.pi
        offset_x = index * self.step_distance * math.cos(backward_angle)
        offset_y = index * self.step_distance * math.sin(backward_angle)

        return Pose2D(
            x=self.front_pose.x + offset_x,
            y=self.front_pose.y + offset_y,
            theta=self.front_pose.theta,
        )

    def is_spot_occupied(self, pose: Pose2D) -> bool:
        """Helper to check if any observed agent is at a specific queue pose."""
        for oa in self.agent.belief.observed_agents:
            if oa.kind != AgentKind.HUMAN:
                continue
            if self.is_at_pose(oa, target=pose):
                return True
        return False

    def find_available_index(self, max_search: int = 20) -> int:
        """Recursively/Iteratively finds the first empty index in the queue."""
        if self.chaos_mode:
            return 0

        for i in range(max_search):
            pose_to_check = self.calculate_queue_pose(i)
            if not self.is_spot_occupied(pose_to_check):
                return i
        return max_search

    def initialise(self) -> None:
        """Initializes by scanning the queue line for the first empty spot."""
        self.current_index = self.find_available_index()
        self.assigned_pose = self.calculate_queue_pose(self.current_index)
        self.service_started = False

        if self.chaos_mode:
            self.agent.state.desired_velocity *= 1.5

    def is_at_pose(self, state: AgentState, target: Pose2D) -> bool:
        dx = state.pose.x - target.x
        dy = state.pose.y - target.y

        return math.hypot(dx, dy) < DISTANCE_TOLERANCE

    def someone_too_close(self) -> bool:
        """Check if someone is too close in the direction of the queue."""
        state = self.agent.state
        for oa in self.agent.belief.observed_agents:
            if oa.kind != AgentKind.HUMAN:
                continue
            dist = math.hypot(oa.pose.x - state.pose.x, oa.pose.y - state.pose.y)
            if dist < DISTANCE_TOLERANCE:
                return True
        return False

    def update(self) -> py_trees.common.Status:
        # 1. Check if we are at the front and need to perform Idle
        if self.current_index == 0 and self.is_at_pose(self.agent.state, self.front_pose):
            if not self.service_started:
                # Emit IDLE command once to represent waiting to be served at the counter
                self.agent.movement.command = HighLevelCommand(
                    agent_id=self.agent.state.agent_id,
                    type=CommandType.IDLE,
                    interaction_duration=self.service_duration,
                )
                self.service_started = True
                return py_trees.common.Status.RUNNING

            # Monitor service completion
            outcome = self.agent.movement.last_outcome
            if outcome == InteractionOutcome.COMPLETED:
                # Once served, move to the final destination
                self.agent.movement.command = _nav_command(self.agent, self.post_queue_pose)
                if self.is_at_pose(self.agent.state, self.post_queue_pose):
                    return py_trees.common.Status.SUCCESS
                return py_trees.common.Status.RUNNING

            return py_trees.common.Status.RUNNING

        # 2. Progression Logic:
        if self.current_index > 0:
            pose_in_front = self.calculate_queue_pose(self.current_index - 1)

            # If the spot in front of us is empty, move up
            if not self.is_spot_occupied(pose_in_front):
                self.current_index -= 1
                self.assigned_pose = pose_in_front

        # Conflict resolution & spacing:
        # This prevents "stacking" if the target index is empty but someone is standing nearby
        if self.someone_too_close() and not self.chaos_mode:
            # Command agent to stay where he's at
            self.agent.movement.command = _nav_command(self.agent, self.agent.state.pose)
        else:
            # Move to the currently assigned spot in line
            assert self.assigned_pose is not None
            self.agent.movement.command = _nav_command(self.agent, self.assigned_pose)

        return py_trees.common.Status.RUNNING
