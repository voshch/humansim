from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import attrs
import numpy as np
import rclpy
import rclpy.publisher
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import Shape as ShapeMsg
from arena_humansim_msgs.msg import SinkConfig as SinkConfigMsg
from arena_humansim_msgs.msg import SourceConfig as SourceConfigMsg
from arena_humansim_msgs.srv import (
    AddObstacles,
    AddSink,
    AddSource,
    AddWalls,
    Feedback,
    GetProfile,
    RemoveAgents,
    RemoveObstacles,
    RemoveSink,
    RemoveSource,
    RemoveWalls,
    ResetSimulation,
    SetFlow,
    SpawnAgents,
    UpdateRobot,
)
from geometry_msgs.msg import Pose2D as Pose2DMsg
from geometry_msgs.msg import Vector3
from py_trees.trees import BehaviourTree
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock

from arena_humansim.agents import (
    BUILTIN_AGENTS,
    AgentType,
    BaseAgent,
    SampledParams,
    TickPhase,
    create_agent,
    create_agent_from_params,
    resolve_agent_type,
)
from arena_humansim.agents.loader import resolve_agent_type_name
from arena_humansim.animation import MotionAnimation
from arena_humansim.behavior.compiler import BehaviorTreeFactory
from arena_humansim.collision import CollisionResolver
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.manager.despawn_monitor import DespawnMonitor
from arena_humansim.manager.interaction_manager import CommandType, InteractionManager
from arena_humansim.manager.logger import SimulationLogger
from arena_humansim.manager.replay import ReplayManager, ReplayResult
from arena_humansim.manager.spawn_scheduler import SpawnScheduler
from arena_humansim.manager.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.perception import Perception
from arena_humansim.pool import AgentPool
from arena_humansim.utils import RNG
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.scenario import EventScript, ScenarioConfig
from arena_humansim.utils.types import (
    AgentState,
    BehaviorTreeMovement,
    BeliefState,
    HighLevelCommand,
    InteractionOutcome,
    Pose2D,
    Segments,
    Shape,
    SinkConfig,
    SourceConfig,
    SpawnRequest,
    WallAware,
    WaypointMode,
    WaypointMovement,
    WorldAgentState,
    WorldState,
)
from arena_humansim.viz import MarkerPublisher, publish_behavior, publish_global_plan, publish_infrastructure, publish_interaction, publish_local_plan, publish_module_markers, publish_perception, publish_waypoints


@attrs.define
class ObstacleData:
    name: str
    pose: Pose2D
    bb: tuple[float, float, float, float, float, float]  # x_min, x_max, y_min, y_max, z_min, z_max
    interaction_types: tuple[str, ...]
    obstacle_type: str
    wall_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


_MSG_BLOCK = 16


class _AgentStateMsgPool:
    def __init__(self) -> None:
        self._pools = ([AgentStateMsg() for _ in range(_MSG_BLOCK)], [AgentStateMsg() for _ in range(_MSG_BLOCK)])
        self._msgs = (AgentStatesMsg(), AgentStatesMsg())
        self._msgs[0].header.frame_id = "map"
        self._msgs[1].header.frame_id = "map"
        self._idx = 0

    def get(self, n: int) -> AgentStatesMsg:
        pool = self._pools[self._idx]
        while len(pool) < n:
            pool.extend(AgentStateMsg() for _ in range(_MSG_BLOCK))
        msg = self._msgs[self._idx]
        msg.agents = pool[:n]
        self._idx ^= 1
        return msg


class _BackgroundPublisher:
    def __init__(self, publisher: rclpy.publisher.Publisher) -> None:
        self._publisher = publisher
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            msg = self._queue.get()
            if msg is None:
                break
            while not self._queue.empty():
                next_msg = self._queue.get()
                if next_msg is None:
                    return
                msg = next_msg
            self._publisher.publish(msg)

    def publish(self, msg: AgentStatesMsg) -> None:
        self._queue.put(msg)

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=1.0)


def _group_by[K: Hashable](agents: Iterable[BaseAgent], key: Callable[[BaseAgent], K]) -> Iterable[tuple[K, list[BaseAgent]]]:
    groups: dict[K, list[BaseAgent]] = {}
    for agent in agents:
        k = key(agent)
        groups.setdefault(k, []).append(agent)
    return groups.items()


class AgentManager(Node):
    MODE_MASTER = "master"
    MODE_SUBSYSTEM = "subsystem"
    MODE_BENCHMARK = "benchmark"

    def __init__(self):
        super().__init__("arena_humansim")
        self._logger = self.get_logger()
        Loggable.init_logging(self)

        self.declare_parameter("seed", 0)
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("bt_tick_interval", 5)
        self.declare_parameter("perception", "default")
        self.declare_parameter("global_planner", "astar")
        self.declare_parameter("local_planner", "sfm")
        self.declare_parameter("animation", "noop")
        self.declare_parameter("collision", "wall_projection")
        self.declare_parameter("mode", self.MODE_MASTER)
        self.declare_parameter("log_dir", "")
        self.declare_parameter("replay_mode", "")
        self.declare_parameter("waypoint_threshold", 0.1)
        self.declare_parameter("min_speed_for_heading", 0.1)
        self.declare_parameter("publish_markers", 0)
        self.declare_parameter("profile_phases", False)
        self.declare_parameter("profile_interval", 0)

        seed = self.get_parameter("seed").value
        self._dt = self.get_parameter("dt").value
        self._bt_tick_interval = self.get_parameter("bt_tick_interval").value
        self._mode = self.get_parameter("mode").value
        log_dir = self.get_parameter("log_dir").value
        replay_mode = self.get_parameter("replay_mode").value
        self._waypoint_threshold = self.get_parameter("waypoint_threshold").value
        self._min_speed_for_heading = self.get_parameter("min_speed_for_heading").value
        _pm = self.get_parameter("publish_markers").value
        if isinstance(_pm, bool):
            self._publish_markers = 2 if _pm else 0
        else:
            self._publish_markers = int(_pm)
        self._profile_phases = self.get_parameter("profile_phases").value
        self._profile_interval = self.get_parameter("profile_interval").value
        self._phase_accum: dict[str, list[float]] = {}

        self._module_selections = {
            "perception": self.get_parameter("perception").value,
            "global_planner": self.get_parameter("global_planner").value,
            "local_planner": self.get_parameter("local_planner").value,
            "animation": self.get_parameter("animation").value,
            "collision": self.get_parameter("collision").value,
        }

        self._rng = RNG(seed)

        self._perception_cache: dict[str, Perception] = {}
        default_name = self._module_selections["perception"]
        self._perception_cache[default_name] = Perception.get(default_name)()
        self._global_planner = GlobalPlanner.create(
            self._module_selections["global_planner"],
        )
        self._local_planner = LocalPlanner.create(
            self._module_selections["local_planner"],
        )
        self._interaction_manager = InteractionManager(rng_manager=self._rng)
        self._animation = MotionAnimation.create(
            self._module_selections["animation"],
        )
        self._collision = CollisionResolver.create(
            self._module_selections["collision"],
        )
        self._wall_aware: tuple[WallAware, ...] = (
            self._local_planner,
            self._global_planner,
            self._collision,
        )

        self._module_pool: dict[str, Any] = {
            self._module_selections["global_planner"]: self._global_planner,
            self._module_selections["local_planner"]: self._local_planner,
            self._module_selections["animation"]: self._animation,
            self._module_selections["collision"]: self._collision,
            default_name: self._perception_cache[default_name],
        }

        self._behavior_trees: dict[int, BehaviourTree | None] = {}
        self._bt_factories: dict[tuple, BehaviorTreeFactory] = {}
        self._world_knowledge = WorldKnowledge()
        self._event_bus = EventBus()
        self._event_scripts: list[EventScript] = []
        self._event_scripts_by_tick: dict[int, list] = {}

        self._waypoint_rng = self._rng.get_substream("waypoint_advance")
        self._spawn_scheduler = SpawnScheduler(
            rng=self._rng.get_substream("spawn_scheduler"),
        )
        self._despawn_monitor = DespawnMonitor()
        self._last_spawned_ids: list[int] = []
        self._last_despawned_ids: list[int] = []

        self._pool = AgentPool()

        self._agent_types: dict[str, AgentType] = dict(BUILTIN_AGENTS)
        self._agents: dict[int, BaseAgent] = {}
        self._pool_agent_ids: list[int] = []
        self._robots: dict[str, tuple[Pose2D, float]] = {}
        self._walls: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._obstacles: dict[str, ObstacleData] = {}
        self._marker_pub = MarkerPublisher(self) if self._publish_markers > 0 else None
        self._tick_count: int = 0
        self._sim_time_ns: int = 0
        self._agent_states_pool = _AgentStateMsgPool()
        self._tick_phases: dict[str, float] = {}
        self._overrun_count: int = 0
        self._last_overrun_log: float = 0.0
        self._high_level_cmds: dict[int, HighLevelCommand] = {}
        self._cached_intermediate_goals: dict[int, Pose2D] = {}
        self._next_agent_id: int = 1

        self._agent_states_pub = self.create_publisher(
            AgentStatesMsg,
            "agent_states",
            10,
        )
        self._agent_states_bg = _BackgroundPublisher(self._agent_states_pub)

        self._world_state_sub = self.create_subscription(
            AgentStatesMsg,
            "world_state",
            self._world_state_callback,
            10,
        )
        self._latest_world_state: AgentStatesMsg | None = None

        self._spawn_srv = self.create_service(
            SpawnAgents,
            "spawn_agents",
            self._spawn_agents_callback,
        )
        self._remove_srv = self.create_service(
            RemoveAgents,
            "remove_agents",
            self._remove_agents_callback,
        )
        self._update_robot_srv = self.create_service(
            UpdateRobot,
            "update_robot",
            self._update_robot_callback,
        )

        self._set_flow_srv = self.create_service(
            SetFlow,
            "set_flow",
            self._set_flow_callback,
        )
        self._add_source_srv = self.create_service(
            AddSource,
            "add_source",
            self._add_source_callback,
        )
        self._remove_source_srv = self.create_service(
            RemoveSource,
            "remove_source",
            self._remove_source_callback,
        )
        self._add_sink_srv = self.create_service(
            AddSink,
            "add_sink",
            self._add_sink_callback,
        )
        self._remove_sink_srv = self.create_service(
            RemoveSink,
            "remove_sink",
            self._remove_sink_callback,
        )
        self._add_walls_srv = self.create_service(
            AddWalls,
            "add_walls",
            self._add_walls_callback,
        )
        self._remove_walls_srv = self.create_service(
            RemoveWalls,
            "remove_walls",
            self._remove_walls_callback,
        )
        self._add_obstacles_srv = self.create_service(
            AddObstacles,
            "add_obstacles",
            self._add_obstacles_callback,
        )
        self._remove_obstacles_srv = self.create_service(
            RemoveObstacles,
            "remove_obstacles",
            self._remove_obstacles_callback,
        )
        self._reset_srv = self.create_service(
            ResetSimulation,
            "reset",
            self._reset_callback,
        )
        self._get_profile_srv = self.create_service(
            GetProfile,
            "get_profile",
            self._get_profile_callback,
        )

        self._timer: rclpy.timer.Timer | None = None
        if self._mode == self.MODE_MASTER:
            self._setup_master_mode()
        elif self._mode == self.MODE_SUBSYSTEM:
            self._setup_subsystem_mode()
        elif self._mode == self.MODE_BENCHMARK:
            self._setup_benchmark_mode()
        else:
            raise ValueError(f"Unknown mode '{self._mode}'. Use '{self.MODE_MASTER}', '{self.MODE_SUBSYSTEM}', or '{self.MODE_BENCHMARK}'.")

        self._sim_logger = None
        if log_dir:
            config_snapshot = {
                "seed": seed,
                "dt": self._dt,
                "bt_tick_interval": self._bt_tick_interval,
                "modules": self._module_selections,
                "parameters": self._collect_ros_parameters(),
            }
            self._sim_logger = SimulationLogger(log_dir, seed, config_snapshot)
            self._logger.info(f"Logging enabled: {log_dir}")

        self._replay = None
        if replay_mode:
            self._replay = ReplayManager()
            self._replay.load(replay_mode)
            self._logger.info(f"Replay mode: loaded {self._replay.tick_count} ticks from {replay_mode}")

        self._logger.info(f"AgentManager initialized (seed={seed}, dt={self._dt}, mode={self._mode})")

    def _collect_ros_parameters(self) -> dict[str, Any]:
        params = {}
        for name in (
            "seed",
            "dt",
            "bt_tick_interval",
            "mode",
            "perception",
            "global_planner",
            "local_planner",
            "animation",
            "log_dir",
            "replay_mode",
        ):
            try:
                params[name] = self.get_parameter(name).value
            except Exception:
                pass
        return params

    def _resolve_perception_layer(self, name: str) -> Perception:
        if name not in self._perception_cache:
            instance = Perception.get(name)()
            self._perception_cache[name] = instance
            self._module_pool[name] = instance
        return self._perception_cache[name]

    def _init_world_knowledge(self, scenario: ScenarioConfig) -> None:
        if scenario.agent_types:
            self._agent_types = {**BUILTIN_AGENTS, **scenario.agent_types}
        self._world_knowledge.clear()
        for wo_cfg in scenario.world_objects:
            obj = WorldObject(
                object_id=wo_cfg.object_id,
                type=wo_cfg.type,
                pose=Pose2D(x=wo_cfg.pose.x, y=wo_cfg.pose.y, theta=wo_cfg.pose.theta),
                capacity=wo_cfg.capacity,
                satisfies=dict(wo_cfg.satisfies),
            )
            self._world_knowledge.add_object(obj)
        self._event_scripts = list(scenario.event_scripts)
        self._event_scripts_by_tick: dict[int, list[EventScript]] = {}
        for script in self._event_scripts:
            self._event_scripts_by_tick.setdefault(script.tick, []).append(script)

    def _build_base_agent(
        self,
        aid: int,
        agent_msg: AgentStateMsg,
        waypoints: Iterable[Pose2D],
    ) -> BaseAgent:
        import attrs

        type_name = getattr(agent_msg, "agent_type", "") or "adult"

        state = AgentState(
            agent_id=aid,
            pose=Pose2D(
                x=agent_msg.pose.x,
                y=agent_msg.pose.y,
                theta=agent_msg.pose.theta,
            ),
            velocity=(agent_msg.velocity.x, agent_msg.velocity.y),
            desired_velocity=agent_msg.desired_velocity if agent_msg.desired_velocity > 0 else 1.3,
        )

        agent_type = resolve_agent_type_name(type_name, self._agent_types)
        if agent_type is not None:
            agent_type = resolve_agent_type(agent_type, self._agent_types)
            rng = self._rng.get_agent_substream(aid, "params")
            agent = create_agent(agent_type, state, self._module_pool, rng)
        else:
            params = SampledParams(
                name=type_name,
                desired_velocity=state.desired_velocity,
                agent_radius=0.35,
                max_velocity=2.0,
                max_acceleration=1.5,
                max_deceleration=2.5,
                min_turning_radius=0.3,
                pivot_angular_velocity=2.0,
                perception_stack=("default",),
                local_planner=self._module_selections["local_planner"],
                global_planner=self._module_selections["global_planner"],
                animation=self._module_selections["animation"],
            )
            agent = create_agent_from_params(params, state, self._module_pool)

        overrides = {}
        radius_val = getattr(agent_msg, "radius", 0.0)
        if radius_val > 0.0:
            overrides["agent_radius"] = radius_val
        vel_val = agent_msg.desired_velocity
        if vel_val > 0.0:
            overrides["desired_velocity"] = vel_val

        perception_overrides = {}
        for field_name, msg_attr in [
            ("vision_range", "vision_range"),
            ("vision_fov", "vision_fov"),
        ]:
            val = getattr(agent_msg, msg_attr, 0.0)
            if val > 0.0:
                perception_overrides[field_name] = val
        if perception_overrides:
            overrides["perception"] = attrs.evolve(
                agent.params.perception,
                **perception_overrides,
            )

        lp_overrides = {}
        for field_name, msg_attr in [
            ("relaxation_time", "relaxation_time"),
            ("repulsion_strength", "repulsion_strength"),
            ("repulsion_range", "repulsion_range"),
        ]:
            val = getattr(agent_msg, msg_attr, 0.0)
            if val > 0.0:
                lp_overrides[field_name] = val
        if lp_overrides:
            overrides["local_planner_params"] = attrs.evolve(
                agent.params.local_planner_params,
                **lp_overrides,
            )

        if overrides:
            agent.params = attrs.evolve(agent.params, **overrides)
            if "desired_velocity" in overrides:
                agent.state.desired_velocity = overrides["desired_velocity"]

        agent.movement = WaypointMovement(waypoints=waypoints)
        return agent

    def _build_base_agent_from_spawn(
        self,
        aid: int,
        spawn_req: SpawnRequest,
    ) -> BaseAgent:
        agent_msg = AgentStateMsg()
        agent_msg.agent_id = aid
        agent_msg.pose = Pose2DMsg(
            x=spawn_req.pose.x,
            y=spawn_req.pose.y,
            theta=spawn_req.pose.theta,
        )
        agent_msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        agent_msg.desired_velocity = spawn_req.desired_velocity
        agent_msg.radius = spawn_req.agent_radius
        agent_msg.agent_type = spawn_req.agent_type
        return self._build_base_agent(aid, agent_msg, list(spawn_req.waypoints))

    def _compile_behavior_tree(self, agent: BaseAgent) -> None:
        aid = agent.state.agent_id
        type_name = agent.params.name
        agent_type = resolve_agent_type_name(type_name, self._agent_types)
        if agent_type is None or agent_type.mode == "simple" or not agent_type.sequences:
            self._behavior_trees[aid] = None
            return

        if agent_type.source_path is not None:
            key = ("path", str(agent_type.source_path))
        else:
            key = ("content", id(agent_type))
        factory = self._bt_factories.get(key)
        if factory is None:
            factory = BehaviorTreeFactory(agent_type)
            self._bt_factories[key] = factory

        bt = factory.build(
            agent=agent,
            world=self._world_knowledge,
            event_bus=self._event_bus,
            rng=self._rng.get_agent_substream(aid, "behavior"),
            dt=self._dt,
        )
        self._behavior_trees[aid] = bt
        if bt is not None:
            agent.movement = BehaviorTreeMovement()

    def _remove_agent(self, aid: int) -> None:
        self._agents.pop(aid, None)
        if aid in self._pool._id_to_idx:
            idx = self._pool._id_to_idx[aid]
            swapped_id = self._pool.swap_remove(aid)
            if swapped_id is not None:
                self._pool_agent_ids[idx] = swapped_id
            self._pool_agent_ids.pop()
        self._high_level_cmds.pop(aid, None)
        self._behavior_trees.pop(aid, None)
        self._interaction_manager.force_stop(aid)
        self._event_bus.clear_agent(aid)
        self._rng.remove_agent_substreams(aid)

    def _phase_end(self, name: str, t0: float):
        self._tick_phases[name] = (time.perf_counter() - t0) * 1000.0

    def _get_profile_callback(self, request: GetProfile.Request, response: GetProfile.Response) -> GetProfile.Response:
        for name, times in self._phase_accum.items():
            response.phase_names.append(name)
            response.phase_means_ms.append(float(np.mean(times)))
            response.phase_p95s_ms.append(float(np.percentile(times, 95)))
        response.n_ticks = sum(len(t) for t in self._phase_accum.values()) // max(len(self._phase_accum), 1)
        response.n_agents = self._pool.n
        if request.reset:
            self._phase_accum.clear()
        return response

    def _flush_profile(self):
        if not self._phase_accum:
            return
        n_agents = self._pool.n
        parts = [f"tick profile ({n_agents} agents, {self._profile_interval} ticks):"]
        means = {name: float(np.mean(times)) for name, times in self._phase_accum.items()}
        total = sum(means.values())
        for name, times in self._phase_accum.items():
            mean = means[name]
            pct = mean / total * 100 if total > 0 else 0
            p95 = np.percentile(times, 95)
            parts.append(f"  {name:<16s} {pct:5.1f}%  mean={mean:.3f}ms  p95={p95:.3f}ms")
        parts.append(f"  {'TOTAL':<16s} 100.0%  mean={total:.3f}ms  budget={self._dt * 1000:.1f}ms  rtf={self._dt * 1000 / total:.2f}" if total > 0 else "")
        self._logger.info("\n".join(parts))
        self._phase_accum.clear()

    def tick(self):
        self._last_spawned_ids = []
        self._last_despawned_ids = []
        self._tick_phases = {}

        t0 = time.perf_counter()
        despawn_requests = self._despawn_monitor.tick(
            agents=self._agents,
            interaction_check=self._interaction_manager.is_in_interaction,
            tick_count=self._tick_count,
            dt=self._dt,
        )
        for req in despawn_requests:
            if req.force:
                self._interaction_manager.force_stop(req.agent_id)
            self._remove_agent(req.agent_id)
            self._spawn_scheduler.notify_despawn(req.agent_id)
            self._despawn_monitor.unregister(req.agent_id)
            self._last_despawned_ids.append(req.agent_id)
            if self._sim_logger is not None:
                self._sim_logger.record_agent_despawn(
                    agent_id=req.agent_id,
                    reason=req.reason,
                    tick=self._tick_count,
                )
        self._phase_end("despawn", t0)

        t0 = time.perf_counter()
        spawn_requests = self._spawn_scheduler.tick(self._tick_count, self._dt)
        for spawn_req in spawn_requests:
            aid = self._next_agent_id
            self._next_agent_id += 1
            agent = self._build_base_agent_from_spawn(aid, spawn_req)
            self._agents[aid] = agent
            self._pool.add_agent(agent)
            self._pool_agent_ids.append(aid)
            self._compile_behavior_tree(agent)
            spawn_req.lifetime.agent_id = aid
            self._despawn_monitor.register(aid, spawn_req.lifetime)
            self._spawn_scheduler.register_agent(aid, spawn_req.lifetime.source_name)
            self._last_spawned_ids.append(aid)

            mv = agent.movement
            if isinstance(mv, WaypointMovement) and mv.waypoints:
                self._high_level_cmds[aid] = HighLevelCommand(
                    agent_id=aid,
                    type=CommandType.NAVIGATE,
                    target_pose=mv.waypoints[mv.index],
                    desired_velocity=spawn_req.desired_velocity,
                )

            if self._sim_logger is not None:
                self._sim_logger.record_agent_spawn(
                    agent_id=aid,
                    params=agent.params,
                    agent_type_name=agent.params.name,
                )
        self._phase_end("spawn", t0)

        agents = [self._agents[aid] for aid in self._pool_agent_ids]
        pool = self._pool
        is_bt_tick = self._tick_count % self._bt_tick_interval == 0

        world_state = self._consume_world_state()

        # --- SENSE (vectorized perception -> CSR) ---
        t0 = time.perf_counter()
        default_layer = next(iter(self._perception_cache.values()), None)
        if default_layer is not None and default_layer.supports_pool:
            default_layer.compute_pool(pool)

        n = pool.n

        # build beliefs only for BT agents or agents with extra perception
        if is_bt_tick or any(len(a.perception) > 1 for a in agents):
            indptr = pool.neighbor_indptr
            indices = pool.neighbor_indices
            any_extra = any(len(a.perception) > 1 for a in agents)
            agent_states = {aid: agent.state for aid, agent in self._agents.items()} if any_extra else None
            for i, agent in enumerate(agents):
                has_bt = self._behavior_trees.get(agent.state.agent_id) is not None
                has_extra = len(agent.perception) > 1
                if has_bt or has_extra:
                    belief = BeliefState(agent_id=agent.state.agent_id)
                    nbr_idxs = indices[indptr[i] : indptr[i + 1]].tolist()
                    belief.observed_agents = [agents[j].state for j in nbr_idxs]
                    for layer in agent.perception[1:]:
                        belief = layer.compute(agent, agent_states, world_state, belief)
                    agent.belief = belief

        self._run_extra_modules(agents, TickPhase.SENSE)
        self._phase_end("sense", t0)

        # --- DECIDE (BT tick, per-agent) ---
        if is_bt_tick:
            t0 = time.perf_counter()
            pool.sync_back(agents)

            self._process_event_scripts()

            for agent_id in self._pool_agent_ids:
                bt = self._behavior_trees.get(agent_id)
                if bt is not None:
                    bt: BehaviourTree
                    bt.tick()

            for agent_id, agent in self._agents.items():
                mv = agent.movement
                if isinstance(mv, BehaviorTreeMovement) and mv.command is not None:
                    self._high_level_cmds[agent_id] = mv.command

            self._event_bus.clear()
            self._phase_end("decide", t0)

            # --- GLOBAL PLAN ---
            t0 = time.perf_counter()
            for planner, group in _group_by(agents, key=lambda a: a.global_planner):
                planner.compute(group, self._high_level_cmds)

            self._cached_intermediate_goals = {}
            for planner, _group in _group_by(agents, key=lambda a: a.global_planner):
                self._cached_intermediate_goals.update(planner.get_cached_goals())

            pool.set_goals(self._cached_intermediate_goals)
            self._phase_end("global_plan", t0)

        # --- LOCAL PLAN (vectorized SFM or per-agent fallback) ---
        t0 = time.perf_counter()
        pool.store_prev_vel()
        if self._local_planner.supports_pool:
            self._local_planner.compute_pool(pool, store_forces=self._publish_markers >= 2, dt=self._dt)
        else:
            self._local_plan_fallback(agents, self._cached_intermediate_goals, pool)

        self._run_extra_modules(agents, TickPhase.PLAN)
        self._phase_end("local_plan", t0)

        # --- INTERACTIONS (sequential, unchanged) ---
        t0 = time.perf_counter()
        interactions = self._interaction_manager.update(
            self._high_level_cmds,
            dt=self._dt,
        )

        for interaction in interactions.values():
            if interaction.outcome != InteractionOutcome.ACTIVE:
                for pid in interaction.participants:
                    agent = self._agents.get(pid)
                    if agent is not None and isinstance(agent.movement, BehaviorTreeMovement):
                        agent.movement.last_outcome = interaction.outcome
            if interaction.object_id:
                self._world_knowledge.set_queue_length(
                    interaction.object_id,
                    interaction.contract.queue_length,
                )
        self._phase_end("interactions", t0)

        # --- KINEMATICS (vectorized) ---
        t0 = time.perf_counter()
        self._apply_kinematic_constraints_vectorized(pool)
        self._phase_end("kinematics", t0)

        # --- ANIMATION ---
        t0 = time.perf_counter()
        for anim, _ in _group_by(agents, key=lambda a: a.animation):
            anim.compute_batch_pool(pool, interactions, self._dt)

        self._run_extra_modules(agents, TickPhase.ACT)
        self._phase_end("animation", t0)

        # --- INTEGRATION (vectorized) ---
        t0 = time.perf_counter()
        self._integrate_state_vectorized(pool)
        self._phase_end("integrate", t0)

        # --- COLLISION RESOLVE ---
        t0 = time.perf_counter()
        self._collision.resolve(pool)
        self._phase_end("collision", t0)

        # --- WAYPOINTS + PUBLISH (read from pool, no sync needed) ---
        t0 = time.perf_counter()
        self._advance_waypoints(agents, pool)
        msg = self._build_agent_states_msg()
        self._agent_states_bg.publish(msg)

        if self._marker_pub is not None:
            pool.sync_back(agents)
            mlvl = self._publish_markers
            publish_behavior(self._marker_pub, agents, self._high_level_cmds)
            publish_interaction(self._marker_pub, agents, interactions)
            publish_infrastructure(
                self._marker_pub,
                self._spawn_scheduler._sources,
                self._despawn_monitor._sinks,
                self._walls,
                self._world_knowledge._objects,
                self._obstacles,
            )
            if mlvl >= 2:
                velocities = {int(pool.agent_ids[i]): (float(pool.vel[i, 0]), float(pool.vel[i, 1])) for i in range(n)}
                publish_perception(self._marker_pub, agents)
                publish_global_plan(
                    self._marker_pub,
                    agents,
                    self._high_level_cmds,
                    self._cached_intermediate_goals,
                )
                publish_local_plan(self._marker_pub, agents, velocities)
                publish_waypoints(self._marker_pub, agents)
                modules = list(self._perception_cache.values())
                modules.append(self._global_planner)
                modules.append(self._local_planner)
                publish_module_markers(self._marker_pub, modules)
            self._marker_pub.flush()
        self._phase_end("publish", t0)

        if self._publish_markers == 0:
            pool.sync_back(agents)

        if self._sim_logger is not None:
            self._sim_logger.record_tick(
                tick=self._tick_count,
                timestamp=self._tick_count * self._dt,
                agents={aid: agent.state for aid, agent in self._agents.items()},
                interactions=interactions,
                commands=self._high_level_cmds,
            )

        self._tick_count += 1

        if self._profile_phases:
            for name, ms in self._tick_phases.items():
                self._phase_accum.setdefault(name, []).append(ms)
            if self._profile_interval > 0 and self._tick_count % self._profile_interval == 0:
                self._flush_profile()

    def _apply_kinematic_constraints_vectorized(self, pool: AgentPool) -> None:
        n = pool.n
        if n == 0:
            return
        dt = self._dt

        new_vel = pool.vel[:n].copy()
        cur_vel = pool.prev_vel[:n]
        r_min = pool.min_turning_radius[:n]
        w_pivot = pool.pivot_angular_velocity[:n]
        max_acc = pool.max_acceleration[:n]
        max_dec = pool.max_deceleration[:n]
        max_v = pool.max_velocity[:n]

        cur_speed = np.linalg.norm(cur_vel, axis=1)
        moving = cur_speed > self._min_speed_for_heading

        max_ang = np.maximum(w_pivot, cur_speed / np.maximum(r_min, 1e-9))

        desired_angle = np.arctan2(new_vel[:, 1], new_vel[:, 0])
        current_angle = np.arctan2(cur_vel[:, 1], cur_vel[:, 0])
        angle_delta = np.arctan2(
            np.sin(desired_angle - current_angle),
            np.cos(desired_angle - current_angle),
        )
        max_angle = max_ang * dt
        angle_delta = np.clip(angle_delta, -max_angle, max_angle)
        clamped_angle = current_angle + angle_delta
        desired_speed = np.linalg.norm(new_vel, axis=1)
        clamped_vel = np.column_stack(
            [
                desired_speed * np.cos(clamped_angle),
                desired_speed * np.sin(clamped_angle),
            ]
        )
        new_vel = np.where(moving[:, None], clamped_vel, new_vel)

        # acceleration clamping
        accel = new_vel - cur_vel
        accel_mag = np.linalg.norm(accel, axis=1)

        dot_new_cur = np.sum(new_vel * cur_vel, axis=1)
        dot_cur_cur = np.sum(cur_vel * cur_vel, axis=1)
        is_decel = dot_new_cur < dot_cur_cur
        limit = np.where(is_decel, max_dec, max_acc)
        max_dv = limit * dt

        needs_clamp = (accel_mag > 1e-9) & (accel_mag > max_dv)
        scale = np.where(needs_clamp, max_dv / np.maximum(accel_mag, 1e-9), 1.0)
        new_vel = np.where(
            needs_clamp[:, None],
            cur_vel + accel * scale[:, None],
            new_vel,
        )

        # speed clamping
        speed = np.linalg.norm(new_vel, axis=1)
        too_fast = speed > max_v
        new_vel = np.where(
            too_fast[:, None],
            new_vel * (max_v / np.maximum(speed, 1e-9))[:, None],
            new_vel,
        )

        pool.vel[:n] = new_vel

    def _local_plan_fallback(
        self,
        agents: Iterable[BaseAgent],
        intermediate_goals: dict[int, Pose2D],
        pool: AgentPool,
    ) -> None:
        moving = []
        for agent in agents:
            aid = agent.state.agent_id
            goal = intermediate_goals.get(aid)
            if goal is None:
                continue
            dx = agent.state.pose.x - goal.x
            dy = agent.state.pose.y - goal.y
            if dx * dx + dy * dy < self._waypoint_threshold**2:
                continue
            moving.append(agent)
        vel_dict: dict[int, tuple[float, float]] = {}
        for planner, group in _group_by(moving, key=lambda a: a.local_planner):
            vel_dict.update(planner.compute(group, intermediate_goals, dt=self._dt))
        for i in range(pool.n):
            aid = int(pool.agent_ids[i])
            v = vel_dict.get(aid, (0.0, 0.0))
            pool.vel[i, 0] = v[0]
            pool.vel[i, 1] = v[1]

    def _integrate_state_vectorized(self, pool: AgentPool) -> None:
        n = pool.n
        if n == 0:
            return
        dt = self._dt
        vel = pool.vel[:n]
        pos = pool.pos[:n]
        theta = pool.theta[:n]

        pos += vel * dt

        speed = np.linalg.norm(vel, axis=1)
        moving = speed > self._min_speed_for_heading
        target_theta = np.arctan2(vel[:, 1], vel[:, 0])
        delta = np.arctan2(
            np.sin(target_theta - theta),
            np.cos(target_theta - theta),
        )
        r_min = pool.min_turning_radius[:n]
        w_pivot = pool.pivot_angular_velocity[:n]
        max_d = np.maximum(w_pivot, speed / np.maximum(r_min, 1e-9)) * dt
        delta = np.clip(delta, -max_d, max_d)
        theta += np.where(moving, delta, 0.0)

    def _advance_waypoints(self, agents: Iterable[BaseAgent], pool: AgentPool) -> None:
        for i, agent in enumerate(agents):
            mv = agent.movement
            if not isinstance(mv, WaypointMovement):
                continue
            wps = mv.waypoints
            if len(wps) < 2:
                continue
            goal = wps[mv.index]
            r = mv.radii[mv.index] if mv.radii else 0.0
            if r <= 0.0:
                r = self._waypoint_threshold
            dx = float(pool.pos[i, 0]) - goal.x
            dy = float(pool.pos[i, 1]) - goal.y
            dist_sq = dx * dx + dy * dy
            if dist_sq > r * r:
                continue

            idx = mv.index
            n = len(wps)

            if mv.mode == WaypointMode.REPEAT:
                idx = (idx + 1) % n
            elif mv.mode == WaypointMode.REVERSE:
                if mv.forward:
                    if idx >= n - 1:
                        mv.forward = False
                        idx = max(idx - 1, 0)
                    else:
                        idx += 1
                else:
                    if idx <= 0:
                        mv.forward = True
                        idx = min(idx + 1, n - 1)
                    else:
                        idx -= 1
            elif mv.mode == WaypointMode.ONCE:
                if idx >= n - 1:
                    continue
                idx += 1
            elif mv.mode == WaypointMode.RANDOM:
                others = [j for j in range(n) if j != idx]
                idx = int(self._waypoint_rng.choice(others))

            mv.index = idx
            aid = agent.state.agent_id
            self._high_level_cmds[aid] = HighLevelCommand(
                agent_id=aid,
                type=CommandType.NAVIGATE,
                target_pose=wps[idx],
                desired_velocity=float(pool.desired_vel[i]),
            )

    def _run_extra_modules(self, agents: Iterable[BaseAgent], phase: TickPhase):
        from arena_humansim.agents.base import VectorizedModule

        modules_agents: dict[Any, list[BaseAgent]] = {}
        for agent in agents:
            for mod in agent.modules.values():
                if mod.phase() == phase:
                    modules_agents.setdefault(mod, []).append(agent)
        for mod, group in modules_agents.items():
            if isinstance(mod, VectorizedModule):
                mod.step_pool(self._pool, self._pool.n, self._dt)
            else:
                mod.step_batch(group, self._dt)

    def _process_event_scripts(self) -> None:
        for script in self._event_scripts_by_tick.get(self._tick_count, ()):
            if script.target_agent == -1:
                self._event_bus.fire_broadcast(script.event)
            else:
                self._event_bus.fire(script.event, script.target_agent)

    def run_replay(self) -> ReplayResult | None:
        if self._replay is None:
            return None
        result = self._replay.replay(self, logger=self._logger)
        if result.success:
            self._logger.info(f"Replay verification passed: {result.total_ticks} ticks")
        else:
            div = result.first_divergence
            self._logger.warn(f"Replay DIVERGED at tick {div.tick}, agent {div.agent_id}: {div.detail}")
        return result

    def _setup_master_mode(self):
        clock_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._clock_pub = self.create_publisher(Clock, "/clock", clock_qos)
        self._timer = self.create_timer(self._dt, self._master_timer_callback)

    def _setup_subsystem_mode(self):
        # Timer drives ticking at dt intervals, paced by the external /clock
        self._timer = self.create_timer(self._dt, self._subsystem_timer_callback)
        # Accumulated spawn/despawn IDs returned to callers via feedback
        self._accumulated_spawned: list[int] = []
        self._accumulated_despawned: list[int] = []
        # Service receives feedback (robot poses) from orchestrator
        self._feedback_srv = self.create_service(
            Feedback,
            "feedback",
            self._feedback_callback,
        )

    def _check_overrun(self, elapsed: float):
        if elapsed <= self._dt:
            return
        self._overrun_count += 1
        now = time.perf_counter()
        if now - self._last_overrun_log < 5.0:
            return
        self._last_overrun_log = now
        phases = self._tick_phases
        breakdown = ", ".join(f"{name}={ms:.1f}ms" for name, ms in phases.items() if ms > 0.5)
        self._logger.warn(f"sim cannot keep up: tick took {elapsed * 1000:.1f}ms (budget {self._dt * 1000:.1f}ms), {self._overrun_count} overrun(s) since last report, {self._pool.n} agents" + (f" [{breakdown}]" if breakdown else ""))
        self._overrun_count = 0

    def _master_timer_callback(self):
        t0 = time.perf_counter()
        self.tick()
        self._check_overrun(time.perf_counter() - t0)
        self._sim_time_ns += int(self._dt * 1e9)
        clock_msg = Clock()
        clock_msg.clock.sec = int(self._sim_time_ns // int(1e9))
        clock_msg.clock.nanosec = int(self._sim_time_ns % int(1e9))
        self._clock_pub.publish(clock_msg)

    def _subsystem_timer_callback(self):
        now = self.get_clock().now()
        self._sim_time_ns = now.nanoseconds
        t0 = time.perf_counter()
        self.tick()
        self._check_overrun(time.perf_counter() - t0)
        self._accumulated_spawned.extend(self._last_spawned_ids)
        self._accumulated_despawned.extend(self._last_despawned_ids)

    def _setup_benchmark_mode(self):
        clock_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._clock_pub = self.create_publisher(Clock, "/clock", clock_qos)
        self._timer = self.create_timer(0.0, self._benchmark_timer_callback)

    def _benchmark_timer_callback(self):
        self.tick()
        self._sim_time_ns += int(self._dt * 1e9)
        clock_msg = Clock()
        clock_msg.clock.sec = int(self._sim_time_ns // int(1e9))
        clock_msg.clock.nanosec = int(self._sim_time_ns % int(1e9))
        self._clock_pub.publish(clock_msg)

    def _feedback_callback(
        self,
        request: Feedback.Request,
        response: Feedback.Response,
    ) -> Feedback.Response:
        for robot in request.robots:
            name = robot.name or "robot"
            pose = Pose2D(
                x=robot.pose.x,
                y=robot.pose.y,
                theta=robot.pose.theta,
            )
            radius = robot.radius if robot.radius > 0 else 0.3
            self._robots[name] = (pose, radius)

        response.success = True
        response.spawned_ids = self._accumulated_spawned
        response.despawned_ids = self._accumulated_despawned
        self._accumulated_spawned = []
        self._accumulated_despawned = []
        self._logger.debug(f"feedback: {len(request.robots)} robots, spawned={len(response.spawned_ids)}, despawned={len(response.despawned_ids)}")
        return response

    def _world_state_callback(self, msg: AgentStatesMsg):
        self._latest_world_state = msg

    def _consume_world_state(self) -> WorldState:
        world: WorldState = {}
        if self._latest_world_state is not None:
            for agent_msg in self._latest_world_state.agents:
                world[agent_msg.agent_id] = WorldAgentState(
                    pose=Pose2D(
                        x=agent_msg.pose.x,
                        y=agent_msg.pose.y,
                        theta=agent_msg.pose.theta,
                    ),
                    velocity=(agent_msg.velocity.x, agent_msg.velocity.y),
                )
            self._latest_world_state = None
        return world

    def _build_agent_states_msg(self) -> AgentStatesMsg:
        pool = self._pool
        n = pool.n
        msg = self._agent_states_pool.get(n)
        msg.header.stamp.sec = int(self._sim_time_ns // int(1e9))
        msg.header.stamp.nanosec = int(self._sim_time_ns % int(1e9))
        if n == 0:
            return msg
        for i in range(n):
            a = msg.agents[i]
            a.agent_id = int(pool.agent_ids[i])
            a.pose.x = float(pool.pos[i, 0])
            a.pose.y = float(pool.pos[i, 1])
            a.pose.theta = float(pool.theta[i])
            a.velocity.x = float(pool.vel[i, 0])
            a.velocity.y = float(pool.vel[i, 1])
            a.velocity.z = 0.0
            a.desired_velocity = float(pool.desired_vel[i])
            a.radius = float(pool.agent_radius[i])
        return msg

    def _spawn_agents_callback(
        self,
        request: SpawnAgents.Request,
        response: SpawnAgents.Response,
    ) -> SpawnAgents.Response:
        spawned_ids = []
        for agent_msg in request.agents:
            aid = agent_msg.agent_id
            if aid <= 0:
                aid = self._next_agent_id
                self._next_agent_id += 1

            waypoints = [Pose2D(x=pt.pose.x, y=pt.pose.y, theta=pt.pose.theta) for pt in agent_msg.waypoints.points]
            radii = [pt.radius for pt in agent_msg.waypoints.points]

            agent = self._build_base_agent(aid, agent_msg, waypoints)
            agent.movement = WaypointMovement(
                waypoints=waypoints,
                radii=radii,
                mode=WaypointMode(agent_msg.waypoints.mode),
            )
            self._agents[aid] = agent
            self._pool.add_agent(agent)
            self._pool_agent_ids.append(aid)
            self._compile_behavior_tree(agent)

            mv = agent.movement
            if isinstance(mv, WaypointMovement) and mv.waypoints:
                self._high_level_cmds[aid] = HighLevelCommand(
                    agent_id=aid,
                    type=CommandType.NAVIGATE,
                    target_pose=mv.waypoints[mv.index],
                    desired_velocity=agent_msg.desired_velocity,
                )

            if self._sim_logger is not None:
                self._sim_logger.record_agent_spawn(
                    agent_id=aid,
                    params=agent.params,
                    agent_type_name=agent.params.name,
                )

            spawned_ids.append(aid)
            if aid >= self._next_agent_id:
                self._next_agent_id = aid + 1
        response.success = True
        response.message = f"Spawned {len(spawned_ids)} agent(s)"
        response.spawned_ids = spawned_ids
        self._logger.info(response.message)
        return response

    def _remove_agents_callback(
        self,
        request: RemoveAgents.Request,
        response: RemoveAgents.Response,
    ) -> RemoveAgents.Response:
        if not request.agent_ids or -1 in request.agent_ids:
            count = len(self._agents)
            self._agents.clear()
            self._pool_agent_ids.clear()
            self._pool.reset()
            self._high_level_cmds.clear()
            self._behavior_trees.clear()
            self._despawn_monitor.clear()
            self._spawn_scheduler.reset_counts()
            self._interaction_manager.interactions.clear()
            self._event_bus.clear()
            self._next_agent_id = 1
            self._tick_count = 0
            response.success = True
            response.message = f"Removed all {count} agent(s)"
        else:
            removed = 0
            for aid in request.agent_ids:
                if aid in self._agents:
                    self._remove_agent(aid)
                    self._despawn_monitor.unregister(aid)
                    self._spawn_scheduler.notify_despawn(aid)
                    removed += 1
            response.success = True
            response.message = f"Removed {removed} agent(s)"
        self._logger.info(response.message)
        return response

    def _reset_callback(
        self,
        request: ResetSimulation.Request,
        response: ResetSimulation.Response,
    ) -> ResetSimulation.Response:
        self._agents.clear()
        self._pool_agent_ids.clear()
        self._pool.reset()
        self._high_level_cmds.clear()
        self._behavior_trees.clear()
        self._despawn_monitor.clear()
        self._spawn_scheduler.reset_counts()
        self._spawn_scheduler.clear_sources()
        self._despawn_monitor.clear_sinks()
        self._interaction_manager.interactions.clear()
        self._event_bus.clear()
        self._event_scripts.clear()
        self._event_scripts_by_tick.clear()
        self._walls.clear()
        self._obstacles.clear()
        for subsystem in self._wall_aware:
            subsystem.set_walls([])
        self._world_knowledge.clear()
        self._rng.reset()
        self._next_agent_id = 1
        self._tick_count = 0
        self._sim_time_ns = 0
        self._last_spawned_ids = []
        self._last_despawned_ids = []
        response.success = True
        response.message = "Simulation reset"
        self._logger.info(response.message)
        return response

    def _update_robot_callback(
        self,
        request: UpdateRobot.Request,
        response: UpdateRobot.Response,
    ) -> UpdateRobot.Response:
        name = request.name or "robot"
        pose = Pose2D(
            x=request.pose.x,
            y=request.pose.y,
            theta=request.pose.theta,
        )
        radius = request.radius if request.radius > 0 else 0.3
        self._robots[name] = (pose, radius)
        response.success = True
        self._logger.debug(f"update_robot: {name} at ({pose.x:.2f}, {pose.y:.2f}), r={radius:.2f}")
        return response

    @staticmethod
    def _shape_msg_to_shape(shape_msg: ShapeMsg) -> Shape:
        from arena_humansim.utils.types import ShapeType

        _RECT = 0
        _shape_type_map = {1: ShapeType.CIRCLE, 2: ShapeType.POLYGON}
        stype = _shape_type_map.get(shape_msg.type, ShapeType.POLYGON)
        vertices = [Pose2D(x=v.x, y=v.y) for v in shape_msg.vertices]
        if shape_msg.type == _RECT and shape_msg.width > 0 and shape_msg.height > 0:
            hw, hh = shape_msg.width / 2.0, shape_msg.height / 2.0
            vertices = [
                Pose2D(x=-hw, y=-hh),
                Pose2D(x=hw, y=-hh),
                Pose2D(x=hw, y=hh),
                Pose2D(x=-hw, y=hh),
            ]
        return Shape(type=stype, radius=shape_msg.radius, vertices=vertices)

    @staticmethod
    def _source_msg_to_config(src_msg: SourceConfigMsg) -> SourceConfig:
        from arena_humansim.utils.types import AgentTemplate, RateKeyframe, SinkAffinity

        return SourceConfig(
            name=src_msg.name,
            pose=Pose2D(x=src_msg.pose.x, y=src_msg.pose.y, theta=src_msg.pose.theta),
            shape=AgentManager._shape_msg_to_shape(src_msg.shape),
            rate_profile=[RateKeyframe(t=kf.t, rate=kf.rate) for kf in src_msg.rate_profile],
            max_concurrent=src_msg.max_concurrent,
            max_total=src_msg.max_total,
            agent=AgentTemplate(
                desired_velocity_min=src_msg.agent.desired_velocity_min,
                desired_velocity_max=src_msg.agent.desired_velocity_max,
                agent_radius=src_msg.agent.agent_radius,
                agent_type=src_msg.agent.agent_type,
                sink_affinity=[SinkAffinity(sink_name=sa.sink_name, weight=sa.weight) for sa in src_msg.agent.sink_affinity],
            ),
        )

    @staticmethod
    def _sink_msg_to_config(sink_msg: SinkConfigMsg) -> SinkConfig:
        return SinkConfig(
            name=sink_msg.name,
            pose=Pose2D(x=sink_msg.pose.x, y=sink_msg.pose.y, theta=sink_msg.pose.theta),
            shape=AgentManager._shape_msg_to_shape(sink_msg.shape),
            absorption_radius=sink_msg.absorption_radius,
            capacity=sink_msg.capacity,
        )

    def _set_flow_callback(self, request: SetFlow.Request, response: SetFlow.Response) -> SetFlow.Response:
        self._spawn_scheduler.clear_sources()
        self._despawn_monitor.clear_sinks()

        sources = []
        for src_msg in request.flow.sources:
            src = self._source_msg_to_config(src_msg)
            sources.append(src)
            self._spawn_scheduler.add_source(src)

        sinks = {}
        for sink_msg in request.flow.sinks:
            sink = self._sink_msg_to_config(sink_msg)
            sinks[sink.name] = sink
            self._despawn_monitor.add_sink(sink)

        self._spawn_scheduler.set_sinks(sinks)

        response.success = True
        response.message = f"Set {len(sources)} source(s), {len(sinks)} sink(s)"
        self._logger.info(response.message)
        return response

    def _add_source_callback(self, request: AddSource.Request, response: AddSource.Response) -> AddSource.Response:
        src = self._source_msg_to_config(request.source)
        self._spawn_scheduler.add_source(src)
        response.success = True
        response.message = f"Added source {src.name}"
        response.name = src.name
        self._logger.debug(response.message)
        return response

    def _remove_source_callback(self, request: RemoveSource.Request, response: RemoveSource.Response) -> RemoveSource.Response:
        if not request.name:
            self._spawn_scheduler.clear_sources()
            response.message = "Removed all sources"
        else:
            self._spawn_scheduler.remove_source(request.name)
            response.message = f"Removed source {request.name}"
        response.success = True
        self._logger.debug(response.message)
        return response

    def _add_sink_callback(self, request: AddSink.Request, response: AddSink.Response) -> AddSink.Response:
        sink = self._sink_msg_to_config(request.sink)
        self._despawn_monitor.add_sink(sink)
        self._spawn_scheduler.set_sinks(self._despawn_monitor.sinks)
        response.success = True
        response.message = f"Added sink {sink.name}"
        response.name = sink.name
        self._logger.debug(response.message)
        return response

    def _remove_sink_callback(self, request: RemoveSink.Request, response: RemoveSink.Response) -> RemoveSink.Response:
        if not request.name:
            self._despawn_monitor.clear_sinks()
            self._spawn_scheduler.set_sinks({})
            response.message = "Removed all sinks"
        else:
            self._despawn_monitor.remove_sink(request.name)
            self._spawn_scheduler.set_sinks(self._despawn_monitor.sinks)
            response.message = f"Removed sink {request.name}"
        response.success = True
        self._logger.debug(response.message)
        return response

    def _all_wall_segments(self) -> Segments:
        """Collect wall segments from both explicit walls and obstacle bounding boxes."""
        segments = list(self._walls.values())
        for obs in self._obstacles.values():
            segments.extend(obs.wall_segments)
        return segments

    def _refresh_planners(self):
        """Push current wall segments to local planner, global planner, and collision."""
        segments = self._all_wall_segments()
        for subsystem in self._wall_aware:
            subsystem.set_walls(segments)

    def _add_walls_callback(self, request: AddWalls.Request, response: AddWalls.Response) -> AddWalls.Response:
        for name, start, end in zip(request.names, request.starts, request.ends, strict=True):
            self._walls[name] = ((start.x, start.y), (end.x, end.y))
        self._refresh_planners()
        response.success = True
        response.message = f"Added {len(request.names)} wall(s), total {len(self._walls)}"
        self._logger.debug(response.message)
        return response

    def _remove_walls_callback(self, request: RemoveWalls.Request, response: RemoveWalls.Response) -> RemoveWalls.Response:
        if not request.names:
            self._walls.clear()
            response.message = "Removed all walls"
        else:
            for name in request.names:
                self._walls.pop(name, None)
            response.message = f"Removed {len(request.names)} wall(s), remaining {len(self._walls)}"
        self._refresh_planners()
        response.success = True
        self._logger.debug(response.message)
        return response

    @staticmethod
    def _decompose_obstacle_to_walls(
        pose: Pose2D,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Convert a rotated 2D bounding box into 4 wall segments."""
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        corners_local = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min),
        ]
        corners_world = []
        for lx, ly in corners_local:
            wx = pose.x + cos_t * lx - sin_t * ly
            wy = pose.y + sin_t * lx + cos_t * ly
            corners_world.append((wx, wy))

        segments = []
        for i in range(4):
            j = (i + 1) % 4
            segments.append((corners_world[i], corners_world[j]))
        return segments

    def _add_obstacles_callback(self, request: AddObstacles.Request, response: AddObstacles.Response) -> AddObstacles.Response:
        added = 0
        for obs_msg in request.obstacles:
            if obs_msg.name in self._obstacles:
                continue  # noop if already exists
            pose = Pose2D(x=obs_msg.pose.x, y=obs_msg.pose.y, theta=obs_msg.pose.theta)
            wall_segments = self._decompose_obstacle_to_walls(
                pose,
                obs_msg.bb_x_min,
                obs_msg.bb_x_max,
                obs_msg.bb_y_min,
                obs_msg.bb_y_max,
            )
            self._obstacles[obs_msg.name] = ObstacleData(
                name=obs_msg.name,
                pose=pose,
                bb=(
                    obs_msg.bb_x_min,
                    obs_msg.bb_x_max,
                    obs_msg.bb_y_min,
                    obs_msg.bb_y_max,
                    obs_msg.bb_z_min,
                    obs_msg.bb_z_max,
                ),
                interaction_types=tuple(obs_msg.interaction_types),
                obstacle_type=obs_msg.obstacle_type,
                wall_segments=tuple(wall_segments),
            )
            added += 1
        if added:
            self._refresh_planners()
        response.success = True
        response.message = f"Added {added} obstacle(s), total {len(self._obstacles)}"
        self._logger.debug(response.message)
        return response

    def _remove_obstacles_callback(self, request: RemoveObstacles.Request, response: RemoveObstacles.Response) -> RemoveObstacles.Response:
        if not request.names:
            self._obstacles.clear()
            response.message = "Removed all obstacles"
        else:
            for name in request.names:
                self._obstacles.pop(name, None)
            response.message = f"Removed {len(request.names)} obstacle(s), remaining {len(self._obstacles)}"
        self._refresh_planners()
        response.success = True
        self._logger.debug(response.message)
        return response

    def destroy_node(self):
        if self._timer is not None:
            self._timer.cancel()
        if self._profile_phases:
            self._flush_profile()
        if self._sim_logger is not None:
            self._sim_logger.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="arena_humansim node")
    parser.add_argument("--profile", action="store_true", help="Enable per-phase tick profiling")
    parser.add_argument("--profile-interval", type=int, default=0, help="Ticks between profile log dumps (0 = flush at shutdown only)")
    parsed, remaining = parser.parse_known_args(args)

    rclpy.init(args=remaining)
    node = AgentManager()
    if parsed.profile:
        node._profile_phases = True
    if parsed.profile_interval:
        node._profile_interval = parsed.profile_interval
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
