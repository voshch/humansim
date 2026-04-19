# Arena HumanSim

Modular, deterministic pedestrian simulator for the [Arena](https://github.com/Arena-Rosnav) framework. Simulates crowds of autonomous agents with realistic navigation, collision avoidance, social interactions, and behavior trees — all running as a ROS 2 node.

## Architecture

```
AgentManager (ROS 2 Node)                    [core/]
├── AgentPool          Vectorized NumPy state for all agents
├── Perception         KDTree neighbor queries + FOV filtering       [perception/]
├── Behavior Trees     py_trees decision making + needs system       [core/behavior/]
├── Global Planner     A* pathfinding on inflated occupancy grid     [global_planner/]
├── Local Planner      SFM or ORCA collision avoidance               [local_planner/]
├── Animation          Kinematic forward integration                 [animation/]
├── Collision          Wall projection overlap resolution            [collision/]
├── SpawnScheduler     Poisson-process agent spawning at sources
├── DespawnMonitor     Sink-based agent removal with TTL
├── InteractionManager Social interactions (talk, follow, queue, …)  [core/]
├── EventBus           Event-driven scripting
└── SimulationLogger   JSON replay logging
```

Per-module READMEs: [core](arena_humansim/arena_humansim/core/README.md) · [core/behavior](arena_humansim/arena_humansim/core/behavior/README.md) · [core/behavior/nodes](arena_humansim/arena_humansim/core/behavior/nodes/README.md) · [perception](arena_humansim/arena_humansim/perception/README.md) · [global_planner](arena_humansim/arena_humansim/global_planner/README.md) · [local_planner](arena_humansim/arena_humansim/local_planner/README.md) · [animation](arena_humansim/arena_humansim/animation/README.md) · [collision](arena_humansim/arena_humansim/collision/README.md).

### Tick Loop

Each simulation step follows a fixed pipeline:

1. **Spawn / Despawn** — schedule new agents, remove completed ones
2. **Sense** — build neighbor graph (KDTree + FOV pruning → CSR sparse matrix)
3. **Decide** — tick behavior trees (every N ticks), emit high-level commands
4. **Global Plan** — A* pathfinding with LOS simplification and wall push-back
5. **Local Plan** — SFM / ORCA velocity computation (vectorized over pool)
6. **Interact** — update social interaction state machines
7. **Kinematics** — enforce acceleration, speed, and turning-radius limits
8. **Animate** — forward-integrate position and heading
9. **Collide** — resolve agent-wall overlaps
10. **Publish** — broadcast `AgentStates` message + optional RViz markers

## Modules

All modules are swappable via a plugin registry.

| Layer | Options | Default |
|---|---|---|
| [Global Planner](arena_humansim/arena_humansim/global_planner/README.md) | `dijkstra` | `dijkstra` |
| [Local Planner](arena_humansim/arena_humansim/local_planner/README.md) | `sfm`, `orca` | `sfm` |
| [Perception](arena_humansim/arena_humansim/perception/README.md) | `default` | `default` |
| [Animation](arena_humansim/arena_humansim/animation/README.md) | `noop`, `kinematic` | `noop` |
| [Collision](arena_humansim/arena_humansim/collision/README.md) | `wall_projection`, `noop` | `wall_projection` |

## Agent Types

Defined in YAML under [`config/agent_types/`](arena_humansim/config/agent_types/README.md). Each type specifies distributions over physical and behavioral parameters:

```yaml
name: adult
desired_velocity: { mean: 1.1, std: 0.12, clip_low: 0.5, clip_high: 1.5 }
agent_radius: { mean: 0.25, std: 0.02 }
perception:
  vision_range: { mean: 5.0, std: 0.5 }
  vision_fov: { mean: 180.0, std: 10.0 }
local_planner_params:
  relaxation_time: { mean: 0.5 }
  repulsion_strength: { mean: 2.1 }
needs:
  hunger: { initial: { mean: 100 }, decay_rate: { mean: 0.5 } }
```

Types support inheritance via `extends`. Parameters are sampled per-agent from their distributions, producing a heterogeneous crowd from a single type definition.

**Included types:** `adult`, `elder`

## Behavior Trees

Agents can operate in two movement modes:

- **Waypoint** — follow an explicit waypoint list (repeat / reverse / once / random)
- **Behavior Tree** — py_trees decision tree driven by needs, perceptions, and events

BTs are compiled from the agent type's `sequences`, `actions`, and `needs`. Needs decay over time and trigger actions when thresholds are crossed (e.g. hunger < 30 → eat).

**Authoring the YAML:** [`config/agent_types/README.md`](arena_humansim/config/agent_types/README.md#behavior-trees) documents `needs`, `actions`, `sequences`, `steps`, `transitions`, accept steps, and autonomous steps with full field tables and examples.

**Internals / extending primitives:** [`core/behavior/README.md`](arena_humansim/arena_humansim/core/behavior/README.md) covers cross-node invariants (nav-before-advertise, patience, accept semantics). [`core/behavior/nodes/README.md`](arena_humansim/arena_humansim/core/behavior/nodes/README.md) covers adding new `py_trees.Behaviour` primitives.

## Interactions

Matcher semantics (advertise/search/accept, visibility gating, queueing, service binding) live in [`core/README.md`](arena_humansim/arena_humansim/core/README.md). The `interaction_radius` cascade is documented in [`config/scenarios/README.md`](arena_humansim/config/scenarios/README.md).

This is very early-stage now, needs expansion.

| Type | Participants | Description |
|---|---|---|
| `TALK_TO` | 2 | Face-to-face conversation |
| `GROUP_CONVERSATION` | 2+ | Multi-agent group talk |
| `FOLLOW` | 2 | One agent follows another |
| `SIT_ON` / `LIE_ON` | 1 | Occupy furniture |
| `USE` | 1 | Use a world object |
| `QUEUE_USE` | 1+ | Queue for a shared resource |

Interactions follow an advertise → search → accept → stop protocol. Agents in active interactions defer despawn until the interaction ends.

## Spawning & Despawning

- **Sources** emit agents at a configurable Poisson rate, respecting `max_concurrent` and `max_total` caps
- **Sinks** absorb agents that reach them (within `absorption_radius`)
- Agents are routed from source to sink via weighted `sink_affinity`
- TTL-based removal as fallback

## Determinism & Replay

The simulator is fully deterministic given a seed. Seeded RNG substreams are maintained per agent and per component.

`SimulationLogger` writes per-tick JSON snapshots. `ReplayManager` replays them tick-by-tick and validates state within floating-point tolerance (1e-12).

## Usage

### Launch

```bash
ros2 launch arena_humansim arena_humansim.launch.py \
  mode:=master \
  use_sim_time:=false \
  markers:=true \
  rviz:=true
```

Scenarios (world objects, agents, flow, walls) are authored under [`config/scenarios/`](arena_humansim/config/scenarios/README.md).

### Run node directly

```bash
ros2 run arena_humansim arena_humansim_node \
  --ros-args -p mode:=master -p seed:=42 -p dt:=0.05
```

### Benchmark

```bash
ros2 run arena_humansim benchmark
```

Config format and stage semantics: [`config/benchmark/README.md`](arena_humansim/config/benchmark/README.md).

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `mode` | `master` | `master` / `subsystem` |
| `seed` | `0` | RNG seed for deterministic runs |
| `dt` | `0.05` | Simulation timestep (s) |
| `bt_tick_interval` | `5` | BT ticks every N sim ticks |
| `global_planner` | `dijkstra` | Global planner module |
| `local_planner` | `sfm` | Local planner module |
| `animation` | `noop` | Animation module |
| `collision` | `wall_projection` | Collision resolver |
| `publish_markers` | `0` | RViz markers: 0=off, 1=infrastructure+labels+interactions, 2=full |
| `log_dir` | `""` | Directory for replay logs |

### ROS Interface

**Publishes:**
- `agent_states` (`AgentStatesMsg`) — all agent positions, velocities, states

**Subscribes:**
- `world_state` — external robot state updates

**Services:**
- `spawn_agents`, `remove_agents` — direct agent control
- `add_source`, `remove_source`, `add_sink`, `remove_sink` — flow control
- `add_walls`, `remove_walls` — dynamic obstacles
- `set_flow` — bulk configure sources, sinks, walls
- `reset` — clear all simulation state

## Development

Test layout (unit / contracts / integration / ros / perf / replay): [`arena_humansim/tests/README.md`](arena_humansim/tests/README.md).

### Linting

Linting is handled by [Ruff](https://docs.astral.sh/ruff/), driven by [pre-commit](https://pre-commit.com/). Config lives in `arena_humansim/pyproject.toml`; the hook pin is in `.pre-commit-config.yaml`. Auto-formatting is intentionally not enforced.

**One-time setup:**
```bash
pip install pre-commit
pre-commit install
```

**Everyday use:** hooks run automatically on `git commit` against staged files. To run manually:
```bash
pre-commit run            # staged files only
pre-commit run -a         # entire repo
ruff check arena_humansim # check without pre-commit
```

If the hook auto-fixes something, the commit is aborted and the fixes are left unstaged — `git add` and re-commit.

### CI

`.github/workflows/lint.yml` runs the same pre-commit hooks on every push to `master` and every pull request. The GH check uses the exact config and hook pins from `.pre-commit-config.yaml`, so local and CI never drift. Make the check required in branch protection to block merges on lint failures.

Bump the Ruff version with `pre-commit autoupdate`.