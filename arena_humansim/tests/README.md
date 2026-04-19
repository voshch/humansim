# Tests

Six directories, each a different *kind* of test. Layout is load-bearing: `conftest.py` wires ROS only for `integration/` and `ros/`, and `pytest_collection_modifyitems` skips both when `rclpy` isn't on `PYTHONPATH`.

## Layout

| Dir | Purpose | Needs ROS? |
|---|---|---|
| `unit/` | Fast, isolated component tests. No ROS, no pool mutation across tests. Biggest bucket (41 files). | No |
| `contracts/` | Plugin-contract tests that iterate `_registry` for each module type (local_planner, global_planner, perception, animation, collision). Every registered implementation must pass the same suite — this is how new plugins earn their keep. | No |
| `efficacy/` | Per-module behavior tests: does SFM actually repel, does the planner reach the goal, do FOV limits prune neighbors. Iterates `_registry` like contracts but asserts *outcomes* rather than *invariants*. | No |
| `property/` | Hypothesis-based property tests for RNG substreams, pool invariants, planner output bounds, collision idempotency, scenario-loader round-trips. Cheap fuzzing. | No |
| `integration/` | End-to-end tests that construct a real `AgentManager`, tick it, and assert on message output / pool state. Covers determinism, replay, BT tick cadence, subsystem stamps, staggered spawn. | Yes |
| `ros/` | Service / topic tests that talk to a live ROS node via the `RosTestSystem` fixture. Spawn/remove/walls/reset services, waypoint topics, world geometry publishing. | Yes |

## Running

```bash
pytest arena_humansim/tests/unit          # fast loop, no ROS
pytest arena_humansim/tests/contracts     # still fast, no ROS
pytest arena_humansim/tests               # everything, skips ROS suites if rclpy missing
```

ROS-requiring suites skip with reason `ROS2 not discoverable — source install/setup.bash to enable` when `rclpy` or `arena_humansim_msgs.srv` don't import. Source the workspace's `install/setup.bash` before invoking pytest if you want them to run.

## Shared fixtures (`conftest.py`)

Declared once at the tests root, visible everywhere:

- `rng`, `rng_pair` — seed-42 `RNG` instances.
- `walls_empty`, `walls_simple` — canned wall layouts.
- `agent_factory(agent_id, x, y)` — minimal `BaseAgent` with sampled `adult` params and null planners. Use when the code under test only needs the agent's state + params.
- `pool_empty(capacity)`, `pool_with_agents(n)` — `AgentPool` factories.
- `commands_factory(agent_ids, target)` — dicts of `HighLevelCommand` for planner inputs.
- `minimal_scenario` — smallest-possible `ScenarioConfig`.
- `rclpy_context` (session, autouse) — initializes `rclpy` once per session when available; no-ops otherwise.

`tests/ros/_helpers.py` and `tests/integration/_helpers.py` own the ROS-specific fixtures (`ros_system`, `RosTestSystem`, spawn/remove request builders).

## Authoring rules

- **New plugin under `local_planner/`, `global_planner/`, `perception/`, `animation/`, `collision/`** → add coverage to the matching `contracts/` and `efficacy/` file. Both iterate `_registry`; registering the plugin is enough for the contract test to find it.
- **New invariant on pool, RNG, collision, or scenario** → prefer `property/` over a single hand-picked example. Hypothesis catches edge cases worth catching.
- **New service, topic, or parameter on the ROS interface** → add a `ros/` test using `RosTestSystem`. Integration tests go in `integration/` only if you need a full tick loop.
- **Determinism is assumed.** Any new test that reads wall-clock time, `random` (not seeded `RNG`), or system state is wrong. Fix the test, don't loosen the assertion.

## Efficacy vs contract: what goes where

The split is deliberate.

- **Contract** = "this shape of input produces this shape of output, and this invariant never breaks." Loads every registered implementation and runs the same suite against each. Example: `test_local_planner_contract.py::test_velocity_within_max_speed`.
- **Efficacy** = "under scenario X, this implementation produces the *right* answer." Often has implementation-specific tolerances. Example: `test_local_planner_efficacy.py::test_sfm_repels_from_neighbor`.

A failing contract test blocks the plugin from being usable. A failing efficacy test means the plugin is usable but wrong for its stated purpose.
