# Local planners

Velocity commands for the next tick, given each agent's global subgoal and neighbors. Called from the tick pipeline between `global_plan` and `kinematics`.

## Available

| Name | Class | Notes |
|---|---|---|
| `sfm` | `SFMPlanner` | Social Force Model. `supports_pool=True` — vectorized NumPy path. Per-kind gain scales for human↔robot. Default. |
| `orca` | `ORCAPlanner` | Reciprocal velocity obstacles. Per-agent `solve`, no pool path. |
| `straight` | `StraightToGoalPlanner` | Ignores neighbors, drives toward the subgoal at `desired_velocity`. `supports_pool=True`. For debugging and robot policies that don't want avoidance. |

## Contract

```python
class LocalPlanner(WallAware, Loggable, ABC):
    supports_pool: bool = False       # opt into compute_pool fast-path
    needs_global_subgoal: bool = True # set False to skip global planning

    @abstractmethod
    def compute(self, agents, global_goals, dt) -> dict[int, (vx, vy)]: ...

    def compute_pool(self, pool, store_forces=False, dt=1.0) -> None: ...
```

- `compute` is the per-agent fallback and is always required.
- When `supports_pool=True`, `AgentManager` calls `compute_pool` with the whole `AgentPool` and the planner writes back into `pool.vel` directly — no per-agent Python dict allocation. This is the hot path; prefer it for anything that needs to scale past ~50 agents.
- `WallAware.set_walls(segments)` is called once per wall change; cache any derived structures (e.g. closest-point arrays) there, not in `compute*`.
- `publish_markers(pub)` is optional; SFM emits force arrows when `publish_markers=2`.

## Adding a planner

1. Subclass `LocalPlanner` in a new file under `local_planner/`.
2. Implement `compute`. Optionally set `supports_pool=True` and implement `compute_pool` for the vectorized path.
3. Register it in `local_planner/__init__.py` via a `_load_<name>` lazy loader + `_registry.register("<name>")(_load_<name>)` — the lazy pattern keeps import cost off the registry walk.
4. Drop a contract test under `tests/contracts/test_local_planner_contract.py` and an efficacy test under `tests/efficacy/test_local_planner_efficacy.py`.

See the contract-test file for the exact invariants (velocity clipping, `set_walls` idempotency, pool/non-pool agreement) that gate every new planner.

## Parameter sampling

Per-agent local-planner params live under `local_planner_params:` in each agent type yaml (see [config/agent_types/](../../../config/agent_types/)). They're sampled once at spawn and stored on the agent.
