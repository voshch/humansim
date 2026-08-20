# Global planners

Turn a `HighLevelCommand` (usually `NAVIGATE target_pose`) into the next global subgoal for each agent. Runs before the local planner each tick. Responsible for pathfinding, LOS simplification, and wall push-back.

## Available

| Name | Class | Notes |
|---|---|---|
| `dijkstra` | `DijkstraPlanner` | All-pairs shortest-path on an inflated occupancy grid, cached until walls change. Default. |
| `astar` | `AStarPlanner` | Per-agent A* on the same grid. Cheaper for small crowds, scales worse than cached Dijkstra as N grows. |

## Contract

```python
class GlobalPlanner(WallAware, Loggable, ABC):
    @abstractmethod
    def compute(self, agents, high_level_commands) -> dict[int, Pose2D]: ...

    @abstractmethod
    def get_cached_goals(self) -> dict[int, Pose2D]: ...

    def get_cached_paths(self) -> dict[int, list[Pose2D]]: ...
```

- `compute` returns the *next immediate subgoal* per agent - not the final target. Advancement along the cached path is handled by `advance_along_path` (shared base-class helper) using dot-product projection.
- `get_cached_goals` is how the rest of the pipeline (local planner, markers) reads the planner state without re-triggering work.
- `get_cached_paths` is optional; only used for visualization.
- `WallAware.set_walls(segments)` invalidates any grid / graph caches.

## Shared helpers

- `simplify_path(waypoints, min_area=0.01)` - Visvalingam-Whyatt decimation, in the base module. Use this before returning paths so LOS-connected runs collapse to endpoints.
- `GlobalPlanner.advance_along_path(agent_pos, waypoints, current_idx)` - dot-product projection; returns the new cursor index.

## Adding a planner

1. Subclass `GlobalPlanner` in a new file under `global_planner/`.
2. Implement `compute` + `get_cached_goals`. Use `simplify_path` for LOS cleanup.
3. Register in `global_planner/__init__.py` via a `_load_<name>` lazy loader.
4. Add contract coverage in `tests/contracts/test_global_planner_contract.py` and efficacy coverage in `tests/efficacy/test_global_planner_efficacy.py`.

## Benchmarks

See [config/benchmark/astar_vs_dijkstra.yaml](../../config/benchmark/astar_vs_dijkstra.yaml) for the canonical A*-vs-Dijkstra sweep.
