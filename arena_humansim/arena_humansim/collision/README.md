# Collision resolvers

Final step of the tick pipeline. Removes agent-wall overlap introduced by the integrator. Agent-agent collisions are handled by the local planner (SFM/ORCA) and are out of scope here.

## Available

| Name | Class | Notes |
|---|---|---|
| `wall_projection` | `WallProjectionResolver` | Projects overlapping agents along the wall normal, plus a tangent wall-slide on velocity. Up to 3 relaxation passes per tick. Default. |
| `noop` | `NoopCollisionResolver` | Does nothing. Use when the scenario has no walls or overlap is tolerable. |

## Contract

```python
class CollisionResolver(WallAware, Loggable, ABC):
    @abstractmethod
    def resolve(self, pool: AgentPool) -> None: ...
```

- Mutates `pool.pos` and optionally `pool.vel` in place.
- `set_walls(segments)` feeds the current wall set; cache any derived arrays (e.g. segment endpoints, AB vectors) once per wall change.
- Must be idempotent in steady state: resolving a collision-free pool must be a no-op.
- Must be deterministic under a fixed wall set + pool state.

## `wall_projection` semantics

For each agent within `radius + margin` of a wall segment:

1. Find the closest point on the segment (clipped to endpoints).
2. Push the agent out along the outward normal by `(radius + margin) - dist`.
3. Project velocity onto the wall tangent - agents slide along walls rather than bouncing or sticking.

Three relaxation passes are run to settle corner cases where resolving wall A drives the agent into wall B. After that, residual overlap is accepted.

## Adding a resolver

1. Subclass `CollisionResolver` under `collision/`.
2. Implement `resolve(pool)`. Override `set_walls` if you need derived caches.
3. Register in `collision/__init__.py` via a `_load_<name>` lazy loader.
4. Contract coverage: `tests/contracts/test_collision_contract.py` - no-walls no-op, deterministic, bounded-iterations.

Agent-wall `margin` defaults to 1 cm. Increase if you observe jitter at contact; decrease only if you're sure the local planner is keeping agents off walls on its own.
