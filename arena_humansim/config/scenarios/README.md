# Scenarios

Scenarios live here as yaml, loaded by [../../arena_humansim/utils/scenario.py](../../arena_humansim/utils/scenario.py).

## Interaction step fields

Every BT step that joins or creates an interaction uses `SeekNode` under the hood. The per-step yaml maps directly to a `SeekSpec` on the emitted `SEEK` command. Which fields are valid depends on the `InteractionType`'s handle kind (see [../../arena_humansim/core/interaction_kinds.py](../../arena_humansim/core/interaction_kinds.py)):

| Handle   | Interactions                                   | `target:` shape              | `offer:`              |
|----------|------------------------------------------------|------------------------------|-----------------------|
| `NONE`   | `TALK_TO`, `GROUP_CONVERSATION`, `WAVE_AT`     | omitted                      | not allowed           |
| `OBJECT` | `USE`, `SIT_ON`, `LIE_ON`, `QUEUE_USE`         | `str` (object id or type)    | not allowed           |
| `TAG`    | `SERVICE`                                      | `str` (service tag; required for `offer: true`, optional for seekers) | provider side only |
| `AGENT`  | `BLOCK`                                        | `int` (agent id)             | not allowed           |

`WorldKnowledge.resolve` treats an `OBJECT` target as object id first, then falls back to object type (nearest visible). The loader rejects mismatched shapes (e.g. `target: <str>` on BLOCK, `offer: true` on TALK_TO).

### Provider-side flags (SERVICE only, with `offer: true`)

- `min_participants: int` — ACTIVE threshold; counts the provider itself.
- `max_participants: int` — cap (counts the provider). `-1` = unbounded.
- `queueable: bool` — admit seekers into a FIFO queue when full.
- `formation_spec: {type, anchor_kind, params}` — formation imposed on participants. Common choice for mobile providers: `{type: line, anchor_kind: provider, params: {base_step: 0.8}}` (line trails the provider's pose, slot-0 is the provider and is skipped by the formation emitter).

### Ending an interaction

- `duration: {mean: ..., std: ...}` — contract-level timeout. IM tears down with outcome `COMPLETED` when elapsed.
- `{cancel: true}` as its own follow-up step — emits STOP on the agent's current `interaction_id` with `reason=CANCELED`.

Pick one. Mixing them on the same step is allowed (duration is a fallback if Cancel never runs).

## `interaction_radius` (meters)

Two roles, same number:

- **Request:** how close an agent must get to an object-bound target before `SeekNode` is reached in the compiled sequence.
- **Drift eviction:** how far a participant may drift from the nearest peer (or from the bound object) once `ACTIVE`, scaled by `cohesion_multiplier` (default 1.2 in `InteractionManager`). A participant past `interaction_radius * cohesion_multiplier` is evicted with `INTERRUPTED`. Tightening this field tightens both gates together.

Cascade (first non-null wins):

1. `StepDef.interaction_radius` — per-step override.
2. `WorldObject.interaction_radius` — per-object override on the target.
3. `InteractionKind.interaction_radius` — per-type default (see below).
4. `DISTANCE_TOLERANCE` = 0.5 m (from [../../arena_humansim/utils/const.py](../../arena_humansim/utils/const.py)).

### Type defaults

| Interaction          | Default (m)          |
|----------------------|----------------------|
| `USE`                | `DISTANCE_TOLERANCE` |
| `QUEUE_USE`          | `DISTANCE_TOLERANCE` |
| `SIT_ON`             | `DISTANCE_TOLERANCE` |
| `LIE_ON`             | `DISTANCE_TOLERANCE` |
| `GROUP_CONVERSATION` | 3.0                  |
| `TALK_TO`            | 2.0                  |
| `SERVICE`            | 3.0                  |
| `WAVE_AT`            | `DISTANCE_TOLERANCE` |
| `BLOCK`              | `DISTANCE_TOLERANCE` |

Source of truth: the `InteractionKind.interaction_radius` fields in [../../arena_humansim/core/interaction_kinds.py](../../arena_humansim/core/interaction_kinds.py).

### Object override

```yaml
world_objects:
  - object_id: plaza
    type: gathering_area
    pose: {x: 0.0, y: 0.0}
    interaction_radius: 2.0      # overrides the 3.0 GROUP_CONVERSATION default

sequences:
  loiter:
    steps:
      join_conversation:
        target: plaza            # object id
        interaction: GROUP_CONVERSATION
        interaction_radius: 1.5  # tighter still, overrides object + type
        patience: {mean: 60.0}
```

## Examples by handle shape

### Symmetric peer (NONE)

No `target:`, no `offer:`. Both peers run the same step; whichever ticks first creates, the other joins.

```yaml
chat:
  interaction: TALK_TO
  duration: {mean: 8.0, std: 2.0}
  patience: {mean: 30.0}
```

### Object-anchored (OBJECT)

`target:` is an object id or object type. The compiler emits `Resolve → GoTo → Seek`.

```yaml
queue_and_drink:
  target: fountain               # object id, or "fountain" as a type
  interaction: USE
  duration: {mean: 6.0, std: 1.5}
  satisfies: {thirst: 100.0}
```

### Service provider (TAG, `offer: true`)

Creates the interaction; seekers join. Mobile provider uses line formation anchored on itself.

```yaml
offer_ride:
  interaction: SERVICE
  target: escort_ride            # service tag (free-form string)
  offer: true
  min_participants: 2
  max_participants: 2
  queueable: false
  formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}
  patience: {mean: 60.0}
```

### Service seeker (TAG, no `offer`)

Polls for a visible provider of the matching tag.

```yaml
hail:
  interaction: SERVICE
  target: escort_ride
  patience: {mean: 90.0}
```

### Agent-targeted (AGENT — BLOCK)

`target:` is an agent id; `BlockNode` pursues and emits SEEK on arrival.

```yaml
intercept:
  interaction: BLOCK
  target: 42                     # agent id to block
  duration: {mean: 3.0}
```

### Mobile provider (escort / shuttle)

Offer, drive, cancel. The interaction stays live across the drive; line formation trails the provider.

```yaml
offer_A:
  interaction: SERVICE
  target: escort_ride
  offer: true
  min_participants: 2
  max_participants: 2
  queueable: false
  formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}
ride_to_B: {kind: go_to, target_pose: {x: 7.0, y: 0.0}}
drop_B:    {cancel: true}
```

## Robot services

Robot agents (`kind: 1`) can offer named services via a `services:` list. Each tick, `RobotServiceAdvertiser` emits one `SEEK` command with `offer: true` per declared service, producing parallel `SERVICE` interactions that humans bind to.

```yaml
agents:
  - agent_id: 10
    kind: 1
    policy: sfm
    services:
      - tag: water
        max_participants: 4
      - tag: trash
        max_participants: -1     # unbounded
```

- `tag: str` — free-form identifier. Matched string-equal against human seekers' `target:`.
- `max_participants: int` — cap, **counting the robot itself**. Set `N+1` for "N humans at once". Use `-1` for unbounded.
- Seeker side is a plain `{interaction: SERVICE, target: <tag>}` step — same node as any other service seeker.

Matcher semantics (visibility, provider anchor, queueing) live in [../../arena_humansim/core/README.md](../../arena_humansim/core/README.md).
