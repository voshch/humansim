# Scenarios

Scenarios live here as yaml, loaded by `arena_humansim/utils/scenario.py`.

## `interaction_radius` (meters)

How close an agent must get to a target before its BT step may emit `ADVERTISE` for an interaction. Cascade (first non-null wins):

1. `StepDef.interaction_radius` — per-step override in the sequence definition.
2. `WorldObject.interaction_radius` — per-object override on the target.
3. Interaction-type default (see table below).
4. `DISTANCE_TOLERANCE` = 0.5 m (from `utils/const.py`).

### Type defaults

| Interaction        | Default (m)          |
|--------------------|----------------------|
| USE                | DISTANCE_TOLERANCE   |
| QUEUE_USE          | DISTANCE_TOLERANCE   |
| SIT_ON             | DISTANCE_TOLERANCE   |
| LIE_ON             | DISTANCE_TOLERANCE   |
| GROUP_CONVERSATION | 3.0                  |
| TALK_TO            | 2.0                  |
| FOLLOW             | 5.0                  |
| SERVICE            | 3.0                  |

Source of truth: `DEFAULT_INTERACTION_RADIUS` in `core/interaction_manager.py`.

### Example

```yaml
world_objects:
  gathering_area:
    object_id: plaza
    type: gathering_area
    pose: { x: 0.0, y: 0.0 }
    interaction_radius: 2.0   # override the 3.0 GROUP_CONVERSATION default

sequences:
  loiter:
    steps:
      join_conversation:
        target_object_type: gathering_area
        interaction: GROUP_CONVERSATION
        interaction_radius: 1.5   # tighter still, overrides object + type
        patience: { mean: 60.0 }
```

Anchorless steps (no `target_object_*`) skip the proximity gate; they advertise immediately and the interaction manager pairs them via perception visibility.

## Robot services

Robot agents (`kind: 1`) can advertise named functions via a `services:` list. Each tick, `RobotServiceAdvertiser` emits one `ADVERTISE` command per declared service, producing parallel SERVICE interactions that humans bind to.

```yaml
agents:
  - agent_id: 10
    kind: 1
    policy: sfm
    services:
      - tag: water
        max_participants: 4
      - tag: trash
        max_participants: -1   # unbounded
```

- `tag: str` — free-form identifier. Matched string-equal against human accepters' tags (see `AcceptServiceNode`).
- `max_participants: int` — total participants cap, **counting the robot itself**. Set `N+1` for "N humans at once". Use `-1` for unbounded.
- One robot, multiple services — emits one ad per tag per tick, yielding N concurrent interactions on disjoint participants.
- Two robots sharing a tag produce two independent interactions, one centered on each robot.

Matcher semantics (visibility, anchor-on-robot, queueing) live in [core/README.md](../../arena_humansim/core/README.md). Accepter-side BT node is `AcceptServiceNode` (see [core/behavior/README.md](../../arena_humansim/core/behavior/README.md)).
