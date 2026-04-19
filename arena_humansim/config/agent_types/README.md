# Agent types

Each `.yaml` file defines one agent type. Types are *distributions*, not fixed parameter sets — the loader samples a concrete instance per agent at spawn from the `{mean, std, clip_low, clip_high}` triples.

## Fields

Physical:

- `desired_velocity`, `max_velocity` (m/s)
- `max_acceleration`, `max_deceleration` (m/s²)
- `agent_radius` (m)
- `min_turning_radius` (m)
- `pivot_angular_velocity` (rad/s)

Perception (`perception:`):

- `vision_range` (m)
- `vision_fov` (degrees, full angle — 360 = omnidirectional)

Local planner (`local_planner_params:`):

- `relaxation_time` — SFM time constant for reaching `desired_velocity`
- `repulsion_strength`, `repulsion_range` — SFM neighbor repulsion term

Every scalar field is a distribution dict: `{mean, std, clip_low, clip_high}`. Omit `std` (defaults to 0) for a fixed value. `clip_*` bounds are applied after sampling; pick them to keep unphysical samples out (e.g. `desired_velocity.clip_low: 0.5` — nobody walks at 5 cm/s).

## Inheritance

```yaml
name: elder_slow
extends: elder
desired_velocity: {mean: 0.5, std: 0.08, clip_low: 0.2, clip_high: 0.8}
```

`extends:` pulls in the parent's full field set, then this file's fields override. Nested structures (`perception`, `local_planner_params`) merge field-by-field, not as whole replacements. Inheritance chains resolve in the loader; see `arena_humansim/core/agents/__init__.py:resolve_agent_type`.

## Behavior trees

Set `mode: behavior_tree` to drive the agent from a compiled py_trees BT. `mode: simple` (default) ignores `sequences` / `actions` / `needs` and runs waypoint-only movement.

BT authoring is split into four fields on the agent type: `needs`, `utility_weights`, `actions`, `sequences` (plus `initial_sequence` to pick the entry sequence; defaults to `"default"`). See [../../arena_humansim/core/behavior/README.md](../../arena_humansim/core/behavior/README.md) for the cross-node invariants the compiler enforces (nav-before-advertise, patience, accept semantics).

### `needs`

Scalar state in `[0, 100]` that decays every tick and is restored by actions / steps via `satisfies`.

```yaml
needs:
  thirst:
    initial: {mean: 55.0, std: 15.0, clip_low: 25.0, clip_high: 90.0}
    decay_rate: {mean: 8.0, std: 2.0, clip_low: 4.0, clip_high: 14.0}  # units/sec
```

`initial` and `decay_rate` are distribution dicts, sampled per agent. Decay is linear; `satisfies: {thirst: 100.0}` adds 100 on completion (clamped at 100).

### `utility_weights`

Per-need weights used by `AutonomousNode` when a step has `autonomous: true`. Higher weight ⇒ need drives action selection more aggressively.

### `actions`

Candidate actions the autonomous selector can pick from. Fields:

| Field | Meaning |
|---|---|
| `when` | `{need: {below|above: X}}` — preconditions gating the action. |
| `interaction` | `TALK_TO`, `GROUP_CONVERSATION`, `FOLLOW`, `SIT_ON`, `LIE_ON`, `USE`, `QUEUE_USE`, `WAVE_AT`, `BLOCK`, `SERVICE`. Omit for nav-only. |
| `target_object_type` / `target_object_id` | Object the action resolves to. Type = any matching object; id = exact. |
| `duration` | Distribution (seconds). |
| `patience` | Distribution (seconds) capping the whole action (nav + execute). |
| `satisfies` | `{need: amount}` applied on SUCCESS. |

```yaml
actions:
  rest_at_bench:
    target_object_type: bench
    duration: {mean: 5.0, clip_low: 1.0, clip_high: 15.0}
    satisfies: {rest: 30.0}
```

### `sequences`

Named state machines. Each sequence runs its `steps` in declaration order; on completion the compiler routes via `then` (chain) or `transitions` (need-driven preemption) or `on_failure` (recovery).

```yaml
sequences:
  chat:
    steps:
      hold_conversation:
        target_object_type: gathering_area
        interaction: GROUP_CONVERSATION
        duration: {mean: 45.0, std: 15.0}
        patience: {mean: 60.0}
    then: chat                       # loop
    transitions:
      - when: {thirst: {below: 30.0}}
        goto: drink                  # preempt into another sequence
  drink:
    steps:
      queue_and_drink:
        target_object_type: fountain
        interaction: USE
        duration: {mean: 6.0}
        satisfies: {thirst: 100.0}
    then: chat
```

`transitions` evaluate every tick — they can cut a step short. `then` only fires on the last step succeeding. `on_failure: <seq_name>` routes to another sequence when the current one FAILs (unset ⇒ propagate failure to the root). `interruptible: false` disables `transitions` for the sequence.

### `steps`

Each step is either a `StepDef` (anchored to an object or agent) or a `GoToStepDef` (fixed pose). StepDef fields:

| Field | Meaning |
|---|---|
| `target_object_type` / `target_object_id` | Nav target. Omit both for a pure-wait step. |
| `target_agent` | Agent id to pursue. Routes through `BlockNode` → `BLOCK` interaction, with velocity boost and lookahead prediction. |
| `interaction` | Same enum as actions. Pairs with `target_object_*` to emit nav→advertise. |
| `duration` | Distribution (seconds). |
| `patience` | Distribution (seconds). Covers nav + advertise + wait-for-outcome. |
| `satisfies` | `{need: amount}` applied on SUCCESS. |
| `autonomous` | `true` ⇒ `AutonomousNode` scores `actions` and runs the winner. |
| `allowed_actions` / `blocked_actions` | Filter the autonomous candidate pool. |
| `until` | Event-bus event name that exits the autonomous step on fire (e.g. `"agent_ready"`). |
| `until_need` | Need-condition predicate that exits the autonomous step (e.g. `{rest: {above: 80}}`). |
| `interruptible` | Per-step override of the sequence flag. |
| `interaction_radius` | Per-step override of the radius cascade; see [../scenarios/README.md](../scenarios/README.md). |
| `accept` | `true` ⇒ passive accepter (no nav, no resolve). Requires `interaction`; rejects `target_object_*`. |
| `service_tag` | Free-form tag for `SERVICE` matching when `accept: true`. |

GoToStepDef takes `target_pose: {x, y, theta}` instead of a target reference — use for scripted waypoints.

### Pure-wait step

Omit both `target_object_*` and `interaction`; keep `duration`. Produces a `HoldNode` — the agent stops and waits.

### Accept step

```yaml
sell_water:
  accept: true
  interaction: SERVICE
  service_tag: water
  patience: {mean: 30.0}
```

No nav, no advertise-by-object; the accepter binds to any robot advertising the matching `service_tag`. Covered in detail in [../../arena_humansim/core/behavior/README.md](../../arena_humansim/core/behavior/README.md#accept-steps).

### Where this gets compiled

Per-agent trees are built in [../../arena_humansim/core/behavior/compiler.py](../../arena_humansim/core/behavior/compiler.py) from the sampled `AgentType`. To extend the authoring surface (new step fields, new primitive kinds), see [../../arena_humansim/core/behavior/nodes/README.md](../../arena_humansim/core/behavior/nodes/README.md).

## Adding a type

1. Drop a yaml file here (or under a scenario's `agent_types/` to keep it local).
2. Either `extends` an existing type or declare every field explicitly — there's no implicit default.
3. Reference it by file name (stem, without `.yaml`) from scenarios.

## Where this gets sampled

`AgentLoader.create_agent(agent_type_name, rng)` draws one concrete parameter set per agent. The RNG is the agent's seeded substream, so the same seed + type + agent_id gives the same sampled instance across runs.

## Included types

- `adult` — nominal pedestrian (desired 1.1 m/s, 5 m vision, 180° FOV).
- `elder` — slower, narrower FOV, longer SFM relaxation. Demonstrates how heterogeneity drops out of a handful of distribution tweaks.
