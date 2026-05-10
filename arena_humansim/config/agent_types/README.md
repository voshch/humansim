# Agent types

Each `.yaml` file defines one agent type. Types are *distributions*, not fixed parameter sets - the loader samples a concrete instance per agent at spawn from the `{mean, std, clip_low, clip_high}` triples.

## Fields

Physical:

- `desired_velocity`, `max_velocity` (m/s)
- `max_acceleration`, `max_deceleration` (m/s^2)
- `agent_radius` (m)
- `min_turning_radius` (m)
- `pivot_angular_velocity` (rad/s)

Perception (`perception:`):

- `vision_range` (m)
- `vision_fov` (degrees, full angle - 360 = omnidirectional)

Local planner (`local_planner_params:`):

- `relaxation_time` - SFM time constant for reaching `desired_velocity`
- `repulsion_strength`, `repulsion_range` - SFM neighbor repulsion term

Every scalar field is a distribution dict: `{mean, std, clip_low, clip_high}`. Omit `std` (defaults to 0) for a fixed value. `clip_*` bounds are applied after sampling; pick them to keep unphysical samples out (e.g. `desired_velocity.clip_low: 0.5` - nobody walks at 5 cm/s).

## Inheritance

```yaml
name: elder_slow
extends: elder
desired_velocity: {mean: 0.5, std: 0.08, clip_low: 0.2, clip_high: 0.8}
```

`extends:` pulls in the parent's full field set, then this file's fields override. Nested structures (`perception`, `local_planner_params`) merge field-by-field, not as whole replacements. Inheritance chains resolve in the loader; see [../../arena_humansim/core/agents/__init__.py](../../arena_humansim/core/agents/__init__.py) (`resolve_agent_type`).

## Behavior trees

Set `mode: behavior_tree` to drive the agent from a compiled py_trees BT. `mode: simple` (default) ignores `sequences` / `actions` / `needs` and runs waypoint-only movement.

BT authoring is split into four fields on the agent type: `needs`, `utility_weights`, `actions`, `sequences` (plus `initial_sequence` to pick the entry sequence; defaults to `"default"`). See [../../arena_humansim/core/behavior/README.md](../../arena_humansim/core/behavior/README.md) for the cross-node invariants (patience phases, seek/cancel semantics, compiler dispatch table).

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

Per-need weights used by `AutonomousNode` when a step has `autonomous: true`. Higher weight => need drives action selection more aggressively.

### `actions`

Candidate actions the autonomous selector can pick from. Fields:

| Field | Meaning |
|---|---|
| `when` | `{need: {below\|above: X}}` - preconditions gating the action. |
| `interaction` | One of `TALK_TO`, `GROUP_CONVERSATION`, `SIT_ON`, `LIE_ON`, `USE`, `QUEUE_USE`, `WAVE_AT`, `BLOCK`, `SERVICE`. Omit for nav-only. |
| `target` | Object id / object type / service tag / agent id, per the interaction's handle kind. |
| `duration` | Distribution (seconds). |
| `patience` | Distribution (seconds) capping the whole action (nav + execute). |
| `satisfies` | `{need: amount}` applied on SUCCESS. |

```yaml
actions:
  rest_at_bench:
    target: bench                # object type - resolves to nearest visible
    interaction: SIT_ON
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
        target: fountain             # object id (or type)
        interaction: USE
        duration: {mean: 6.0}
        satisfies: {thirst: 100.0}
    then: chat
```

`transitions` evaluate every tick - they can cut a step short. `then` only fires on the last step succeeding. `on_failure: <seq_name>` routes to another sequence when the current one FAILs (unset => propagate failure to the root). `interruptible: false` disables `transitions` for the sequence.

### `steps`

Each step is either a `StepDef` (interaction, pure-wait, cancel) or a `GoToStepDef` (explicit `kind: go_to`).

`StepDef` fields:

| Field | Meaning |
|---|---|
| `interaction` | `TALK_TO` / `GROUP_CONVERSATION` / `WAVE_AT` / `SIT_ON` / `LIE_ON` / `USE` / `QUEUE_USE` / `BLOCK` / `SERVICE`. Omit for a pure-wait step. |
| `target` | Interpreted per the interaction's handle kind: object id/type for `OBJECT`; service tag (str) for `SERVICE`; agent id (int) for `BLOCK`; omit for symmetric types. |
| `offer` | SERVICE provider side. `true` makes this step create-and-wait rather than find-and-join. Required when a SERVICE interaction has no existing provider. |
| `cancel` | `true` => emit STOP with `reason=CANCELED` on the agent's current interaction. Mutually exclusive with `interaction:`. |
| `queueable` | Provider-side override (SERVICE with `offer: true`) - admit seekers into a FIFO queue when full. |
| `min_participants` / `max_participants` | Provider-side overrides on the contract. Count the provider itself. |
| `formation_spec` | Provider-side formation override (`{type, anchor_kind, params}`). |
| `duration` | Distribution (seconds). With `interaction:` -> contract-level timeout (outcome `COMPLETED`). On a pure-wait step -> `HoldNode` duration (NAVIGATE-to-self). |
| `patience` | Distribution (seconds). Covers nav + seek + wait-for-ACTIVE + hold. |
| `satisfies` | `{need: amount}` applied on SUCCESS. |
| `autonomous` | `true` => `AutonomousNode` scores `actions` and runs the winner. |
| `allowed_actions` / `blocked_actions` | Filter the autonomous candidate pool. |
| `until` | Event-bus event name that exits the autonomous step on fire (e.g. `"agent_ready"`). |
| `until_need` | Need-condition predicate that exits the autonomous step (e.g. `{rest: {above: 80}}`). |
| `interruptible` | Per-step override of the sequence flag. |
| `interaction_radius` | Per-step override of the radius cascade; see [../scenarios/README.md](../scenarios/README.md). |
| `on_failure` | `"abort"` (default) / `"skip"` - on step FAILURE. |

`GoToStepDef` (`kind: go_to`) takes exactly one of:

- `target_pose: {x, y, theta}` - scripted waypoint.
- `target: <str>` - object id/type; the compiler emits `Resolve -> GoTo`.

Plus `duration:` (optional hold on arrival), `patience:`, `satisfies:`, `on_failure:`, `interruptible:`.

### Pure-wait step

Omit `interaction:` and `cancel:`; keep `duration:`. Produces `ClearOutcome -> Hold` - `HoldNode` re-emits NAVIGATE-to-self with zero velocity each tick, so the agent parks in place without tearing down any live interaction.

### Cancel step

```yaml
drop: {cancel: true}
```

Emits STOP on the agent's current `interaction_id` with `reason=CANCELED`. Falls back to force_stop (target=-1) if `interaction_id` is `None`; IM no-ops if the agent isn't in any interaction. Mutually exclusive with `interaction:`.

### Meet at an object with a symmetric interaction

Symmetric interactions (TALK_TO, GROUP_CONVERSATION) don't take a `target:`, so "meet at the gathering area" is a two-step pattern: walk first with a `go_to` step, then run the interaction.

```yaml
sequences:
  chat:
    steps:
      walk_to_gathering: {kind: go_to, target: gathering_area}
      hold_conversation:
        interaction: GROUP_CONVERSATION
        duration: {mean: 45.0, std: 15.0}
        patience: {mean: 60.0}
```

### Where this gets compiled

Per-agent trees are built in [../../arena_humansim/core/behavior/compiler.py](../../arena_humansim/core/behavior/compiler.py) from the sampled `AgentType`. To extend the authoring surface (new step fields, new primitive kinds), see [../../arena_humansim/core/behavior/nodes/README.md](../../arena_humansim/core/behavior/nodes/README.md).

## Adding a type

1. Drop a yaml file here (or under a scenario's `agent_types/` to keep it local).
2. Either `extends` an existing type or declare every field explicitly - there's no implicit default.
3. Reference it by file name (stem, without `.yaml`) from scenarios.

## Where this gets sampled

`AgentLoader.create_agent(agent_type_name, rng)` draws one concrete parameter set per agent. The RNG is the agent's seeded substream, so the same seed + type + agent_id gives the same sampled instance across runs.

## Included types

- `adult` - nominal pedestrian (desired 1.1 m/s, 5 m vision, 180deg FOV).
- `elder` - slower, narrower FOV, longer SFM relaxation. Demonstrates how heterogeneity drops out of a handful of distribution tweaks.
