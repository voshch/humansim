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

## Adding a type

1. Drop a yaml file here (or under a scenario's `agent_types/` to keep it local).
2. Either `extends` an existing type or declare every field explicitly — there's no implicit default.
3. Reference it by file name (stem, without `.yaml`) from scenarios.

## Where this gets sampled

`AgentLoader.create_agent(agent_type_name, rng)` draws one concrete parameter set per agent. The RNG is the agent's seeded substream, so the same seed + type + agent_id gives the same sampled instance across runs.

## Included types

- `adult` — nominal pedestrian (desired 1.1 m/s, 5 m vision, 180° FOV).
- `elder` — slower, narrower FOV, longer SFM relaxation. Demonstrates how heterogeneity drops out of a handful of distribution tweaks.
