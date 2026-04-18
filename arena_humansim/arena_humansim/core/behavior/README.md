# Behavior tree nodes

## `ConcreteStepNode` — nav-before-advertise invariant

For a step with `interaction:` and a resolvable `target_object_*`:

- The node emits `NAVIGATE` toward the target pose while the agent is farther than `interaction_radius`.
- Only when `_at_target(agent, target_pose, interaction_radius)` holds does the node emit `ADVERTISE`.
- For an anchorless step (no `target_object_*`) the node emits `ADVERTISE` immediately — perception-visibility in the interaction manager gates pairing.

The invariant enforced by this node: *an agent joins `interaction.participants` / `interaction.contract.queue` only when it was within `interaction_radius` of the target object at the moment of advertise.* The interaction manager trusts inbound `ADVERTISE` commands.

See [config/scenarios/README.md](../../../../config/scenarios/README.md) for the `interaction_radius` cascade (step > object > type default > `DISTANCE_TOLERANCE`).

## `patience`

Spans the whole step. On an interaction step, that's:
1. Nav to within `interaction_radius` (may return FAILURE if it times out).
2. Advertise.
3. Wait for `COMPLETED` / `INTERRUPTED` outcome.

A step with `patience: 5.0` and a target 20 m away may fail during phase 1 before ever advertising. This matches the symmetric behavior of non-interaction steps, which also use patience to cap nav + duration.

## `AcceptServiceNode`

Parks the agent until the matcher pairs it into a SERVICE interaction matching `service_tag`.

- Emits an `ADVERTISE` command with `interaction_type=SERVICE`, `service_tag=<tag>`, `target_agent=from_agent.agent_id if set else -1`.
- Lifecycle mirrors `AcceptNode`: RUNNING while waiting, SUCCESS on `COMPLETED`, FAILURE on `INTERRUPTED`.
- Autonomous need-gating: wrap in a `ConditionNode` that reads `BeliefState.extra` (e.g. `thirst > 0.5`) so the accept only runs when the need is active.

```yaml
- sequence:
    - condition: { expr: "extra.thirst > 0.5" }
    - accept_service: { service_tag: water }
```
