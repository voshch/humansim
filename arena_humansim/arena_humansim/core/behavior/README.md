# Behavior tree nodes

Cross-node invariants enforced by the compiler live here. For the package layout and per-node authoring rules, see [nodes/README.md](nodes/README.md).

## Nav-before-advertise invariant

For a step with `interaction:` and a resolvable `target_object_*`, the compiler emits a `Sequence` of primitives: `ResolveObjectNode` → `GoToNode` → `AdvertiseInteractionNode`. `GoToNode` emits `NAVIGATE` toward the target pose while the agent is farther than `interaction_radius`; once `_at_target` holds it returns SUCCESS, and only then does `AdvertiseInteractionNode` run and emit `ADVERTISE`.

For an accept step (`accept: true`, no `target_object_*`) there is no resolve / nav phase — `AcceptInteractionNode` runs immediately and perception-visibility in the interaction manager gates pairing.

The invariant enforced by the anchored chain: *an agent joins `interaction.participants` / `interaction.contract.queue` only when it was within `interaction_radius` of the target object at the moment of advertise.* The interaction manager trusts inbound `ADVERTISE` commands.

See [config/scenarios/README.md](../../../../config/scenarios/README.md) for the `interaction_radius` cascade (step > object > type default > `DISTANCE_TOLERANCE`).

## `patience`

Spans the whole step via `PatienceWatchdogNode` running in the step's outer `Parallel`. On an interaction step, patience covers:
1. Nav to within `interaction_radius` (may return FAILURE if it times out).
2. Advertise.
3. Wait for `COMPLETED` / `INTERRUPTED` outcome.

A step with `patience: 5.0` and a target 20 m away may fail during phase 1 before ever advertising. This matches the symmetric behavior of non-interaction steps, which also use patience to cap nav + duration.

## Accept steps

`accept: true` routes through `_expand_accept_step`, which emits a bare `AcceptInteractionNode` (no resolve, no nav). Use it for passive roles — vendors, greeters, responders — that wait to be paired. `service_tag` is forwarded on the ad so the matcher's `_bind_service` branch can match by tag. `interaction` is required; `target_object_*` is rejected by the scenario validator.

```yaml
sequences:
  default:
    steps:
      sell_water:
        accept: true
        interaction: SERVICE
        service_tag: water
        patience: { mean: 30.0 }
```
