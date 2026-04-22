# Behavior tree nodes

Cross-node invariants enforced by the compiler live here. For the package layout and per-node authoring rules, see [nodes/README.md](nodes/README.md).

## One primitive for interactions: `SeekNode`

There are no `ADVERTISE` / `accept` phases anymore — `AdvertiseInteractionNode` and `AcceptInteractionNode` are gone, and so is `InteractionType.FOLLOW` (collapsed into `SERVICE` with `offer: true` plus a line formation anchored on the provider).

Every interaction is driven by one BT node: `SeekNode`. It emits `SEEK`; the interaction manager (via `_handle_seek` and the per-`HandleKind` strategy in [../../core/interaction_kinds.py](../../core/interaction_kinds.py)) either joins a matching open interaction or, if creation is allowed for this `InteractionKind`, creates one. `CancelNode` tears an interaction down (STOP with `reason=CANCELED`).

## Compiler dispatch

| Flags on the step | Compiled inner sequence |
|---|---|
| `cancel: true` | `ClearOutcome → Cancel` |
| `interaction:` one of `USE` / `SIT_ON` / `LIE_ON` / `QUEUE_USE` with `target:` | `ClearOutcome → Resolve → GoTo → Seek → Satisfy?` |
| `interaction:` (any other, or object-bound without `target:`) | `ClearOutcome → Seek → Satisfy?` |
| `interaction: BLOCK` with `target: <int>` | `ClearOutcome → Block` (pursuit + SEEK on arrival) |
| `kind: go_to, target_pose:` | unchanged — nav to pose |
| `kind: go_to, target:` | `ClearOutcome → Resolve → GoTo` |
| only `duration:` (pure wait) | `ClearOutcome → Hold` |

Every compiled step is wrapped in a `Parallel(SuccessOnOne)` with a `PatienceWatchdogNode` — step-level patience spans every phase below.

## Patience phases

For an interaction step, patience covers:

1. **Nav to within `interaction_radius`** (object-bound steps only — the `Resolve → GoTo` prefix).
2. **Seek emit** — `SeekNode` issues a SEEK command; IM either joins an existing interaction or creates a new one.
3. **Wait for `ACTIVE`** — `SeekNode` stays `RUNNING` until `is_bound(aid)` is true (contract reached `min_participants`).
4. **Hold** (optional) — if the step has `duration:` without an interaction, a `HoldNode` pins the agent at the current pose via NAVIGATE-to-self. With `interaction:` + `duration:`, the duration threads to the contract and IM tears the interaction down with outcome `COMPLETED`.
5. **Cancel** — either BT-controlled via a follow-up `{cancel: true}` step, or contract-timed by IM.

A step with `patience: {mean: 5.0}` and a target 20 m away may fail during phase 1 before ever emitting SEEK. Non-interaction steps are symmetric: patience caps nav + duration.

## `interaction_radius`

Cascade: step override > object override > per-`InteractionKind` default (see [../../core/interaction_kinds.py](../../core/interaction_kinds.py)) > `DISTANCE_TOLERANCE`. Full table and authoring examples in [../../../config/scenarios/README.md](../../../config/scenarios/README.md).

## Signals on `BehaviorTreeMovement`

IM writes `interaction_id` and `last_outcome`; BT nodes read them. `ClearOutcomeNode` at the head of every compiled step resets them so stale signals don't short-circuit a re-entry. Full rules and outcome encoding in [nodes/README.md](nodes/README.md#authoring-rules).

For compile-shape examples (symmetric peer, object-anchored, mobile provider, BT-controlled teardown, mid-sequence cancel), see [nodes/README.md](nodes/README.md#bt-flow-idioms).
