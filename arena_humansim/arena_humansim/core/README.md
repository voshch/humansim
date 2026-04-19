# core

The simulator's central subsystems live here: the agent pool, the tick-scheduling `AgentManager`, behavior-tree compilation, interaction routing, and shared access/formation primitives. Most of the state that survives across ticks is owned by something in this directory.

This README focuses on **`InteractionManager`** — the only piece of core with non-obvious behavior that isn't already covered by a sibling readme. See also:

- [behavior/README.md](behavior/README.md) — behavior-tree node invariants (ADVERTISE, patience, SERVICE accept).

## InteractionManager

Pure router over `ADVERTISE` / `STOP` / `IDLE` commands. Discovery — "who should I pair with?" — is delegated to perception, not reimplemented here.

## `_try_bind` rules

For every unbound `_Advertisement`, the matcher tries each rule in order. First match wins.

1. **Explicit interaction.** `ad.interaction_target >= 0` → join that specific interaction if it exists and is still open.
2. **Object-anchored.** `ad.object_id is not None` → join the existing interaction for that (object, type) pair, or create one with the ad's agent as creator.
3. **Targeted agent.** `ad.target_agent >= 0` → find an ad from that specific agent of the same type and pair.
4. **Open / anchorless.** Pair with the closest *visible* open ad of the same type. Visibility comes from perception (range + FOV); distance is only used as tiebreak among visible candidates. No global `DISCOVERY_RADIUS` — each agent's `vision_range` and `vision_fov` shape the gate.

Rule 4 fails silently when no visibility callback is wired, so tests that don't care about pairing can skip the setup.

## `visibility_lookup` contract

```python
visibility_lookup: Callable[[int], set[int]]
```

Given an agent_id, return the set of agent_ids that agent currently perceives. The interaction manager calls this from `_find_visible_open_ad` during rule 4.

In the main loop it's backed by the pool's neighbor CSR — see `pool.visible_agent_ids` and `DefaultPerception.compute_pool`. In tests it's a simple dict lookup.

## SERVICE interaction type

`InteractionType.SERVICE` is special-cased by the matcher. Before rules 1–4, `_bind_ad` dispatches SERVICE ads to `_bind_service`.

- **Pairing criteria.** Same `service_tag` (string-equal) and mutual visibility via the same `visibility_lookup` used by rule 4.
- **Anchor rule.** If exactly one peer in the pair is ROBOT-kind, that peer is the initiator and occupies index 0 of `participants`. This centers `f_formation` on the robot.
- **Parallel interactions.** Two robots offering the same tag yield two independent interactions, one per robot. The store already indexes multiple ads per agent.
- **`max_participants` threading.** An ad's `max_participants` overrides the contract default (`-1` = unbounded). The robot ad wins over a human seeker ad when both set it. Counts every participant including the robot itself — set `N+1` for "N humans at once".
- **Queueing.** Contract default is `queueable=True` with `FIFOQueue()`; overflow past the cap queues in arrival order.

Defaults: formation `f_formation`, `interaction_radius` 3.0.

## What lives elsewhere

- **Proximity gating for anchored `ADVERTISE`** — the compiler emits `ResolveObjectNode` → `GoToNode` → `AdvertiseInteractionNode` for interaction steps; `GoToNode` only returns SUCCESS once within `interaction_radius`, so `AdvertiseInteractionNode` never fires before the agent has arrived. IM trusts the ad.
- **Formation defaults per interaction type** — `DEFAULT_FORMATION_BY_INTERACTION` (same file), keyed by `InteractionType`. Overridden by `WorldObject.formation` when present.
- **Approach-tolerance defaults per interaction type** — `DEFAULT_INTERACTION_RADIUS` (same file). See [config/scenarios/README.md](../../config/scenarios/README.md) for the full cascade.
