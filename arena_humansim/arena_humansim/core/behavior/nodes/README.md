# nodes

Composable `py_trees.Behaviour` primitives the compiler assembles into per-agent trees. See [../README.md](../README.md) for the cross-node invariants (nav-before-advertise, patience semantics, accept steps).

## Layout

| Module | Contents |
|---|---|
| `helpers.py` | `_nav_command`, `_interaction_command`, `_at_target`, `_resolve_interaction_radius`, `_sample_param_dist`, `_bt_logger`. |
| `utility.py` | Utility scoring: `check_condition`, `preconditions_met`, `score_actions`. |
| `primitives.py` | Leaf nodes that don't resolve world state: `ClearOutcomeNode`, `PatienceWatchdogNode`, `HoldNode`, `SatisfyNode`, `NeedsDecayNode`. |
| `navigation.py` | `ResolveObjectNode`, `GoToNode` — target resolution and approach. |
| `interaction.py` | `AdvertiseInteractionNode` (object-anchored ADVERTISE) and `AcceptInteractionNode` (bare ADVERTISE for accept steps). Both wait on interaction outcome. |
| `autonomous.py` | `AutonomousNode` — utility-based action selection inside a step. |
| `state_machine.py` | `SequenceStateMachine` — runs one compiled sequence with `transitions` / `then` / `on_failure` edges. |

Public API is re-exported from `__init__.py`. Import `from arena_humansim.core.behavior.nodes import ...`; submodules are internal and can be reorganized freely.

## Authoring rules

- **One agent per node.** Never emit a `HighLevelCommand` for anyone but `self._agent`.
- **Re-emit nav every tick.** Formations overwrite `movement.command`, so `GoToNode` and `HoldNode` re-emit unconditionally. Interaction nodes are the exception: advertise once and let IM hold the slot.
- **STOP on exit.** If a node takes over `movement.command`, its `terminate()` must restore STOP — IM only releases the participant slot on STOP. `AdvertiseInteractionNode.terminate` fires on SUCCESS too, for this reason.
- **Composite teardown.** Use `stop(INVALID)` to cascade `terminate()` into children; `terminate()` alone is a leaf-only hook. See `SequenceStateMachine._goto`.
- **Outcome round-trip.** Interaction nodes read `BehaviorTreeMovement.last_outcome` (COMPLETED → SUCCESS, INTERRUPTED → FAILURE) and clear it. Put a `ClearOutcomeNode` at the head of any sequence that may re-enter, so a stale outcome doesn't short-circuit the next run.
- **Watchdogs never succeed.** `PatienceWatchdogNode` returns only RUNNING or FAILURE. It sits in a `Parallel(SuccessOnOne)` where the sibling sequence signals success.
- **Context over args.** Multi-node step chains (resolve → go-to → advertise) thread `StepContext` rather than passing resolved poses through constructor args — initialisation order in `Sequence` isn't guaranteed to match authoring order.

## Adding a node

1. Pick the module whose concern matches; add a new module and import from `__init__.py` if none fits.
2. Subclass `py_trees.behaviour.Behaviour`. Keep state under `self._` attributes and reset in `initialise()` — nodes are constructed once and re-ticked.
3. If the node emits a `HighLevelCommand`, pick once-vs-every-tick per the rules above and implement `terminate()` to STOP.
4. Wire it into [../compiler.py](../compiler.py) — most primitives are constructed inside the step-builder helpers.
5. Cover it in [tests/unit/test_node_primitives.py](../../../../tests/unit/test_node_primitives.py) or `test_behavior_nodes.py`.
