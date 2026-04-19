from __future__ import annotations

from pathlib import Path

import pytest

from arena_humansim.core.agents.types import GoToStepDef, StepDef
from arena_humansim.utils.scenario import load_scenario

_SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "config" / "scenarios"
_SCENARIO_PATHS = sorted(_SCENARIOS_DIR.glob("*.yaml"))


def _hold_node_step(step: StepDef | GoToStepDef) -> bool:
    if isinstance(step, GoToStepDef):
        return False
    if step.autonomous:
        return False
    if step.interaction is not None:
        return False
    return step.duration is not None or step.target_object_type is not None or step.target_object_id is not None


@pytest.mark.parametrize("scenario_path", _SCENARIO_PATHS, ids=lambda p: p.name)
def test_hold_node_steps_never_carry_interaction(scenario_path: Path) -> None:
    scenario = load_scenario(str(scenario_path))
    for atype_name, atype in scenario.agent_types.items():
        for seq_name, seq in atype.sequences.items():
            for step_name, step in seq.steps.items():
                if not _hold_node_step(step):
                    continue
                assert isinstance(step, StepDef)
                assert step.interaction is None, (
                    f"{scenario_path.name}:{atype_name}.{seq_name}.{step_name} compiles to a HoldNode "
                    f"but carries interaction={step.interaction!r}; a HoldNode STOP emission would race formation binding"
                )


def test_formation_only_binds_via_interaction_advertisement() -> None:
    source_path = Path(__file__).resolve().parents[2] / "arena_humansim" / "core" / "interaction_manager.py"
    text = source_path.read_text()

    # Only ADVERTISE and STOP branches exist in the command dispatcher; STOP leaves formations, never joins.
    assert "if ctype == CommandType.ADVERTISE" in text
    assert "self._post_ad(cmd)" in text
    assert "elif ctype == CommandType.STOP" in text

    # Formation join side-effect is gated through accept()/_bind_ad, both fed from _post_ad (ADVERTISE path).
    call_sites = [line for line in text.splitlines() if "_on_formation_join(" in line and "def " not in line]
    assert call_sites, "expected at least one _on_formation_join call site"

    # Sanity: STOP handler does not call _on_formation_join.
    stop_idx = text.index("elif ctype == CommandType.STOP")
    match_idx = text.index("def _match_ads", stop_idx)
    stop_block = text[stop_idx:match_idx]
    assert "_on_formation_join" not in stop_block
