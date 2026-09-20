from __future__ import annotations

import numpy as np
import pytest
from arena_humansim.utils.evaluation.correspondence import HAUSDORFF_PASS_M, SCENARIOS, VEL_PASS_MPS, compare, run_ours, run_rvo2, run_straight_reference


def test_straight_matches_closed_form() -> None:
    for make in SCENARIOS.values():
        starts, goals = make()
        hd, dv = compare(run_ours("straight", starts, goals), run_straight_reference(starts, goals))
        assert max(hd) < 1e-6
        assert dv < 1e-6


def test_orca_tracks_rvo2() -> None:
    pytest.importorskip("rvo2")
    medians = []
    errors = []
    for make in SCENARIOS.values():
        starts, goals = make()
        hd, dv = compare(run_ours("orca", starts, goals), run_rvo2(starts, goals))
        medians.append(float(np.median(hd)))
        errors.append(dv)
    assert float(np.median(medians)) < HAUSDORFF_PASS_M, medians
    assert float(np.mean(errors)) < VEL_PASS_MPS, errors


def test_orca_is_exact_in_sparse_interaction() -> None:
    pytest.importorskip("rvo2")
    starts, goals = SCENARIOS["sparse_random_10"]()
    hd, _ = compare(run_ours("orca", starts, goals), run_rvo2(starts, goals))
    assert float(np.median(hd)) < 0.05
