from __future__ import annotations

import numpy as np

from arena_humansim.occlusion.noop import NoopOccluder


def _occ() -> NoopOccluder:
    return NoopOccluder()


def test_always_returns_all_true_with_no_walls() -> None:
    occ = _occ()
    occ.set_walls([])
    p_a = np.array([[0.0, 0.0], [1.0, 2.0]])
    p_b = np.array([[5.0, 0.0], [-1.0, 3.0]])
    np.testing.assert_array_equal(occ.clear(p_a, p_b), np.ones(2, dtype=np.bool_))


def test_always_returns_all_true_despite_walls() -> None:
    # NoopOccluder ignores wall geometry entirely.
    occ = _occ()
    occ.set_walls([((0.0, -10.0), (0.0, 10.0))])
    p_a = np.array([[-3.0, 0.0], [0.5, 0.5]])
    p_b = np.array([[3.0, 0.0], [-0.5, 0.5]])
    np.testing.assert_array_equal(occ.clear(p_a, p_b), np.ones(2, dtype=np.bool_))


def test_repeated_set_walls_does_not_change_behavior() -> None:
    occ = _occ()
    walls = [((x, -1.0), (x, 1.0)) for x in range(10)]
    occ.set_walls(walls)
    occ.set_walls([])
    occ.set_walls(walls)
    p_a = np.array([[-5.0, 0.0]])
    p_b = np.array([[5.0, 0.0]])
    np.testing.assert_array_equal(occ.clear(p_a, p_b), np.array([True]))


def test_empty_rays_returns_empty() -> None:
    occ = _occ()
    result = occ.clear(np.empty((0, 2)), np.empty((0, 2)))
    assert result.shape == (0,)
    assert result.dtype == np.bool_
