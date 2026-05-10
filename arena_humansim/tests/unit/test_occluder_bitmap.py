from __future__ import annotations

import numpy as np
import pytest

from arena_humansim.occlusion.bitmap import BitmapOccluder


def _occ() -> BitmapOccluder:
    return BitmapOccluder()


def test_empty_walls_returns_all_true() -> None:
    occ = _occ()
    occ.set_walls([])
    p_a = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, -3.0]])
    p_b = np.array([[1.0, 0.0], [-1.0, 2.0], [10.0, 7.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.ones(3, dtype=np.bool_))


def test_wall_between_two_points_blocks_ray() -> None:
    # Vertical wall at x=0 from y=-1 to y=1 blocks ray from (-2, 0) to (2, 0).
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    p_a = np.array([[-2.0, 0.0]])
    p_b = np.array([[2.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([False]))


def test_wall_offset_to_side_does_not_block() -> None:
    # Vertical wall at x=5 does not intersect the ray from (0,0) to (3,0).
    occ = _occ()
    occ.set_walls([((5.0, -1.0), (5.0, 1.0))])
    p_a = np.array([[0.0, 0.0]])
    p_b = np.array([[3.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_mixed_rays_some_blocked_some_clear() -> None:
    # Horizontal wall at y=0 from x=-1 to x=1.
    occ = _occ()
    occ.set_walls([((-1.0, 0.0), (1.0, 0.0))])
    # ray 0: crosses the wall (from y=-1 to y=1, x=0) -> blocked
    # ray 1: does not cross (from y=2 to y=3, x=0) -> clear
    # ray 2: lateral, wall at y=0 between x=-1..1, ray goes from (3,0) to (5,0); endpoint on wall row is borderline; use a safe side
    # ray 1: fully above, guaranteed clear
    p_a = np.array([[0.0, -1.0], [0.0, 2.0]])
    p_b = np.array([[0.0, 1.0], [0.0, 3.0]])
    result = occ.clear(p_a, p_b)
    assert result[0] == False  # noqa: E712
    assert result[1] == True  # noqa: E712


def test_clear_single_point_pair_no_division_by_zero() -> None:
    # Both endpoints identical - zero-length ray; must return True (no wall hit).
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    p_a = np.array([[0.0, 0.0]])
    p_b = np.array([[0.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_set_walls_empty_after_non_empty_resets_to_pass_through() -> None:
    occ = _occ()
    occ.set_walls([((0.0, -1.0), (0.0, 1.0))])
    p_a = np.array([[-2.0, 0.0]])
    p_b = np.array([[2.0, 0.0]])
    assert occ.clear(p_a, p_b)[0] == False  # noqa: E712

    occ.set_walls([])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_wall_perpendicular_ray_not_blocked() -> None:
    # Horizontal wall at y=5; ray travels horizontally at y=0; should not block.
    occ = _occ()
    occ.set_walls([((-10.0, 5.0), (10.0, 5.0))])
    p_a = np.array([[-3.0, 0.0]])
    p_b = np.array([[3.0, 0.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([True]))


def test_diagonal_wall_blocks_crossing_ray() -> None:
    # 45deg wall from (-1,-1) to (1,1) blocks a ray from (-1,1) to (1,-1).
    occ = _occ()
    occ.set_walls([((-1.0, -1.0), (1.0, 1.0))])
    p_a = np.array([[-1.0, 1.0]])
    p_b = np.array([[1.0, -1.0]])
    result = occ.clear(p_a, p_b)
    np.testing.assert_array_equal(result, np.array([False]))


def test_empty_input_returns_empty_array() -> None:
    occ = _occ()
    occ.set_walls([((0.0, 0.0), (0.0, 1.0))])
    p_a = np.empty((0, 2))
    p_b = np.empty((0, 2))
    result = occ.clear(p_a, p_b)
    assert result.shape == (0,)
