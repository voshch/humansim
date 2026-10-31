from __future__ import annotations

import math

import numpy as np
import pytest

from arena_humansim.utils.funnel import Corner, _beyond, _reach, _tangent, funnel, polyline

Point = tuple[float, float]


def _taut(s: Point, g: Point, portals: list[tuple[Point, Point]], pinned: tuple[int, ...] = (), rho: float = 0.0) -> list:
    events = [p for portal in portals for p in portal]
    return funnel(
        np.array(s),
        np.array(g),
        [-1, 1] * len(portals),
        list(range(len(events))),
        [p[0] for p in events],
        [p[1] for p in events],
        [e in pinned for e in range(len(events))],
        [rho] * len(events),
    )


def _xy(corners: list) -> list[Point]:
    return [(c[0], c[1]) for c in corners]


def test_tangent_of_two_points_is_their_direction() -> None:
    assert _tangent((0.0, 0.0, 0.0, 0.0), (3.0, 4.0, 0.0, 0.0)) == pytest.approx((0.6, 0.8))


@pytest.mark.parametrize("sigma", [-1.0, 1.0])
def test_tangent_touches_the_disc_of_its_end_corner(sigma: float) -> None:
    u = _tangent((0.0, 0.0, 0.0, 0.0), (2.0, 0.0, sigma, 1.0))
    assert math.hypot(*u) == pytest.approx(1.0)
    assert abs(u[0] * 0.0 - u[1] * 2.0) == pytest.approx(1.0)


def test_tangent_of_coincident_corners_falls_back_to_the_x_axis() -> None:
    assert _tangent((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)) == (1.0, 0.0)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0.0, 0.0, 0.0, 0.0), (2.0, 0.0, 1.0, 1.0), math.sqrt(3.0)),
        ((0.0, 0.0, 1.0, 0.5), (2.0, 0.0, 1.0, 0.5), 2.0),
        ((0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 1.0, 1.0), 0.0),
    ],
    ids=["point_to_disc", "same_side_discs", "inside_the_disc"],
)
def test_reach_is_the_tangent_leg_length(a: Corner, b: Corner, expected: float) -> None:
    assert _reach(a, b) == pytest.approx(expected)


def test_beyond_compares_projections_on_the_leg() -> None:
    origin = (0.0, 0.0, 0.0, 0.0)
    goal = (2.0, 0.0, 0.0, 0.0)
    assert _beyond((3.0, 1.0, 0.0, 0.0), origin, (1.0, 0.0), goal)
    assert not _beyond((1.0, 1.0, 0.0, 0.0), origin, (1.0, 0.0), goal)


def test_channel_without_portals_is_the_straight_line() -> None:
    assert _xy(_taut((0.0, 0.0), (5.0, 1.0), [])) == [(0.0, 0.0), (5.0, 1.0)]


def test_wide_portal_leaves_the_line_straight() -> None:
    assert _xy(_taut((0.0, 0.0), (2.0, 0.0), [((1.0, 1.0), (1.0, -1.0))])) == [(0.0, 0.0), (2.0, 0.0)]


@pytest.mark.parametrize(
    ("portal", "goal", "corner"),
    [
        (((1.0, 3.0), (1.0, 1.0)), (2.0, 1.0), (1.0, 1.0)),
        (((1.0, -1.0), (1.0, -3.0)), (2.0, -1.0), (1.0, -1.0)),
    ],
    ids=["right_vertex", "left_vertex"],
)
def test_goal_outside_the_portal_turns_at_its_near_vertex(portal: tuple[Point, Point], goal: Point, corner: Point) -> None:
    assert _xy(_taut((0.0, 0.0), goal, [portal])) == [(0.0, 0.0), corner, goal]


def test_consecutive_corners_come_out_in_channel_order() -> None:
    portals = [((1.0, 3.0), (1.0, 1.0)), ((2.0, 3.0), (2.0, 1.0))]
    assert _xy(_taut((0.0, 0.0), (3.0, 0.0), portals)) == [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)]


def test_forced_event_becomes_a_corner_of_a_straight_channel() -> None:
    portals = [((1.0, 1.0), (1.0, -1.0))]
    assert _xy(_taut((0.0, 0.0), (2.0, 0.0), portals, pinned=(1,))) == [(0.0, 0.0), (1.0, -1.0), (2.0, 0.0)]


def test_corner_keeps_its_side_and_radius() -> None:
    corner = _taut((0.0, 0.0), (2.0, 1.0), [((1.0, 3.0), (1.0, 1.0))], rho=0.2)[1]
    assert corner[:4] == (1.0, 1.0, 1.0, 0.2)


def test_polyline_of_a_straight_leg_is_its_end_points() -> None:
    assert polyline([(0.0, 0.0, 0.0, 0.0), (2.0, 1.0, 0.0, 0.0)]).tolist() == [[0.0, 0.0], [2.0, 1.0]]


def test_polyline_arc_stays_within_one_percent_of_the_corner_radius() -> None:
    r = 0.2
    poly = polyline([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, r), (2.0, 1.0, 0.0, 0.0)])
    assert poly[0].tolist() == [0.0, 0.0]
    assert poly[-1].tolist() == [2.0, 1.0]
    assert len(poly) == 5
    dist = np.hypot(*(poly[1:-1] - np.array([1.0, 1.0])).T)
    assert np.all(dist >= r)
    assert np.all(dist <= 1.01 * r)


def test_strict_polyline_refuses_a_turn_against_the_corner_side() -> None:
    corners: list = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, -1.0, 0.2), (2.0, 1.0, 0.0, 0.0)]
    assert polyline(corners, strict=True) is None
    assert len(polyline(corners)) == 3


def test_strict_polyline_accepts_a_small_turn_against_the_corner_side() -> None:
    corners: list = [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, -1.0, 0.0), (2.0, -0.05, 0.0, 0.0)]
    poly = polyline(corners, strict=True)
    assert poly is not None
    assert poly.tolist() == [[0.0, 0.0], [1.0, 0.0], [2.0, -0.05]]


def test_strict_polyline_refuses_overlapping_discs() -> None:
    assert polyline([(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 1.0, 0.5), (2.0, 0.0, 0.0, 0.0)], strict=True) is None
