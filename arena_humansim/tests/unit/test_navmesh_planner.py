from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import Pose2D, Segments

_INFLATION = 0.38


def _wall_with_gap(x1: float, y1: float, x2: float, y2: float, gap_center_t: float, gap: float) -> Segments:
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    a = gap_center_t * length - gap / 2
    b = gap_center_t * length + gap / 2
    return [
        ((x1, y1), (x1 + ux * a, y1 + uy * a)),
        ((x1 + ux * b, y1 + uy * b), (x2, y2)),
    ]


def _rotate_point(p: tuple[float, float], deg: float) -> tuple[float, float]:
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    x, y = p
    return (c * x - s * y, s * x + c * y)


def _rotate(segs: Segments, deg: float) -> Segments:
    return [(_rotate_point(a, deg), _rotate_point(b, deg)) for a, b in segs]


def _two_rooms(door_width: float, rot_deg: float = 0.0) -> Segments:
    segs: Segments = [
        ((0.0, 0.0), (12.0, 0.0)),
        ((12.0, 0.0), (12.0, 6.0)),
        ((12.0, 6.0), (0.0, 6.0)),
        ((0.0, 6.0), (0.0, 0.0)),
        *_wall_with_gap(6.0, 0.0, 6.0, 6.0, 0.5, door_width),
    ]
    return _rotate(segs, rot_deg) if rot_deg else segs


def _closed_room() -> Segments:
    return [
        ((0.0, 0.0), (6.0, 0.0)),
        ((6.0, 0.0), (6.0, 6.0)),
        ((6.0, 6.0), (0.0, 6.0)),
        ((0.0, 6.0), (0.0, 0.0)),
    ]


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    l2 = np.maximum((d * d).sum(-1), 1e-12)
    t = np.clip(((p - a) * d).sum(-1) / l2, 0.0, 1.0)
    c = a + t[..., None] * d
    return np.hypot(p[..., 0] - c[..., 0], p[..., 1] - c[..., 1])


def _sample_polyline(poly: np.ndarray, step: float = 0.01) -> np.ndarray:
    pts = [poly[0:1]]
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        length = float(np.hypot(*(b - a)))
        n = max(int(length / step), 1)
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        pts.append(a[None, :] + t * (b - a))
    return np.vstack(pts)


def _path_wall_clearance(poly: np.ndarray, walls: np.ndarray) -> float:
    samples = _sample_polyline(poly)
    a, b = walls[:, :2], walls[:, 2:]
    d = _point_seg_dist(samples[:, None, :], a[None, :, :], b[None, :, :])
    return float(d.min())


def _crossing_y(poly: np.ndarray, x: float) -> float:
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:], strict=True):
        if (x0 - x) * (x1 - x) <= 0 and x0 != x1:
            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    raise AssertionError("path does not cross x")


def _crosses(
    planner: NavMeshPlanner,
    start: tuple[float, float],
    goal: tuple[float, float],
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> bool:
    agent = agent_factory(agent_id=1, x=start[0], y=start[1])
    cmds = commands_factory(agent_ids=[1], target=goal)
    planner.compute([agent], cmds)
    end = planner.get_cached_paths()[1][-1]
    return math.hypot(end.x - goal[0], end.y - goal[1]) < 1e-6


@pytest.mark.parametrize("rot_deg", [0.0, 30.0])
def test_door_just_under_double_inflation_has_no_path(
    rot_deg: float,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(_two_rooms(0.74, rot_deg))
    start, goal = _rotate_point((2.0, 3.0), rot_deg), _rotate_point((10.0, 3.0), rot_deg)
    assert not _crosses(planner, start, goal, agent_factory, commands_factory)


@pytest.mark.parametrize("rot_deg", [0.0, 30.0])
def test_door_just_over_double_inflation_has_path(
    rot_deg: float,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(_two_rooms(0.78, rot_deg))
    start, goal = _rotate_point((2.0, 3.0), rot_deg), _rotate_point((10.0, 3.0), rot_deg)
    assert _crosses(planner, start, goal, agent_factory, commands_factory)


def test_metre_door_crossed_where_grid_planners_close(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(_two_rooms(1.0))
    assert _crosses(planner, (2.0, 3.0), (10.0, 3.0), agent_factory, commands_factory)


def test_cached_paths_keep_hard_clearance(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    segs = _two_rooms(0.8)
    planner = NavMeshPlanner(inflation_radius=_INFLATION, comfort_radius=0.0)
    planner.set_walls(segs)
    starts = [(2.0, 1.0), (2.0, 5.0), (1.0, 3.0)]
    agents = [agent_factory(agent_id=i, x=x, y=y) for i, (x, y) in enumerate(starts)]
    cmds = commands_factory(agent_ids=list(range(len(starts))), target=(10.0, 3.0))
    planner.compute(agents, cmds)
    paths = planner.get_cached_paths()
    assert len(paths) == len(starts)
    walls = np.array([[a[0], a[1], b[0], b[1]] for a, b in segs])
    for wps in paths.values():
        poly = np.array([(w.x, w.y) for w in wps])
        assert _path_wall_clearance(poly, walls) >= _INFLATION - 0.006


def test_comfort_radius_centres_door_crossing(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    segs = _two_rooms(1.0)
    door_x, door_y = 6.0, 3.0
    start, goal = (1.0, 1.0), (11.0, 2.0)

    tight = NavMeshPlanner(inflation_radius=_INFLATION, comfort_radius=0.0)
    tight.set_walls(segs)
    tight.compute([agent_factory(agent_id=1, x=start[0], y=start[1])], commands_factory(agent_ids=[1], target=goal))
    tight_poly = np.array([(w.x, w.y) for w in tight.get_cached_paths()[1]])
    tight_y = _crossing_y(tight_poly, door_x)

    comfy = NavMeshPlanner(inflation_radius=_INFLATION, comfort_radius=0.6)
    comfy.set_walls(segs)
    comfy.compute([agent_factory(agent_id=1, x=start[0], y=start[1])], commands_factory(agent_ids=[1], target=goal))
    comfy_poly = np.array([(w.x, w.y) for w in comfy.get_cached_paths()[1]])
    comfy_y = _crossing_y(comfy_poly, door_x)

    assert abs(comfy_y - door_y) < abs(tight_y - door_y)
    assert abs(comfy_y - door_y) <= 0.1


def test_configure_rebuilds_from_current_walls(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=0.45)
    planner.set_walls(_two_rooms(0.8))
    assert not _crosses(planner, (2.0, 3.0), (10.0, 3.0), agent_factory, commands_factory)
    planner.configure(inflation_radius=0.38, resolution=0.2, comfort_radius=0.6)
    assert _crosses(planner, (2.0, 3.0), (10.0, 3.0), agent_factory, commands_factory)


def test_configure_resolution_only_keeps_cached_paths(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6)
    planner.set_walls(_two_rooms(1.0))
    agent = agent_factory(agent_id=1, x=2.0, y=3.0)
    cmds = commands_factory(agent_ids=[1], target=(10.0, 3.0))
    planner.compute([agent], cmds)
    before = planner.get_cached_paths()
    assert 1 in before
    planner.configure(inflation_radius=0.38, resolution=0.4, comfort_radius=0.6)
    assert planner.get_cached_paths() == before


def test_unreachable_target_routes_to_closest_reachable_point(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(_closed_room())
    agent = agent_factory(agent_id=1, x=3.0, y=3.0)
    cmds = commands_factory(agent_ids=[1], target=(8.0, 3.0))
    planner.compute([agent], cmds)
    path = planner.get_cached_paths()[1]
    assert path[-1].x == pytest.approx(6.0 - _INFLATION, abs=0.02)
    assert path[-1].y == pytest.approx(3.0, abs=0.02)
    walls = np.array(_closed_room(), dtype=np.float64).reshape(-1, 4)
    assert _path_wall_clearance(np.array([(w.x, w.y) for w in path]), walls) >= _INFLATION - 5e-3
    planner.compute([agent], cmds)
    assert planner.get_cached_paths()[1] is path


def test_walled_in_agent_keeps_direct_goal(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    cell = [((2.8, 2.8), (3.2, 2.8)), ((3.2, 2.8), (3.2, 3.2)), ((3.2, 3.2), (2.8, 3.2)), ((2.8, 3.2), (2.8, 2.8))]
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls([*_closed_room(), *cell])
    agent = agent_factory(agent_id=1, x=3.0, y=3.0)
    cmds = commands_factory(agent_ids=[1], target=(5.0, 5.0))
    goals = planner.compute([agent], cmds)
    assert 1 not in planner.get_cached_paths()
    assert goals[1] == planner.snap_terminal(cmds[1].target_pose)


def test_shared_goal_computed_together_and_cached(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(_two_rooms(1.0))
    starts = [(2.0, 1.0), (2.0, 5.0), (1.0, 3.0)]
    agents = [agent_factory(agent_id=i, x=x, y=y) for i, (x, y) in enumerate(starts)]
    cmds = commands_factory(agent_ids=list(range(len(starts))), target=(10.0, 3.0))
    first = planner.compute(agents, cmds)
    assert set(first) == {0, 1, 2}
    assert all(i in planner.get_cached_paths() for i in range(3))
    second = planner.compute(agents, cmds)
    assert second == first


def test_set_walls_empty_leaves_no_map(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(_two_rooms(1.0))
    planner.set_walls([])
    agent = agent_factory(agent_id=1, x=2.0, y=3.0)
    cmds = commands_factory(agent_ids=[1], target=(9.5, 4.5))
    goals = planner.compute([agent], cmds)
    assert goals[1].x == pytest.approx(9.5)
    assert goals[1].y == pytest.approx(4.5)


def test_snap_terminal_moves_point_to_inflation_clearance() -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls([((0.0, -3.0), (0.0, 3.0))])
    pose = Pose2D(x=0.1, y=0.0, theta=0.7)
    out = planner.snap_terminal(pose)
    assert abs(out.x) >= _INFLATION - 1e-6
    assert out.theta == pytest.approx(pose.theta)


def test_snap_terminal_identity_when_already_clear() -> None:
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls([((0.0, -3.0), (0.0, 3.0))])
    pose = Pose2D(x=5.0, y=0.0, theta=0.3)
    out = planner.snap_terminal(pose)
    assert out is pose


def test_closing_a_door_blocks_it_without_meshing_again(
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    rooms = _two_rooms(1.0)
    door = ((6.0, 2.5), (6.0, 3.5))
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(rooms)
    mesh = planner._mesh
    assert _crosses(planner, (2.0, 3.0), (10.0, 3.0), agent_factory, commands_factory)
    planner.set_walls([*rooms, door])
    assert planner._mesh is mesh
    assert not _crosses(planner, (2.0, 3.0), (10.0, 3.0), agent_factory, commands_factory)
    walls = np.array([*rooms, door], dtype=np.float64).reshape(-1, 4)
    path = np.array([(w.x, w.y) for w in planner.get_cached_paths()[1]])
    assert _path_wall_clearance(path, walls) >= _INFLATION - 5e-3
    planner.set_walls(rooms)
    assert planner._mesh is mesh
    assert _crosses(planner, (2.0, 3.0), (10.0, 3.0), agent_factory, commands_factory)


def test_wall_off_the_mesh_edges_meshes_again() -> None:
    rooms = _two_rooms(1.0)
    planner = NavMeshPlanner(inflation_radius=_INFLATION)
    planner.set_walls(rooms)
    mesh = planner._mesh
    planner.set_walls([*rooms, ((2.0, 1.0), (4.0, 5.0))])
    assert planner._mesh is not mesh
