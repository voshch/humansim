from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

if TYPE_CHECKING:
    from arena_humansim.agents import BaseAgent
    from arena_humansim.utils.types import HighLevelCommand, InteractionState, Pose2D

_FRAME = "map"


def rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=r, g=g, b=b, a=a)


def mk(ns: str, mid: int, mtype: int, stamp) -> Marker:
    m = Marker()
    m.header.frame_id = _FRAME
    m.header.stamp = stamp
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def arrow(ns, mid, stamp, ox, oy, dx, dy, color, shaft=0.03, head_d=0.06, head_l=0.06, z=0.1):
    m = mk(ns, mid, Marker.ARROW, stamp)
    m.points = [Point(x=ox, y=oy, z=z), Point(x=ox + dx, y=oy + dy, z=z)]
    m.scale.x, m.scale.y, m.scale.z = shaft, head_d, head_l
    m.color = color
    return m


def text(ns, mid, stamp, x, y, text, color, size=0.2, z=0.8):
    m = mk(ns, mid, Marker.TEXT_VIEW_FACING, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.scale.z = size
    m.color = color
    m.text = text
    return m


def line_strip(ns, mid, stamp, pts, color, width=0.02, z=0.05):
    m = mk(ns, mid, Marker.LINE_STRIP, stamp)
    m.scale.x = width
    m.color = color
    m.points = [Point(x=px, y=py, z=z) for px, py in pts]
    return m


def sphere(ns, mid, stamp, x, y, color, radius=0.08, z=0.1):
    m = mk(ns, mid, Marker.SPHERE, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.scale.x = m.scale.y = m.scale.z = radius * 2.0
    m.color = color
    return m


def cylinder(ns, mid, stamp, x, y, color, radius=0.5, height=0.02, z=0.0):
    m = mk(ns, mid, Marker.CYLINDER, stamp)
    m.pose.position.x, m.pose.position.y = x, y
    m.pose.position.z = z + height / 2.0
    m.scale.x = m.scale.y = radius * 2.0
    m.scale.z = height
    m.color = color
    return m


def cube(ns, mid, stamp, x, y, z, sx, sy, sz, yaw, color):
    m = mk(ns, mid, Marker.CUBE, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
    m.pose.orientation.z = math.sin(yaw / 2.0)
    m.pose.orientation.w = math.cos(yaw / 2.0)
    m.color = color
    return m


_C_CONE = rgba(0.2, 0.6, 1.0, 0.12)
_C_OBS = rgba(0.2, 0.8, 0.2, 0.4)
_C_CMD = rgba(1.0, 1.0, 1.0, 0.9)
_C_PATH = rgba(0.4, 0.9, 0.4, 0.5)
_C_IGOAL = rgba(0.1, 1.0, 0.1, 0.8)
_C_GOAL = rgba(1.0, 0.3, 0.3, 0.8)
_C_VEL = rgba(0.0, 0.7, 1.0, 0.9)
_C_ILINK = rgba(0.9, 0.9, 0.2, 0.6)
_C_ILABEL = rgba(1.0, 1.0, 0.5, 0.9)
_C_WP = rgba(0.7, 0.5, 1.0, 0.5)
_C_WP_ACT = rgba(1.0, 0.5, 1.0, 0.8)
_C_WP_RAD = rgba(0.7, 0.5, 1.0, 0.15)
_C_SRC = rgba(0.2, 1.0, 0.5, 0.4)
_C_SINK = rgba(1.0, 0.3, 0.3, 0.3)
_C_WALL = rgba(0.6, 0.0, 0.0, 0.6)
_C_WOBJ = rgba(0.3, 0.7, 1.0, 0.5)
_C_OBST = rgba(0.8, 0.5, 0.1, 0.4)
_C_OBST_LBL = rgba(1.0, 0.8, 0.3, 0.9)
_NEED_COLORS = [
    rgba(0.2, 0.8, 0.2, 0.8),
    rgba(0.8, 0.8, 0.2, 0.8),
    rgba(0.2, 0.6, 1.0, 0.8),
    rgba(0.9, 0.4, 0.1, 0.8),
    rgba(0.8, 0.2, 0.8, 0.8),
]


def _shape_outline(pose, shape) -> list[tuple[float, float]]:
    from arena_humansim.utils.types import ShapeType

    cx, cy, th = pose.x, pose.y, pose.theta
    cos_t, sin_t = math.cos(th), math.sin(th)

    if shape.type == ShapeType.CIRCLE and shape.radius > 0:
        n = 24
        pts = []
        for i in range(n + 1):
            a = 2.0 * math.pi * i / n
            pts.append((cx + shape.radius * math.cos(a), cy + shape.radius * math.sin(a)))
        return pts
    if shape.vertices:
        pts = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in shape.vertices]
        pts.append(pts[0])
        return pts
    return []


def _shape_area_marker(ns, mid, stamp, pose, shape, color, z=0.01):
    from arena_humansim.utils.types import ShapeType

    if shape.type == ShapeType.CIRCLE and shape.radius > 0:
        return cylinder(ns, mid, stamp, pose.x, pose.y, color, radius=shape.radius, height=0.01, z=z)
    if shape.vertices:
        cx, cy, th = pose.x, pose.y, pose.theta
        cos_t, sin_t = math.cos(th), math.sin(th)
        world = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in shape.vertices]
        m = mk(ns, mid, Marker.TRIANGLE_LIST, stamp)
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = color
        ox, oy = world[0]
        for i in range(1, len(world) - 1):
            m.points.append(Point(x=ox, y=oy, z=z))
            m.points.append(Point(x=world[i][0], y=world[i][1], z=z))
            m.points.append(Point(x=world[i + 1][0], y=world[i + 1][1], z=z))
        return m
    return None


_CMD_NAMES = {0: "NAV", 1: "ADV", 2: "SRCH", 3: "ACC", 4: "DEC", 5: "STOP"}
_ITYPE_NAMES = {
    0: "TALK",
    1: "GROUP",
    2: "FOLLOW",
    3: "SIT",
    4: "LIE",
    5: "USE",
    6: "QUEUE",
}


class MarkerPublisher:
    def __init__(self, node: Node):
        self._node = node
        self._pub = node.create_publisher(MarkerArray, "viz", 5)
        self._pending: list[Marker] = []

    def _stamp(self):
        return self._node.get_clock().now().to_msg()

    def flush(self):
        if self._pending:
            delete_all = Marker()
            delete_all.action = Marker.DELETEALL
            ma = MarkerArray()
            ma.markers = [delete_all] + self._pending
            self._pub.publish(ma)
            self._pending = []

    # ------------------------------------------------------------------
    # perception: vision cones + observation lines
    # ------------------------------------------------------------------
    def publish_perception(self, agents: Iterable[BaseAgent]) -> None:
        stamp = self._stamp()
        ma = MarkerArray()
        for agent in agents:
            aid = agent.state.agent_id
            p = agent.params.perception
            pose = agent.state.pose
            ma.markers.append(vision_cone(aid, stamp, pose, p.vision_range, p.vision_fov))
            if agent.belief is not None:
                for i, obs in enumerate(agent.belief.observed_agents):
                    ma.markers.append(
                        arrow(
                            "observed",
                            aid * 100 + i,
                            stamp,
                            pose.x,
                            pose.y,
                            obs.pose.x - pose.x,
                            obs.pose.y - pose.y,
                            _C_OBS,
                            shaft=0.015,
                            head_d=0.03,
                            head_l=0.03,
                        )
                    )
        self._pending.extend(ma.markers)

    # ------------------------------------------------------------------
    # behavior: command labels + need bars
    # ------------------------------------------------------------------
    def publish_behavior(
        self,
        agents: Iterable[BaseAgent],
        cmds: dict[int, HighLevelCommand],
    ) -> None:
        stamp = self._stamp()
        ma = MarkerArray()
        for agent in agents:
            aid = agent.state.agent_id
            x, y = agent.state.pose.x, agent.state.pose.y
            cmd = cmds.get(aid)
            if cmd is not None:
                ma.markers.append(
                    text(
                        "cmd",
                        aid,
                        stamp,
                        x,
                        y,
                        _CMD_NAMES.get(cmd.type, str(cmd.type)),
                        _C_CMD,
                        z=0.9,
                    )
                )
            if agent.needs is not None:
                for i, (name, need) in enumerate(agent.needs.needs.items()):
                    clr = _NEED_COLORS[i % len(_NEED_COLORS)]
                    m = mk("needs", aid * 100 + i, Marker.CUBE, stamp)
                    m.pose.position.x = x + 0.25
                    m.pose.position.y = y + 0.15 * i - 0.15
                    m.pose.position.z = 0.7
                    m.scale.x = max(0.3 * need.value / 100.0, 0.01)
                    m.scale.y = m.scale.z = 0.05
                    m.color = clr
                    ma.markers.append(m)
                    ma.markers.append(
                        text(
                            "needs_label",
                            aid * 100 + i,
                            stamp,
                            x + 0.5,
                            m.pose.position.y,
                            f"{name}:{need.value:.0f}",
                            clr,
                            size=0.1,
                            z=0.7,
                        )
                    )
        self._pending.extend(ma.markers)

    # ------------------------------------------------------------------
    # global_plan: cached paths + intermediate goals + final goal arrows
    # ------------------------------------------------------------------
    def publish_global_plan(
        self,
        agents: Sequence[BaseAgent],
        cmds: dict[int, HighLevelCommand],
        intermediate_goals: dict[int, Any],
    ) -> None:
        stamp = self._stamp()
        ma = MarkerArray()

        cached_paths: dict[int, list] = {}
        seen: set[int] = set()
        for agent in agents:
            pid = id(agent.global_planner)
            if pid not in seen:
                seen.add(pid)
                cached_paths.update(agent.global_planner.get_cached_paths())

        for agent in agents:
            aid = agent.state.agent_id

            path = cached_paths.get(aid)
            if path and len(path) > 1:
                ma.markers.append(
                    line_strip(
                        "path",
                        aid,
                        stamp,
                        [(wp.x, wp.y) for wp in path],
                        _C_PATH,
                    )
                )

            ig = intermediate_goals.get(aid)
            if ig is not None:
                gx = ig.x if hasattr(ig, "x") else float(ig[0])
                gy = ig.y if hasattr(ig, "y") else float(ig[1])
                ma.markers.append(
                    sphere(
                        "igoal",
                        aid,
                        stamp,
                        gx,
                        gy,
                        _C_IGOAL,
                        0.1,
                    )
                )

            cmd = cmds.get(aid)
            if cmd is not None:
                tp = cmd.target_pose
                ma.markers.append(
                    arrow(
                        "goal",
                        aid,
                        stamp,
                        tp.x,
                        tp.y,
                        0.3 * math.cos(tp.theta),
                        0.3 * math.sin(tp.theta),
                        _C_GOAL,
                        shaft=0.05,
                        head_d=0.1,
                        head_l=0.08,
                    )
                )
        self._pending.extend(ma.markers)

    # ------------------------------------------------------------------
    # local_plan: velocity arrows
    # ------------------------------------------------------------------
    def publish_local_plan(
        self,
        agents: Iterable[BaseAgent],
        velocities: dict[int, tuple[float, float]],
    ) -> None:
        stamp = self._stamp()
        ma = MarkerArray()
        for agent in agents:
            aid = agent.state.agent_id
            vel = velocities.get(aid)
            if vel is None:
                continue
            vx, vy = vel
            if abs(vx) < 1e-4 and abs(vy) < 1e-4:
                continue
            ma.markers.append(
                arrow(
                    "vel",
                    aid,
                    stamp,
                    agent.state.pose.x,
                    agent.state.pose.y,
                    vx,
                    vy,
                    _C_VEL,
                )
            )
        self._pending.extend(ma.markers)

    # ------------------------------------------------------------------
    # interaction: links between participants + labels
    # ------------------------------------------------------------------
    def publish_interaction(
        self,
        agents: Iterable[BaseAgent],
        interactions: dict[int, InteractionState],
    ) -> None:
        stamp = self._stamp()
        ma = MarkerArray()
        amap = {a.state.agent_id: a for a in agents}

        for iid, inter in interactions.items():
            parts = inter.participants

            if len(parts) >= 2:
                link = mk("interaction_links", iid, Marker.LINE_LIST, stamp)
                link.scale.x = 0.03
                link.color = _C_ILINK
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts)):
                        a1, a2 = amap.get(parts[i]), amap.get(parts[j])
                        if a1 and a2:
                            link.points.append(Point(x=a1.state.pose.x, y=a1.state.pose.y, z=0.3))
                            link.points.append(Point(x=a2.state.pose.x, y=a2.state.pose.y, z=0.3))
                if link.points:
                    ma.markers.append(link)

            live = [p for p in parts if p in amap]
            if live:
                cx = sum(amap[p].state.pose.x for p in live) / len(live)
                cy = sum(amap[p].state.pose.y for p in live) / len(live)
                itype = _ITYPE_NAMES.get(inter.type, str(inter.type))
                lbl = f"{itype} [{len(parts)}p"
                if inter.contract.queue:
                    lbl += f" +{len(inter.contract.queue)}q"
                lbl += "]"
                ma.markers.append(
                    text(
                        "interaction_label",
                        iid,
                        stamp,
                        cx,
                        cy,
                        lbl,
                        _C_ILABEL,
                        z=0.5,
                    )
                )
        self._pending.extend(ma.markers)

    # ------------------------------------------------------------------
    # waypoints: path + active waypoint + acceptance radius
    # ------------------------------------------------------------------
    def publish_waypoints(self, agents: Iterable[BaseAgent]) -> None:
        from arena_humansim.utils.types import WaypointMovement

        stamp = self._stamp()
        ma = MarkerArray()
        for agent in agents:
            mv = agent.movement
            if not isinstance(mv, WaypointMovement) or not mv.waypoints:
                continue
            aid = agent.state.agent_id
            wps = mv.waypoints

            if len(wps) > 1:
                ma.markers.append(
                    line_strip(
                        "wp_path",
                        aid,
                        stamp,
                        [(wp.x, wp.y) for wp in wps],
                        _C_WP,
                    )
                )
            for i, wp in enumerate(wps):
                active = i == mv.index
                ma.markers.append(
                    sphere(
                        "wp",
                        aid * 100 + i,
                        stamp,
                        wp.x,
                        wp.y,
                        _C_WP_ACT if active else _C_WP,
                        0.1 if active else 0.06,
                    )
                )
            goal = wps[mv.index]
            r = mv.radii[mv.index] if mv.radii and mv.index < len(mv.radii) else 0.3
            if r > 0:
                ma.markers.append(
                    cylinder(
                        "wp_rad",
                        aid,
                        stamp,
                        goal.x,
                        goal.y,
                        _C_WP_RAD,
                        radius=r,
                    )
                )
        self._pending.extend(ma.markers)

    # ------------------------------------------------------------------
    # infrastructure: sources, sinks, walls, world objects
    # ------------------------------------------------------------------
    def publish_infrastructure(self, sources, sinks, walls, world_objects, obstacles=None) -> None:
        stamp = self._stamp()
        ma = MarkerArray()

        for i, (name, src) in enumerate(sources.items()):
            area = _shape_area_marker("sources", i, stamp, src.pose, src.shape, _C_SRC)
            if area is not None:
                ma.markers.append(area)
            else:
                ma.markers.append(sphere("sources", i, stamp, src.pose.x, src.pose.y, _C_SRC, 0.15, z=0.0))
            ma.markers.append(
                text(
                    "src_label",
                    i,
                    stamp,
                    src.pose.x,
                    src.pose.y,
                    name,
                    _C_SRC,
                    0.15,
                    z=0.3,
                )
            )

        for i, (name, sink) in enumerate(sinks.items()):
            area = _shape_area_marker("sinks", i, stamp, sink.pose, sink.shape, _C_SINK)
            if area is not None:
                ma.markers.append(area)
            else:
                ma.markers.append(sphere("sinks", i, stamp, sink.pose.x, sink.pose.y, _C_SINK, 0.15, z=0.0))
            ma.markers.append(
                text(
                    "sink_label",
                    i,
                    stamp,
                    sink.pose.x,
                    sink.pose.y,
                    name,
                    _C_SINK,
                    0.15,
                    z=0.3,
                )
            )

        if walls:
            wm = mk("walls", 0, Marker.LINE_LIST, stamp)
            wm.scale.x = 0.05
            wm.color = _C_WALL
            for (x1, y1), (x2, y2) in walls.values():
                wm.points.append(Point(x=x1, y=y1, z=0.05))
                wm.points.append(Point(x=x2, y=y2, z=0.05))
            if wm.points:
                ma.markers.append(wm)

        for i, (oid, obj) in enumerate(world_objects.items()):
            m = mk("world_obj", i, Marker.CUBE, stamp)
            m.pose.position.x, m.pose.position.y = obj.pose.x, obj.pose.y
            m.pose.position.z = 0.15
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = _C_WOBJ
            ma.markers.append(m)
            ma.markers.append(
                text(
                    "world_obj_label",
                    i,
                    stamp,
                    obj.pose.x,
                    obj.pose.y,
                    f"{obj.type}\n{oid}",
                    _C_WOBJ,
                    0.12,
                    z=0.4,
                )
            )

        if obstacles:
            for i, (name, obs) in enumerate(obstacles.items()):
                x_min, x_max, y_min, y_max, z_min, z_max = obs.bb
                # Cube center in local frame
                cx_local = (x_min + x_max) / 2.0
                cy_local = (y_min + y_max) / 2.0
                cz = (z_min + z_max) / 2.0
                sx = max(x_max - x_min, 0.01)
                sy = max(y_max - y_min, 0.01)
                sz = max(z_max - z_min, 0.01)
                # Transform center to world frame
                cos_t = math.cos(obs.pose.theta)
                sin_t = math.sin(obs.pose.theta)
                wx = obs.pose.x + cos_t * cx_local - sin_t * cy_local
                wy = obs.pose.y + sin_t * cx_local + cos_t * cy_local

                ma.markers.append(
                    cube(
                        "obstacles",
                        i,
                        stamp,
                        wx,
                        wy,
                        cz,
                        sx,
                        sy,
                        sz,
                        obs.pose.theta,
                        _C_OBST,
                    )
                )
                label = obs.obstacle_type or name
                if obs.interaction_types:
                    label += f"\n[{', '.join(obs.interaction_types)}]"
                ma.markers.append(
                    text(
                        "obstacle_label",
                        i,
                        stamp,
                        wx,
                        wy,
                        label,
                        _C_OBST_LBL,
                        size=0.15,
                        z=cz + sz / 2.0 + 0.15,
                    )
                )

        self._pending.extend(ma.markers)

    def publish_module_markers(self, agents: Sequence[BaseAgent], modules) -> None:
        stamp = self._stamp()
        ma = MarkerArray()
        seen: set[int] = set()
        for mod in modules:
            mid = id(mod)
            if mid in seen:
                continue
            seen.add(mid)
            ma.markers.extend(mod.get_markers(agents, stamp))
        self._pending.extend(ma.markers)


def vision_cone(aid, stamp, pose, vision_range, vision_fov):
    m = mk("vision_cone", aid, Marker.TRIANGLE_LIST, stamp)
    m.scale.x = m.scale.y = m.scale.z = 1.0
    m.color = _C_CONE
    segs = 12
    half = math.radians(min(vision_fov, 360.0) * 0.5)
    h = pose.theta
    ox, oy, z = pose.x, pose.y, 0.02
    for i in range(segs):
        a0 = h - half + 2.0 * half * i / segs
        a1 = h - half + 2.0 * half * (i + 1) / segs
        m.points.append(Point(x=ox, y=oy, z=z))
        m.points.append(Point(x=ox + vision_range * math.cos(a0), y=oy + vision_range * math.sin(a0), z=z))
        m.points.append(Point(x=ox + vision_range * math.cos(a1), y=oy + vision_range * math.sin(a1), z=z))
    return m
