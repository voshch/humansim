from __future__ import annotations

import math
import queue
import threading
from typing import TYPE_CHECKING

from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

if TYPE_CHECKING:
    from arena_humansim.utils.types import Pose2D

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


class MarkerView:
    """Scoped handle to a marker namespace."""

    def __init__(self, pub: MarkerPublisher, ns: str, mtype: int, dirty: bool):
        self._pub = pub
        self._ns = ns
        self._mtype = mtype
        self._dirty = dirty

    def get(self, mid: int) -> tuple[Marker, bool]:
        return self._pub.get(self._ns, mid, self._mtype, self._dirty)

    def clear(self):
        """Remove all markers in this namespace from the scene.
        DELETE markers are emitted at next flush via stale detection."""
        to_remove = [k for k in self._pub._scene if k[0] == self._ns]
        for k in to_remove:
            del self._pub._scene[k]
            self._pub._dirty.discard(k)


class MarkerPublisher:
    def __init__(self, node: Node):
        self._node = node
        self._pub = node.create_publisher(MarkerArray, "viz", 5)
        self._scene: dict[tuple[str, int], Marker] = {}
        self._pool: dict[str, list[Marker]] = {}
        self._touched: set[tuple[str, int]] = set()
        self._dirty: set[tuple[str, int]] = set()
        self._ns_count: dict[str, int] = {}
        self._queue: queue.SimpleQueue[list[MarkerArray] | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def _publish_loop(self):
        while True:
            batch = self._queue.get()
            if batch is None:
                break
            while not self._queue.empty():
                next_batch = self._queue.get()
                if next_batch is None:
                    return
                batch = next_batch
            for ma in batch:
                self._pub.publish(ma)

    def _stamp(self):
        return self._node.get_clock().now().to_msg()

    def view(self, ns: str, mtype: int, count: int = 0, dirty: bool = True) -> MarkerView:
        have = self._ns_count.get(ns, 0)
        if count > have:
            pool = self._pool.setdefault(ns, [])
            for _ in range(count - have):
                pool.append(mk(ns, 0, mtype, None))
            self._ns_count[ns] = count
        return MarkerView(self, ns, mtype, dirty)

    def get(self, ns: str, mid: int, mtype: int, dirty: bool = True) -> tuple[Marker, bool]:
        key = (ns, mid)
        self._touched.add(key)
        if key in self._scene:
            if dirty:
                self._dirty.add(key)
            return self._scene[key], False
        pool = self._pool.get(ns)
        if pool:
            m = pool.pop()
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
        else:
            m = mk(ns, mid, mtype, None)
        self._scene[key] = m
        self._dirty.add(key)
        return m, True

    def flush(self):
        stamp = self._stamp()
        touched_ns = {ns for ns, _ in self._touched}
        stale = [(ns, mid) for ns, mid in self._scene if ns in touched_ns and (ns, mid) not in self._touched]

        deletes: list[Marker] = []
        for ns, mid in stale:
            del self._scene[(ns, mid)]
            self._dirty.discard((ns, mid))
            m = Marker()
            m.header.frame_id = _FRAME
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            deletes.append(m)

        adds: list[Marker] = []
        for key in self._dirty:
            m = self._scene[key]
            m.header.stamp = stamp
            adds.append(m)

        batch: list[MarkerArray] = []
        if deletes:
            ma = MarkerArray()
            ma.markers = deletes
            batch.append(ma)
        if adds:
            ma = MarkerArray()
            ma.markers = adds
            batch.append(ma)

        if batch:
            self._queue.put(batch)

        self._touched.clear()
        self._dirty.clear()

    def forget_all(self):
        m = Marker()
        m.action = Marker.DELETEALL
        ma = MarkerArray()
        ma.markers = [m]
        self._queue.put([ma])
        self._scene.clear()
        self._pool.clear()
        self._ns_count.clear()
        self._touched.clear()
        self._dirty.clear()

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=1.0)


def publish_behavior(pub: MarkerPublisher, agents, cmds) -> None:
    cmd_view = pub.view("cmd", Marker.TEXT_VIEW_FACING)
    for agent in agents:
        aid = agent.state.agent_id
        x, y = agent.state.pose.x, agent.state.pose.y
        cmd = cmds.get(aid)
        if cmd is not None:
            m, new = cmd_view.get(aid)
            if new:
                m.color = _C_CMD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.9
            m.scale.z = 0.2
            m.text = _CMD_NAMES.get(cmd.type, str(cmd.type))
        if agent.needs is not None:
            for i, (name, need) in enumerate(agent.needs.needs.items()):
                clr = _NEED_COLORS[i % len(_NEED_COLORS)]
                bar_ns = f"needs_{name}"
                bar_view = pub.view(bar_ns, Marker.CUBE)
                m, new = bar_view.get(aid)
                if new:
                    m.scale.y = m.scale.z = 0.05
                    m.color = clr
                m.pose.position.x = x + 0.25
                m.pose.position.y = y + 0.15 * i - 0.15
                m.pose.position.z = 0.7
                m.scale.x = max(0.3 * need.value / 100.0, 0.01)

                lbl_ns = f"needs_{name}_label"
                lbl_view = pub.view(lbl_ns, Marker.TEXT_VIEW_FACING)
                m, new = lbl_view.get(aid)
                if new:
                    m.color = clr
                    m.scale.z = 0.1
                m.pose.position.x = x + 0.5
                m.pose.position.y = y + 0.15 * i - 0.15
                m.pose.position.z = 0.7
                m.text = f"{name}:{need.value:.0f}"


def publish_interaction(pub: MarkerPublisher, agents, interactions) -> None:
    amap = {a.state.agent_id: a for a in agents}
    links_view = pub.view("interaction_links", Marker.LINE_LIST)
    label_view = pub.view("interaction_label", Marker.TEXT_VIEW_FACING)
    for iid, inter in interactions.items():
        parts = inter.participants
        if len(parts) >= 2:
            m, new = links_view.get(iid)
            if new:
                m.scale.x = 0.03
                m.color = _C_ILINK
            m.points.clear()
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    a1, a2 = amap.get(parts[i]), amap.get(parts[j])
                    if a1 and a2:
                        m.points.append(Point(x=a1.state.pose.x, y=a1.state.pose.y, z=0.3))
                        m.points.append(Point(x=a2.state.pose.x, y=a2.state.pose.y, z=0.3))
        live = [p for p in parts if p in amap]
        if live:
            cx = sum(amap[p].state.pose.x for p in live) / len(live)
            cy = sum(amap[p].state.pose.y for p in live) / len(live)
            itype = _ITYPE_NAMES.get(inter.type, str(inter.type))
            lbl = f"{itype} [{len(parts)}p"
            if inter.contract.queue:
                lbl += f" +{len(inter.contract.queue)}q"
            lbl += "]"
            m, new = label_view.get(iid)
            if new:
                m.color = _C_ILABEL
                m.scale.z = 0.2
            m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.5
            m.text = lbl


def publish_infrastructure(pub: MarkerPublisher, sources, sinks, walls, world_objects, obstacles=None) -> None:
    for i, (name, src) in enumerate(sources.items()):
        from arena_humansim.utils.types import ShapeType
        if src.shape.type == ShapeType.CIRCLE and src.shape.radius > 0:
            m, new = pub.get("sources", i, Marker.CYLINDER, dirty=False)
            if new:
                m.pose.position.x, m.pose.position.y = src.pose.x, src.pose.y
                m.pose.position.z = 0.01 + 0.01 / 2.0
                m.scale.x = m.scale.y = src.shape.radius * 2.0
                m.scale.z = 0.01
                m.color = _C_SRC
        elif src.shape.vertices:
            m, new = pub.get("sources", i, Marker.TRIANGLE_LIST, dirty=False)
            if new:
                cx, cy, th = src.pose.x, src.pose.y, src.pose.theta
                cos_t, sin_t = math.cos(th), math.sin(th)
                world = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in src.shape.vertices]
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.color = _C_SRC
                ox, oy = world[0]
                for vi in range(1, len(world) - 1):
                    m.points.append(Point(x=ox, y=oy, z=0.01))
                    m.points.append(Point(x=world[vi][0], y=world[vi][1], z=0.01))
                    m.points.append(Point(x=world[vi + 1][0], y=world[vi + 1][1], z=0.01))
        else:
            m, new = pub.get("sources", i, Marker.SPHERE, dirty=False)
            if new:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = src.pose.x, src.pose.y, 0.0
                m.scale.x = m.scale.y = m.scale.z = 0.15 * 2.0
                m.color = _C_SRC
        m, new = pub.get("src_label", i, Marker.TEXT_VIEW_FACING, dirty=False)
        if new:
            m.pose.position.x, m.pose.position.y, m.pose.position.z = src.pose.x, src.pose.y, 0.3
            m.scale.z = 0.15
            m.color = _C_SRC
            m.text = name

    for i, (name, sink) in enumerate(sinks.items()):
        from arena_humansim.utils.types import ShapeType
        if sink.shape.type == ShapeType.CIRCLE and sink.shape.radius > 0:
            m, new = pub.get("sinks", i, Marker.CYLINDER, dirty=False)
            if new:
                m.pose.position.x, m.pose.position.y = sink.pose.x, sink.pose.y
                m.pose.position.z = 0.01 + 0.01 / 2.0
                m.scale.x = m.scale.y = sink.shape.radius * 2.0
                m.scale.z = 0.01
                m.color = _C_SINK
        elif sink.shape.vertices:
            m, new = pub.get("sinks", i, Marker.TRIANGLE_LIST, dirty=False)
            if new:
                cx, cy, th = sink.pose.x, sink.pose.y, sink.pose.theta
                cos_t, sin_t = math.cos(th), math.sin(th)
                world = [(cx + cos_t * v.x - sin_t * v.y, cy + sin_t * v.x + cos_t * v.y) for v in sink.shape.vertices]
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.color = _C_SINK
                ox, oy = world[0]
                for vi in range(1, len(world) - 1):
                    m.points.append(Point(x=ox, y=oy, z=0.01))
                    m.points.append(Point(x=world[vi][0], y=world[vi][1], z=0.01))
                    m.points.append(Point(x=world[vi + 1][0], y=world[vi + 1][1], z=0.01))
        else:
            m, new = pub.get("sinks", i, Marker.SPHERE, dirty=False)
            if new:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = sink.pose.x, sink.pose.y, 0.0
                m.scale.x = m.scale.y = m.scale.z = 0.15 * 2.0
                m.color = _C_SINK
        m, new = pub.get("sink_label", i, Marker.TEXT_VIEW_FACING, dirty=False)
        if new:
            m.pose.position.x, m.pose.position.y, m.pose.position.z = sink.pose.x, sink.pose.y, 0.3
            m.scale.z = 0.15
            m.color = _C_SINK
            m.text = name

    if walls:
        m, new = pub.get("walls", 0, Marker.LINE_LIST, dirty=False)
        if new:
            m.scale.x = 0.05
            m.color = _C_WALL
            for (x1, y1), (x2, y2) in walls.values():
                m.points.append(Point(x=x1, y=y1, z=0.05))
                m.points.append(Point(x=x2, y=y2, z=0.05))

    for i, (oid, obj) in enumerate(world_objects.items()):
        m, new = pub.get("world_obj", i, Marker.CUBE, dirty=False)
        if new:
            m.pose.position.x, m.pose.position.y = obj.pose.x, obj.pose.y
            m.pose.position.z = 0.15
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = _C_WOBJ
        m, new = pub.get("world_obj_label", i, Marker.TEXT_VIEW_FACING, dirty=False)
        if new:
            m.pose.position.x, m.pose.position.y = obj.pose.x, obj.pose.y
            m.pose.position.z = 0.4
            m.scale.z = 0.12
            m.color = _C_WOBJ
            m.text = f"{obj.type}\n{oid}"

    if obstacles:
        for i, (name, obs) in enumerate(obstacles.items()):
            x_min, x_max, y_min, y_max, z_min, z_max = obs.bb
            cx_local = (x_min + x_max) / 2.0
            cy_local = (y_min + y_max) / 2.0
            cz = (z_min + z_max) / 2.0
            sx = max(x_max - x_min, 0.01)
            sy = max(y_max - y_min, 0.01)
            sz = max(z_max - z_min, 0.01)
            cos_t = math.cos(obs.pose.theta)
            sin_t = math.sin(obs.pose.theta)
            wx = obs.pose.x + cos_t * cx_local - sin_t * cy_local
            wy = obs.pose.y + sin_t * cx_local + cos_t * cy_local
            m, new = pub.get("obstacles", i, Marker.CUBE, dirty=False)
            if new:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = wx, wy, cz
                m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
                m.pose.orientation.z = math.sin(obs.pose.theta / 2.0)
                m.pose.orientation.w = math.cos(obs.pose.theta / 2.0)
                m.color = _C_OBST
            label = obs.obstacle_type or name
            if obs.interaction_types:
                label += f"\n[{', '.join(obs.interaction_types)}]"
            m, new = pub.get("obstacle_label", i, Marker.TEXT_VIEW_FACING, dirty=False)
            if new:
                m.pose.position.x, m.pose.position.y = wx, wy
                m.pose.position.z = cz + sz / 2.0 + 0.15
                m.scale.z = 0.15
                m.color = _C_OBST_LBL
                m.text = label


def publish_perception(pub: MarkerPublisher, agents) -> None:
    cone_view = pub.view("vision_cone", Marker.TRIANGLE_LIST)
    obs_view = pub.view("observed", Marker.ARROW)
    for agent in agents:
        aid = agent.state.agent_id
        p = agent.params.perception
        pose = agent.state.pose
        m, new = cone_view.get(aid)
        if new:
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = _C_CONE
        m.points.clear()
        segs = 12
        half = math.radians(min(p.vision_fov, 360.0) * 0.5)
        h = pose.theta
        ox, oy, z = pose.x, pose.y, 0.02
        for si in range(segs):
            a0 = h - half + 2.0 * half * si / segs
            a1 = h - half + 2.0 * half * (si + 1) / segs
            m.points.append(Point(x=ox, y=oy, z=z))
            m.points.append(Point(x=ox + p.vision_range * math.cos(a0), y=oy + p.vision_range * math.sin(a0), z=z))
            m.points.append(Point(x=ox + p.vision_range * math.cos(a1), y=oy + p.vision_range * math.sin(a1), z=z))
        if agent.belief is not None:
            for i, obs in enumerate(agent.belief.observed_agents):
                m, new = obs_view.get(aid * 100 + i)
                if new:
                    m.scale.x, m.scale.y, m.scale.z = 0.015, 0.03, 0.03
                    m.color = _C_OBS
                    m.points = [Point(), Point()]
                m.points[0].x, m.points[0].y, m.points[0].z = pose.x, pose.y, 0.1
                m.points[1].x = obs.pose.x
                m.points[1].y = obs.pose.y
                m.points[1].z = 0.1


def publish_global_plan(pub: MarkerPublisher, agents, cmds, intermediate_goals: dict[int, Pose2D]) -> None:
    path_view = pub.view("path", Marker.LINE_STRIP)
    igoal_view = pub.view("igoal", Marker.SPHERE)
    goal_view = pub.view("goal", Marker.ARROW)

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
            m, new = path_view.get(aid)
            if new:
                m.scale.x = 0.02
                m.color = _C_PATH
            m.points = [Point(x=wp.x, y=wp.y, z=0.05) for wp in path]

        ig = intermediate_goals.get(aid)
        if ig is not None:
            gx = ig.x
            gy = ig.y
            m, new = igoal_view.get(aid)
            if new:
                m.scale.x = m.scale.y = m.scale.z = 0.1 * 2.0
                m.color = _C_IGOAL
            m.pose.position.x, m.pose.position.y, m.pose.position.z = gx, gy, 0.1

        cmd = cmds.get(aid)
        if cmd is not None:
            tp = cmd.target_pose
            m, new = goal_view.get(aid)
            if new:
                m.scale.x, m.scale.y, m.scale.z = 0.05, 0.1, 0.08
                m.color = _C_GOAL
                m.points = [Point(), Point()]
            m.points[0].x, m.points[0].y, m.points[0].z = tp.x, tp.y, 0.1
            m.points[1].x = tp.x + 0.3 * math.cos(tp.theta)
            m.points[1].y = tp.y + 0.3 * math.sin(tp.theta)
            m.points[1].z = 0.1


def publish_local_plan(pub: MarkerPublisher, agents, velocities) -> None:
    vel_view = pub.view("vel", Marker.ARROW)
    for agent in agents:
        aid = agent.state.agent_id
        vel = velocities.get(aid)
        if vel is None:
            continue
        vx, vy = vel
        if abs(vx) < 1e-4 and abs(vy) < 1e-4:
            continue
        m, new = vel_view.get(aid)
        if new:
            m.scale.x, m.scale.y, m.scale.z = 0.03, 0.06, 0.06
            m.color = _C_VEL
            m.points = [Point(), Point()]
        m.points[0].x, m.points[0].y, m.points[0].z = agent.state.pose.x, agent.state.pose.y, 0.1
        m.points[1].x = agent.state.pose.x + vx
        m.points[1].y = agent.state.pose.y + vy
        m.points[1].z = 0.1


def publish_waypoints(pub: MarkerPublisher, agents) -> None:
    from arena_humansim.utils.types import WaypointMovement

    wp_path_view = pub.view("wp_path", Marker.LINE_STRIP)
    wp_view = pub.view("wp", Marker.SPHERE)
    wp_rad_view = pub.view("wp_rad", Marker.CYLINDER)
    for agent in agents:
        mv = agent.movement
        if not isinstance(mv, WaypointMovement) or not mv.waypoints:
            continue
        aid = agent.state.agent_id
        wps = mv.waypoints
        if len(wps) > 1:
            m, new = wp_path_view.get(aid)
            if new:
                m.scale.x = 0.02
                m.color = _C_WP
            m.points = [Point(x=wp.x, y=wp.y, z=0.05) for wp in wps]
        for i, wp in enumerate(wps):
            active = i == mv.index
            m, new = wp_view.get(aid * 100 + i)
            if new:
                pass
            m.color = _C_WP_ACT if active else _C_WP
            radius = 0.1 if active else 0.06
            m.scale.x = m.scale.y = m.scale.z = radius * 2.0
            m.pose.position.x, m.pose.position.y, m.pose.position.z = wp.x, wp.y, 0.1
        goal = wps[mv.index]
        r = mv.radii[mv.index] if mv.radii and mv.index < len(mv.radii) else 0.3
        if r > 0:
            m, new = wp_rad_view.get(aid)
            if new:
                m.color = _C_WP_RAD
            m.pose.position.x, m.pose.position.y = goal.x, goal.y
            m.pose.position.z = 0.0 + 0.02 / 2.0
            m.scale.x = m.scale.y = r * 2.0
            m.scale.z = 0.02


def publish_module_markers(pub: MarkerPublisher, modules) -> None:
    seen: set[int] = set()
    for mod in modules:
        mid = id(mod)
        if mid in seen:
            continue
        seen.add(mid)
        mod.publish_markers(pub)


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
