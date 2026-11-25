from __future__ import annotations

import enum
import math
import operator
from collections.abc import Sequence

import attrs
import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segment

WAIT_RADIUS = 1.0
HOLD_RADIUS = 0.3
SLOT_SPACING = 0.7
SLOT_MARGIN = 0.35
STAIR_SPEED = 0.7
LANDING_BACKOFF = 0.1
SEAM_PROBE = 0.05
THROUGH_CLEAR = 1.0
LANE_MARGIN = 0.5


def _inside(pts: tuple[tuple[float, float], ...], x: float, y: float) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(pts, pts[-1:] + pts[:-1], strict=True):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def _slots(zone: Zone) -> tuple[Pose2D, ...]:
    """Spots on a grid inside the footprint, the one farthest from the entry first, the hold pose when none fits."""
    if not zone.footprint:
        return (zone.hold,)
    xs, ys = [x for x, _ in zone.footprint], [y for _, y in zone.footprint]
    axes = []
    for lo, hi in ((min(xs) + SLOT_MARGIN, max(xs) - SLOT_MARGIN), (min(ys) + SLOT_MARGIN, max(ys) - SLOT_MARGIN)):
        n = max(1, math.floor(max(hi - lo, 0.0) / SLOT_SPACING) + 1)
        mid = (lo + hi) / 2.0
        axes.append([mid + (i - (n - 1) / 2.0) * SLOT_SPACING for i in range(n)])
    spots = [Pose2D(x=x, y=y, theta=zone.hold.theta) for x in axes[0] for y in axes[1] if _inside(zone.footprint, x, y)]
    spots.sort(key=lambda p: -math.hypot(p.x - zone.entry.x, p.y - zone.entry.y))
    return tuple(spots) or (zone.hold,)


def _walk_time(path: list[Pose2D] | None, speed: float) -> float:
    """Time to walk a planned path, inf without one."""
    if not path:
        return math.inf
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:], strict=False)) / speed


def _center(zone: Zone) -> tuple[float, float]:
    """Footprint area centroid, the vertex mean for a degenerate footprint, the hold pose without one."""
    pts = zone.footprint
    if not pts:
        return zone.hold.x, zone.hold.y
    area = cx = cy = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1], strict=True):
        cross = ax * by - bx * ay
        area += cross
        cx += (ax + bx) * cross
        cy += (ay + by) * cross
    if abs(area) < 1e-9:
        return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    return cx / (3.0 * area), cy / (3.0 * area)


def _lane(zone: Zone) -> Pose2D:
    """Point on the hold line halfway to the right side and LANE_MARGIN off it, the hold pose on a half too narrow for that."""
    ax, ay = zone.axis
    right = min((-(x - zone.hold.x) * ay + (y - zone.hold.y) * ax for x, y in zone.footprint), default=0.0)
    across = max(right / 2.0, right + LANE_MARGIN)
    if across >= 0.0:
        return zone.hold
    return Pose2D(x=zone.hold.x - ay * across, y=zone.hold.y + ax * across, theta=zone.hold.theta)


@attrs.frozen
class Zone:
    name: str
    destinations: tuple[str, ...]
    entry: Pose2D
    wait: Pose2D
    hold: Pose2D
    footprint: tuple[tuple[float, float], ...] = ()
    delay: float = 0.0
    kind: str = ""
    belt: float = 0.0
    slots: tuple[Pose2D, ...] = attrs.field(init=False, default=attrs.Factory(_slots, takes_self=True))
    center: tuple[float, float] = attrs.field(init=False, default=attrs.Factory(_center, takes_self=True))
    lane: Pose2D = attrs.field(init=False, default=attrs.Factory(_lane, takes_self=True))

    @property
    def axis(self) -> tuple[float, float]:
        return math.cos(self.entry.theta), math.sin(self.entry.theta)

    def along(self, x: float, y: float) -> float:
        """Distance of x, y from the entry along the zone's facing."""
        ax, ay = self.axis
        return (x - self.entry.x) * ax + (y - self.entry.y) * ay

    def across(self, x: float, y: float) -> float:
        """Distance of x, y to the left of the line through the entry along the zone's facing."""
        ax, ay = self.axis
        return -(x - self.entry.x) * ay + (y - self.entry.y) * ax

    def point(self, along: float, across: float = 0.0) -> Pose2D:
        """The pose `along` meters from the entry and `across` meters to the left of its line, facing as the entry."""
        ax, ay = self.axis
        return Pose2D(x=self.entry.x + ax * along - ay * across, y=self.entry.y + ay * along + ax * across, theta=self.entry.theta)

    @property
    def length(self) -> float:
        """Distance from the entry to the far edge of the footprint, the seam of a stair half."""
        return max(self.along(x, y) for x, y in self.footprint)

    def contains(self, x: float, y: float) -> bool:
        """Whether a pedestrian at x, y stands inside the zone."""
        if self.kind:
            return _inside(self.footprint, x, y)
        if math.hypot(x - self.hold.x, y - self.hold.y) <= HOLD_RADIUS:
            return True
        return _inside(self.footprint, x, y)

    def past_seam(self, x: float, y: float) -> bool:
        """Whether a pedestrian at x, y stands on this stair half at or beyond its seam trigger line."""
        return self.contains(x, y) and self.along(x, y) >= self.along(self.hold.x, self.hold.y)


class Phase(enum.IntEnum):
    WALK = 0
    WAIT = 1
    BOARD = 2
    HOLD = 3


@attrs.define
class Transfer:
    goal: tuple[float, float]
    zone: str
    destination: str
    phase: Phase = Phase.WALK
    since: int = 0
    slot: int = -1
    held: int = 0
    yielded: int = 0
    ride: bool = False
    aboard: bool = False
    side: float = 0.0


@attrs.frozen
class _Graph:
    area: dict[int, int]
    edges: dict[int, list[tuple[str, str, int]]]
    sides: dict[str, frozenset[int]]


class TeleportZones:
    """Teleport zones, their access and the pedestrians transferring through them."""

    def __init__(self) -> None:
        self.zones: dict[str, Zone] = {}
        self.open: dict[str, bool] = {}
        self.transfers: dict[int, Transfer] = {}
        self.riders: set[int] = set()
        self._banned: dict[int, tuple[tuple[float, float], list[str]]] = {}
        self._held: set[int] = set()
        self._walks: dict[int, tuple[tuple[float, float], frozenset[str]]] = {}
        self._joined: dict[str, str] = {}
        self.gates: list[Segment] = []
        self.through: dict[int, Pose2D] = {}
        self._landed: dict[int, str] = {}

    def set_zones(self, zones: Sequence[Zone], is_open: Sequence[bool]) -> None:
        """Replace every zone, keeping the transfers through zones that come back unchanged or stairs that keep their destination."""
        old = self.zones
        self.zones = {z.name: z for z in zones}
        self.open = {z.name: bool(o) for z, o in zip(zones, is_open, strict=True)}
        kept = {name for name, z in self.zones.items() if old.get(name) == z}
        for aid, transfer in list(self.transfers.items()):
            zone = self.zones.get(transfer.zone)
            stair = zone is not None and bool(zone.kind) and transfer.zone in old and bool(old[transfer.zone].kind) and transfer.destination in zone.destinations
            if transfer.zone not in kept and not stair:
                del self.transfers[aid]
        for aid, (goal, names) in list(self._banned.items()):
            names = [name for name in names if name in kept]
            if names:
                self._banned[aid] = (goal, names)
            else:
                del self._banned[aid]
        self.riders.clear()
        self._held.clear()
        self._walks.clear()
        self._join()

    def _join(self) -> None:
        """Find the stair halves that meet their partner in place and the seam of each pair."""
        self._joined = {}
        gates: dict[frozenset[str], Segment] = {}
        for name, zone in self.zones.items():
            if not zone.kind:
                continue
            beyond = zone.point(zone.length + SEAM_PROBE)
            partner = next((other for other in self.zones.values() if other.kind and (other.name in zone.destinations or name in other.destinations) and other.contains(beyond.x, beyond.y)), None)
            if partner is None:
                continue
            self._joined[name] = partner.name
            sides = [zone.across(x, y) for x, y in zone.footprint]
            right, left = zone.point(zone.length, min(sides)), zone.point(zone.length, max(sides))
            gates.setdefault(frozenset((name, partner.name)), ((right.x, right.y), (left.x, left.y)))
        self.gates = list(gates.values())

    def clear_transfers(self) -> None:
        self.transfers.clear()
        self._banned.clear()
        self._walks.clear()
        self.riders.clear()
        self._held.clear()
        self._landed.clear()

    def forget(self, aid: int) -> None:
        self.transfers.pop(aid, None)
        self._banned.pop(aid, None)
        self._walks.pop(aid, None)
        self._held.discard(aid)
        self.through.pop(aid, None)

    def steer(self, agents: Sequence[BaseAgent], cmds: dict[int, HighLevelCommand]) -> dict[int, HighLevelCommand]:
        """Commands with each transferring pedestrian's target swapped for its cabin's wait pose or spot inside, or its stair's lane point on the hold line, short of the seam on a pair joined in place. `through` holds the subgoals that bypass the planners."""
        self.through = {}
        if not self.zones:
            return cmds
        out = dict(cmds)
        groups: dict[GlobalPlanner, list[tuple[BaseAgent, HighLevelCommand]]] = {}
        for agent in agents:
            cmd = cmds.get(agent.state.agent_id)
            if cmd is not None and cmd.type == CommandType.NAVIGATE and agent.state.kind != KIND_ROBOT:
                groups.setdefault(agent.global_planner, []).append((agent, cmd))
        blocked: dict[tuple[str, ...], set[str]] = {}
        for planner, group in groups.items():
            graph = self._graph(planner)
            comps = planner.components([agent.state.pose for agent, _ in group] + [cmd.target_pose for _, cmd in group])
            plans = planner.get_cached_paths() if self._joined else {}
            for k, (agent, cmd) in enumerate(group):
                aid = agent.state.agent_id
                near = (agent.state.pose.x, agent.state.pose.y)
                goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
                transfer = self.transfers.get(aid)
                if transfer is not None and transfer.goal != goal:
                    transfer = None
                    del self.transfers[aid]
                banned = self._banned.get(aid)
                if banned is not None and banned[0] != goal:
                    banned = None
                    del self._banned[aid]
                climbs = agent.params.climbs
                if climbs not in blocked:
                    blocked[climbs] = self._blocked(climbs)
                if transfer is not None and self._joined.get(transfer.zone) == transfer.destination:
                    zone = self.zones[transfer.zone]
                    on = zone.contains(*near) or self.zones[transfer.destination].contains(*near)
                    if transfer.aboard and not on and zone.along(*near) > zone.length:
                        del self.transfers[aid]
                        transfer = None
                    else:
                        transfer.aboard = on
                if transfer is not None and transfer.ride and not self._aboard(transfer, *near) and (transfer.zone in blocked[climbs] or (comps[k] and not comps[k] & graph.sides[transfer.zone])):
                    del self.transfers[aid]
                    transfer = None
                if transfer is None or (transfer.phase == Phase.WALK and not transfer.ride and not self._aboard(transfer, *near)):
                    start, end = comps[k], comps[len(group) + k]
                    keep = None if transfer is None else (transfer.zone, transfer.destination)
                    edge = self._route(graph, start, end, blocked[climbs] | set(() if banned is None else banned[1]), near, keep)
                    if edge is None and banned is not None and len(banned[1]) > 1:
                        self._banned[aid] = (goal, banned[1][-1:])
                        edge = self._route(graph, start, end, blocked[climbs] | set(banned[1][-1:]), near, keep)
                    if edge is None and banned is not None:
                        del self._banned[aid]
                        edge = self._route(graph, start, end, blocked[climbs], near, keep)
                    ride = False
                    walkable = bool({graph.area.get(c, c) for c in start} & {graph.area.get(c, c) for c in end})
                    if (edge is None and walkable) or (edge is not None and edge[0] in self._joined):
                        timed = self._ride(planner, agent, cmd.target_pose, goal, blocked[climbs] | set(self._banned[aid][1] if aid in self._banned else ()))
                        if timed is not None:
                            edge, ride = timed, True
                    if edge is None:
                        self.transfers.pop(aid, None)
                        plan = plans.get(aid)
                        if plan and any((name in blocked[climbs] or not self.zones[name].destinations) and self.zones[name].contains(plan[-1].x, plan[-1].y) and not self.zones[name].contains(*goal) for name in self._joined):
                            self.through[aid] = agent.state.pose
                        continue
                    if transfer is None or (transfer.zone, transfer.destination) != edge:
                        transfer = self.transfers[aid] = Transfer(goal=goal, zone=edge[0], destination=edge[1], ride=ride, side=self.zones[edge[0]].across(*near))
                z = self.zones[transfer.zone]
                if z.kind:
                    out[aid] = attrs.evolve(cmd, target_pose=self._inner(z, transfer.side))
                    if transfer.aboard:
                        self.through[aid] = z.point(z.length + self.zones[transfer.destination].length + THROUGH_CLEAR, z.across(*near))
                    continue
                out[aid] = attrs.evolve(cmd, target_pose=z.wait if transfer.phase in (Phase.WALK, Phase.WAIT) else z.slots[transfer.slot])
        return out

    def _inner(self, zone: Zone, across: float = 0.0) -> Pose2D:
        """Where a stair half is walked to: its lane point on the hold line, on a pair joined in place a point short of the seam `across` meters off its center line."""
        if zone.name not in self._joined:
            return zone.lane
        sides = [zone.across(x, y) for x, y in zone.footprint]
        right, left = min(sides) + LANE_MARGIN, max(sides) - LANE_MARGIN
        across = min(max(across, right), left) if right <= left else (right + left) / 2.0
        return attrs.evolve(zone.point(max(zone.length - THROUGH_CLEAR, zone.length / 2.0), across), theta=zone.hold.theta)

    def _ride(self, planner: GlobalPlanner, agent: BaseAgent, target: Pose2D, goal: tuple[float, float], blocked: set[str]) -> tuple[str, str] | None:
        """Zone and destination of the first stair on the quickest way to a goal, None when walking is quicker or no stair leads there."""
        aid = agent.state.agent_id
        key = (goal, frozenset(blocked))
        if self._walks.get(aid) == key:
            return None
        rides = [(name, dest) for name, zone in sorted(self.zones.items()) if name in self._joined and name not in blocked for dest in zone.destinations if dest in self.zones]
        if not rides:
            self._walks[aid] = key
            return None
        near = agent.state.pose
        speed = max(agent.params.desired_velocity, 0.1)
        walk = _walk_time(planner.paths([(near, target)])[0], speed)
        ride = [self._ride_time(name, dest, speed, agent.params.stair_effort) for name, dest in rides]
        reach = [math.hypot(self.zones[name].entry.x - near.x, self.zones[name].entry.y - near.y) / speed for name, _ in rides]
        rest = [math.hypot(target.x - self.zones[dest].entry.x, target.y - self.zones[dest].entry.y) / speed for _, dest in rides]
        first, last = min(map(operator.add, reach, ride)), min(map(operator.add, ride, rest))
        kept = [i for i in range(len(rides)) if min(reach[i], first) + ride[i] + min(rest[i], last) < walk]
        rides, ride = [rides[i] for i in kept], [ride[i] for i in kept]
        if rides:
            mouths = [self.zones[name].entry for name, _ in rides]
            lands = [self.zones[dest].entry for _, dest in rides]
            walks = [_walk_time(path, speed) for path in planner.paths([*((near, mouth) for mouth in mouths), *((land, target) for land in lands)])]
            board, rest = walks[: len(rides)], walks[len(rides) :]
            comps = planner.components(lands + mouths)
            legs = [(i, j) for i in range(len(rides)) if math.isinf(rest[i]) for j in range(len(rides)) if comps[i] & comps[len(rides) + j]]
            onward = dict(zip(legs, (_walk_time(path, speed) for path in planner.paths([(lands[i], mouths[j]) for i, j in legs])), strict=True))
            for _ in rides:
                for (i, j), hop in onward.items():
                    rest[i] = min(rest[i], hop + ride[j] + rest[j])
            times = [board[i] + ride[i] + rest[i] for i in range(len(rides))]
            best = min(range(len(rides)), key=times.__getitem__)
            if times[best] < walk:
                return rides[best]
        self._walks[aid] = key
        return None

    def _ride_time(self, name: str, dest: str, speed: float, effort: float) -> float:
        """Time over a stair from its entry to its partner's: carried at belt speed, climbed at stair speed times the effort on a still one."""
        zone, far = self.zones[name], self.zones[dest]
        run = math.hypot(zone.hold.x - zone.entry.x, zone.hold.y - zone.entry.y) + math.hypot(far.hold.x - far.entry.x, far.hold.y - far.entry.y)
        return run / abs(zone.belt) if zone.belt else run / min(speed, STAIR_SPEED) * effort

    def _blocked(self, climbs: tuple[str, ...]) -> set[str]:
        """Stairs halves a pedestrian climbing these kinds is not routed into: other kinds and closed halves."""
        return {name for name, z in self.zones.items() if z.kind and (z.kind not in climbs or not self.open[name])}

    def _aboard(self, transfer: Transfer, x: float, y: float) -> bool:
        zone = self.zones[transfer.zone]
        return bool(zone.kind) and (zone.contains(x, y) or (self._joined.get(zone.name) == transfer.destination and self.zones[transfer.destination].contains(x, y)))

    def carry(self, pool: AgentPool) -> None:
        """Stand the pedestrians on a running belt and carry them at belt speed, cap walkers on a still stair at stair speed and keep the ones held at a seam still."""
        self.riders = set()
        for i, zone in self._on_stairs(pool):
            aid = int(pool.agent_ids[i])
            if zone.belt:
                self.riders.add(aid)
                pool.desired_vel[i] = 0.0
            if aid in self._held:
                pool.vel[i] = 0.0
            elif zone.belt:
                ax, ay = zone.axis
                pool.vel[i, 0] = zone.belt * ax
                pool.vel[i, 1] = zone.belt * ay
            else:
                speed = math.hypot(float(pool.vel[i, 0]), float(pool.vel[i, 1]))
                if speed > STAIR_SPEED:
                    pool.vel[i] *= STAIR_SPEED / speed
            pool.prev_vel[i] = pool.vel[i]

    def standing(self, pool: AgentPool) -> np.ndarray:
        """Mask over the pool's rows of the pedestrians standing on a running belt."""
        mask = np.zeros(pool.n, dtype=np.bool_)
        for aid in self.riders:
            idx = pool._id_to_idx.get(aid)
            if idx is not None:
                mask[idx] = True
        return mask

    def _on_stairs(self, pool: AgentPool) -> list[tuple[int, Zone]]:
        """Pool rows of the self-driven pedestrians standing on a stair half, with that half."""
        stairs = [z for z in self.zones.values() if z.kind]
        if not stairs:
            return []
        n = pool.n
        pos = pool.pos[:n]
        free = (pool.kind[:n] != KIND_ROBOT) & (pool.policy_idx[:n] >= 0)
        found: dict[int, Zone] = {}
        for zone in stairs:
            xs, ys = [x for x, _ in zone.footprint], [y for _, y in zone.footprint]
            near = free & (pos[:, 0] >= min(xs)) & (pos[:, 0] <= max(xs)) & (pos[:, 1] >= min(ys)) & (pos[:, 1] <= max(ys))
            for i in np.flatnonzero(near).tolist():
                if i not in found and zone.contains(float(pos[i, 0]), float(pos[i, 1])):
                    found[i] = zone
        return sorted(found.items(), key=lambda item: item[0])

    @staticmethod
    def _landing(zone: Zone, dest: Zone, x: float, y: float, theta: float) -> tuple[float, float, float]:
        """Pose past the destination's seam matching a crossing of the zone's seam at x, y, the side offset kept relative to the heading, on the destination's center line when the offset leaves its footprint."""
        ax, ay = zone.axis
        bx, by = dest.axis
        across = -(x - zone.hold.x) * ay + (y - zone.hold.y) * ax
        lx, ly = dest.hold.x - bx * LANDING_BACKOFF, dest.hold.y - by * LANDING_BACKOFF
        if dest.contains(lx + by * across, ly - bx * across):
            lx, ly = lx + by * across, ly - bx * across
        return lx, ly, math.remainder(theta + dest.entry.theta + math.pi - zone.entry.theta, math.tau)

    @staticmethod
    def _occupants(pool: AgentPool, i: int, x: float, y: float) -> list[int]:
        """Rows other than i that row i standing at x, y would overlap."""
        n, r = pool.n, float(pool.agent_radius[i])
        overlap = np.hypot(pool.pos[:n, 0] - x, pool.pos[:n, 1] - y) < pool.agent_radius[:n] + r
        overlap[i] = False
        return np.flatnonzero(overlap).tolist()

    def _fits(self, pool: AgentPool, i: int, j: int, landed: Sequence[tuple[float, float, float]]) -> bool:
        """Whether row i can stand where row j stands once j has left: clear of this tick's landings, and of the other agents when it is the larger of the two."""
        x, y, r = float(pool.pos[j, 0]), float(pool.pos[j, 1]), float(pool.agent_radius[i])
        if any(math.hypot(lx - x, ly - y) < lr + r for lx, ly, lr in landed):
            return False
        return r <= float(pool.agent_radius[j]) or self._occupants(pool, i, x, y) == [j]

    def _cross(self, pool: AgentPool) -> list[tuple[int, float, float, float]]:
        """Teleports of the pedestrians past a stair seam whose landing is clear or who swap spots crossing one stair opposite ways, the others held still at the seam."""
        rows = self._on_stairs(pool)
        on = {int(pool.agent_ids[i]): zone.name for i, zone in rows}
        self._landed = {aid: name for aid, name in self._landed.items() if on.get(aid) == name}
        self._held = set()
        crossing: dict[int, tuple[Zone, Zone, float, float, float]] = {}
        for i, zone in rows:
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            aid = int(pool.agent_ids[i])
            transfer = self.transfers.get(aid)
            routed = transfer.destination if transfer is not None and transfer.zone == zone.name else None
            if not zone.destinations or zone.name in self._joined or (self._landed.get(aid) == zone.name and routed is None) or not zone.past_seam(x, y):
                continue
            dest = self.zones[routed or zone.destinations[0]]
            crossing[i] = (zone, dest, *self._landing(zone, dest, x, y, float(pool.theta[i])))
        moves: dict[int, tuple[float, float, float]] = {}
        blocked: dict[int, list[int]] = {}
        landed: list[tuple[float, float, float]] = []
        for i, (_, _, lx, ly, theta) in crossing.items():
            r = float(pool.agent_radius[i])
            others = self._occupants(pool, i, lx, ly)
            if others or any(math.hypot(ox - lx, oy - ly) < orad + r for ox, oy, orad in landed):
                blocked[i] = others
                continue
            moves[i] = (lx, ly, theta)
            landed.append((lx, ly, r))
        for i, others in blocked.items():
            zone, dest = crossing[i][:2]
            for j in others:
                if i in moves or j in moves or j not in blocked or crossing[j][0] is not dest or crossing[j][1] is not zone or not self._fits(pool, i, j, landed) or not self._fits(pool, j, i, landed):
                    continue
                for a, b in ((i, j), (j, i)):
                    moves[a] = (float(pool.pos[b, 0]), float(pool.pos[b, 1]), crossing[a][4])
                    landed.append((*moves[a][:2], float(pool.agent_radius[a])))
        teleports: list[tuple[int, float, float, float]] = []
        for i, (zone, dest, *_) in crossing.items():
            aid = int(pool.agent_ids[i])
            if i in moves:
                self._landed[aid] = dest.name
                teleports.append((aid, *moves[i]))
                continue
            self._held.add(aid)
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            ax, ay = zone.axis
            excess = zone.along(x, y) - zone.along(zone.hold.x, zone.hold.y)
            pool.pos[i, 0] = x - excess * ax
            pool.pos[i, 1] = y - excess * ay
            pool.vel[i] = 0.0
            pool.prev_vel[i] = 0.0
            pool.desired_vel[i] = 0.0
        return teleports

    def hold(self, pool: AgentPool, tick: int, timeout_ticks: int, dt: float, let_out_ticks: int) -> list[tuple[int, float, float, float]]:
        """Advance every cabin rider's phase, slow the waiting ones to desired speed 0, stand the holding ones still and return the due self-teleports, stair crossings included, as id, x, y, heading."""
        robots = self._zone_robots(pool)
        alighting = self._alighting(pool) if let_out_ticks > 0 else set()
        teleports: list[tuple[int, float, float, float]] = []
        for aid in list(self.transfers):
            transfer = self.transfers[aid]
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                self.forget(aid)
                continue
            zone = self.zones[transfer.zone]
            if zone.kind:
                continue
            is_open = self.open[transfer.zone]
            x, y = float(pool.pos[idx, 0]), float(pool.pos[idx, 1])
            covered = self._covered(zone, robots.get(zone.name, ()), float(pool.agent_radius[idx]))
            if transfer.phase == Phase.WALK and math.hypot(x - zone.wait.x, y - zone.wait.y) <= WAIT_RADIUS:
                transfer.phase, transfer.since = Phase.WAIT, tick
            if transfer.phase == Phase.BOARD and transfer.slot in covered:
                slot = self._free_slot(zone, covered)
                if slot is None:
                    transfer.phase, transfer.slot = Phase.WAIT, -1
                else:
                    transfer.slot = slot
            if transfer.phase == Phase.WAIT and not is_open:
                transfer.yielded = 0
            if transfer.phase == Phase.WAIT and is_open:
                if zone.name in alighting and transfer.yielded < let_out_ticks:
                    transfer.yielded += 1
                else:
                    slot = self._free_slot(zone, covered)
                    if slot is not None:
                        transfer.phase, transfer.slot, transfer.since = Phase.BOARD, slot, tick
            elif transfer.phase == Phase.BOARD and self._at_slot(zone, transfer, x, y):
                transfer.phase, transfer.held = Phase.HOLD, tick
            elif transfer.phase == Phase.BOARD and not is_open and not zone.contains(x, y):
                transfer.phase, transfer.slot = Phase.WAIT, -1
            elif transfer.phase == Phase.HOLD and is_open and not zone.contains(x, y):
                transfer.phase = Phase.BOARD
            if transfer.phase != Phase.WALK and tick - transfer.since >= timeout_ticks:
                del self.transfers[aid]
                names = self._banned.setdefault(aid, (transfer.goal, []))[1]
                if transfer.zone in names:
                    names.remove(transfer.zone)
                names.append(transfer.zone)
                continue
            if transfer.phase in (Phase.WAIT, Phase.HOLD):
                pool.desired_vel[idx] = 0.0
            if transfer.phase == Phase.HOLD:
                pool.vel[idx] = 0.0
                pool.prev_vel[idx] = 0.0
            if transfer.phase == Phase.HOLD and zone.delay >= 0 and tick - transfer.held >= round(zone.delay / dt):
                cx, cy = zone.center
                dx, dy = self.zones[transfer.destination].center
                teleports.append((aid, x - cx + dx, y - cy + dy, float(pool.theta[idx])))
        teleports.extend(self._cross(pool))
        return teleports

    def _free_slot(self, zone: Zone, covered: set[int]) -> int | None:
        taken = {t.slot for t in self.transfers.values() if t.zone == zone.name and t.phase in (Phase.BOARD, Phase.HOLD)} | covered
        return next((i for i in range(len(zone.slots)) if i not in taken), None)

    def _alighting(self, pool: AgentPool) -> set[str]:
        """Zones someone stands in who is not waiting for, boarding or holding in that zone."""
        out: set[str] = set()
        for i in range(pool.n):
            transfer = self.transfers.get(int(pool.agent_ids[i]))
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            for zone in self.zones.values():
                if zone.name in out or (transfer is not None and transfer.zone == zone.name and transfer.phase != Phase.WALK):
                    continue
                if zone.contains(x, y):
                    out.add(zone.name)
        return out

    def _zone_robots(self, pool: AgentPool) -> dict[str, list[tuple[float, float, float]]]:
        """Robots standing in each cabin as x, y, radius."""
        out: dict[str, list[tuple[float, float, float]]] = {}
        cabins = [z for z in self.zones.values() if not z.kind]
        for i in range(pool.n):
            if pool.kind[i] != KIND_ROBOT:
                continue
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            for zone in cabins:
                if zone.contains(x, y):
                    out.setdefault(zone.name, []).append((x, y, float(pool.agent_radius[i])))
        return out

    @staticmethod
    def _covered(zone: Zone, robots: Sequence[tuple[float, float, float]], radius: float) -> set[int]:
        """Spots a pedestrian of this radius cannot stand on because a robot in the zone overlaps them."""
        return {i for i, spot in enumerate(zone.slots) for x, y, r in robots if math.hypot(spot.x - x, spot.y - y) < r + radius}

    @staticmethod
    def _at_slot(zone: Zone, transfer: Transfer, x: float, y: float) -> bool:
        spot = zone.slots[transfer.slot]
        return math.hypot(x - spot.x, y - spot.y) <= HOLD_RADIUS

    def _graph(self, planner: GlobalPlanner) -> _Graph:
        """Teleport edges between areas, an area being the components that one zone's entry and hold pose join."""
        names = sorted(self.zones)
        comps = planner.components([p for name in names for p in (self.zones[name].entry, self._inner(self.zones[name]))])
        parent: dict[int, int] = {}

        def find(c: int) -> int:
            while parent.setdefault(c, c) != c:
                c = parent[c]
            return c

        sides = {name: comps[2 * i] | comps[2 * i + 1] for i, name in enumerate(names)}
        for side in sides.values():
            roots = sorted({find(c) for c in side})
            for r in roots[1:]:
                parent[r] = roots[0]
        area = {c: find(c) for c in parent}
        edges: dict[int, list[tuple[str, str, int]]] = {}
        for name in names:
            if not sides[name]:
                continue
            a = area[min(sides[name])]
            for dest in self.zones[name].destinations:
                if dest not in sides or not sides[dest]:
                    continue
                b = area[min(sides[dest])]
                if a != b:
                    edges.setdefault(a, []).append((name, dest, b))
        return _Graph(area=area, edges=edges, sides=sides)

    def _route(
        self,
        graph: _Graph,
        start: frozenset[int],
        goal: frozenset[int],
        banned: set[str],
        near: tuple[float, float],
        keep: tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        """Zone and destination of the first edge of a fewest-teleports route from start to goal, `keep` when it is one, else the one whose entry is nearest to `near`, None when start reaches goal on foot or no route exists."""
        starts = {graph.area.get(c, c) for c in start}
        goals = {graph.area.get(c, c) for c in goal}
        if not starts or not goals or starts & goals:
            return None
        frontier: dict[int, set[tuple[str, str]]] = {c: set() for c in starts}
        seen = set(starts)
        while frontier:
            reached: dict[int, set[tuple[str, str]]] = {}
            for c, firsts in frontier.items():
                for name, dest, to in graph.edges.get(c, ()):
                    if name in banned or to in seen:
                        continue
                    reached.setdefault(to, set()).update(firsts or {(name, dest)})
            done = set().union(*(firsts for to, firsts in reached.items() if to in goals))
            if done:
                if keep in done:
                    return keep
                return min(done, key=lambda edge: (math.hypot(self.zones[edge[0]].entry.x - near[0], self.zones[edge[0]].entry.y - near[1]), edge))
            seen.update(reached)
            frontier = reached
        return None
