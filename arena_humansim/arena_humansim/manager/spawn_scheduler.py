import math
from collections.abc import Sequence

import numpy as np

from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import AgentLifetime, Pose2D, RateKeyframe, Shape, ShapeType, SinkConfig, SourceConfig, SpawnRequest


class SpawnScheduler(Loggable):
    def __init__(self, rng: np.random.Generator):
        self._sources: dict[str, SourceConfig] = {}
        self._sinks: dict[str, SinkConfig] = {}
        self._alive_count: dict[str, int] = {}
        self._total_count: dict[str, int] = {}
        self._agent_source: dict[int, str] = {}
        self._rng = rng

    def tick(self, tick_count: int, dt: float) -> list[SpawnRequest]:
        sim_time_s = tick_count * dt
        requests = []
        for src in self._sources.values():
            rate = self._interpolate_rate(src.rate_profile, sim_time_s)
            if rate <= 0.0:
                continue

            if src.max_total > 0 and self._total_count.get(src.name, 0) >= src.max_total:
                continue
            if src.max_concurrent > 0 and self._alive_count.get(src.name, 0) >= src.max_concurrent:
                continue

            n = int(self._rng.poisson(rate * dt))

            if src.max_concurrent > 0:
                n = min(n, src.max_concurrent - self._alive_count.get(src.name, 0))
            if src.max_total > 0:
                n = min(n, src.max_total - self._total_count.get(src.name, 0))

            for _ in range(n):
                req = self._sample_spawn_request(src, tick_count)
                requests.append(req)
                self._alive_count[src.name] = self._alive_count.get(src.name, 0) + 1
                self._total_count[src.name] = self._total_count.get(src.name, 0) + 1

        if requests:
            self._logger.debug(f"Tick {tick_count}: {len(requests)} spawn request(s)")
        return requests

    def register_agent(self, agent_id: int, source_name: str) -> None:
        self._agent_source[agent_id] = source_name

    def notify_despawn(self, agent_id: int) -> None:
        source_name = self._agent_source.pop(agent_id, None)
        if source_name is not None and source_name in self._alive_count:
            self._alive_count[source_name] = max(0, self._alive_count[source_name] - 1)

    def add_source(self, config: SourceConfig) -> str:
        self._sources[config.name] = config
        self._alive_count.setdefault(config.name, 0)
        self._total_count.setdefault(config.name, 0)
        self._logger.info(f"Source added: {config.name}")
        return config.name

    def remove_source(self, name: str) -> None:
        self._sources.pop(name, None)
        self._logger.info(f"Source removed: {name}")

    def clear_sources(self) -> None:
        self._sources.clear()

    def set_sinks(self, sinks: dict[str, SinkConfig]) -> None:
        self._sinks = sinks

    def reset_counts(self) -> None:
        self._alive_count.clear()
        self._total_count.clear()
        self._agent_source.clear()

    @staticmethod
    def _interpolate_rate(profile: Sequence[RateKeyframe], t: float) -> float:
        if not profile:
            return 0.0
        if len(profile) == 1:
            return profile[0].rate
        if t <= profile[0].t:
            return profile[0].rate
        if t >= profile[-1].t:
            return profile[-1].rate

        for i in range(len(profile) - 1):
            if profile[i].t <= t <= profile[i + 1].t:
                dt = profile[i + 1].t - profile[i].t
                if dt <= 0:
                    return profile[i + 1].rate
                alpha = (t - profile[i].t) / dt
                return profile[i].rate + alpha * (profile[i + 1].rate - profile[i].rate)

        return profile[-1].rate

    def _sample_spawn_request(self, src: SourceConfig, tick_count: int) -> SpawnRequest:
        tmpl = src.agent

        pose = self._sample_pose_in_shape(src.pose, src.shape)

        desired_velocity = self._rng.uniform(tmpl.desired_velocity_min, tmpl.desired_velocity_max)

        target_sink_name = ""
        waypoints: list[Pose2D] = []
        if tmpl.sink_affinity:
            weights = np.array([sa.weight for sa in tmpl.sink_affinity])
            weights = weights / weights.sum()
            chosen = int(self._rng.choice(len(tmpl.sink_affinity), p=weights))
            target_sink_name = tmpl.sink_affinity[chosen].sink_name

            if target_sink_name and target_sink_name in self._sinks:
                sink_pose = self._sinks[target_sink_name].pose
                waypoints = [Pose2D(x=sink_pose.x, y=sink_pose.y, theta=sink_pose.theta)]
                pose.theta = math.atan2(sink_pose.y - pose.y, sink_pose.x - pose.x)

        lifetime = AgentLifetime(
            source_name=src.name,
            spawn_tick=tick_count,
            target_sink_name=target_sink_name,
        )

        return SpawnRequest(
            pose=pose,
            desired_velocity=desired_velocity,
            agent_radius=tmpl.agent_radius,
            agent_type=tmpl.agent_type,
            waypoints=waypoints,
            lifetime=lifetime,
        )

    def _sample_pose_in_shape(self, center: Pose2D, shape: Shape) -> Pose2D:
        if shape.type == ShapeType.CIRCLE and shape.radius > 0:
            r = shape.radius * math.sqrt(float(self._rng.random()))
            angle = float(self._rng.uniform(0, 2 * math.pi))
            return Pose2D(
                x=center.x + r * math.cos(angle),
                y=center.y + r * math.sin(angle),
                theta=0.0,
            )
        if shape.vertices:
            return self._sample_pose_in_polygon(shape.vertices, center)
        return Pose2D(x=center.x, y=center.y, theta=0.0)

    def _sample_pose_in_polygon(self, vertices: Sequence[Pose2D], center: Pose2D) -> Pose2D:
        n = len(vertices)
        if n < 3:
            if vertices:
                return Pose2D(x=center.x + vertices[0].x, y=center.y + vertices[0].y)
            return Pose2D(x=center.x, y=center.y)

        triangles = []
        areas = []
        v0 = vertices[0]
        for i in range(1, n - 1):
            v1, v2 = vertices[i], vertices[i + 1]
            area = abs((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)) / 2.0
            triangles.append((v0, v1, v2))
            areas.append(area)

        total_area = sum(areas)
        if total_area <= 0:
            return Pose2D(x=center.x + v0.x, y=center.y + v0.y)

        r = float(self._rng.random()) * total_area
        cumulative = 0.0
        tri = triangles[0]
        for tri, area in zip(triangles, areas):
            cumulative += area
            if cumulative >= r:
                break

        u = float(self._rng.random())
        v = float(self._rng.random())
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v

        a, b, c = tri
        x = center.x + w * a.x + u * b.x + v * c.x
        y = center.y + w * a.y + u * b.y + v * c.y
        return Pose2D(x=x, y=y, theta=0.0)
