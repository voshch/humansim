import math

import attrs

from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import Pose2D


@attrs.define
class WorldObject:
    object_id: str
    type: str
    pose: Pose2D = attrs.Factory(Pose2D)
    capacity: int = 1
    satisfies: dict[str, float] = attrs.Factory(dict)
    formation: "FormationSpec | None" = None
    interaction_radius: float | None = None


_VALID_FORMATION_TYPES = ("line", "cluster", "f_formation", "dyad")
_VALID_ANCHOR_KINDS = ("object", "agent", "pose", "centroid")


@attrs.define
class FormationSpec:
    """Runtime representation of a scenario FormationConfig on a WorldObject."""

    type: str
    params: dict[str, float] = attrs.Factory(dict)
    anchor_kind: str = "object"
    anchor_ref: str | None = None
    anchor_pose: Pose2D | None = None

    @classmethod
    def from_config(cls, cfg: object) -> "FormationSpec | None":
        if cfg is None:
            return None
        ftype = getattr(cfg, "type", None)
        if not ftype:
            return None
        if ftype not in _VALID_FORMATION_TYPES:
            raise ValueError(f"Unknown formation type '{ftype}'. Valid: {_VALID_FORMATION_TYPES}")
        params_cfg = getattr(cfg, "params", None) or {}
        anchor_cfg = getattr(cfg, "anchor", None)
        anchor_kind = "object"
        anchor_ref: str | None = None
        anchor_pose: Pose2D | None = None
        if anchor_cfg is not None:
            anchor_kind = getattr(anchor_cfg, "kind", "object")
            if anchor_kind not in _VALID_ANCHOR_KINDS:
                raise ValueError(f"Unknown anchor kind '{anchor_kind}'. Valid: {_VALID_ANCHOR_KINDS}")
            anchor_ref = getattr(anchor_cfg, "ref", None)
            pose_cfg = getattr(anchor_cfg, "pose", None)
            if pose_cfg is not None:
                anchor_pose = Pose2D(
                    x=float(getattr(pose_cfg, "x", 0.0)),
                    y=float(getattr(pose_cfg, "y", 0.0)),
                    theta=float(getattr(pose_cfg, "theta", 0.0)),
                )
        return cls(
            type=ftype,
            params=dict(params_cfg),
            anchor_kind=anchor_kind,
            anchor_ref=anchor_ref,
            anchor_pose=anchor_pose,
        )


class WorldKnowledge(Loggable):
    def __init__(self) -> None:
        self._objects: dict[str, WorldObject] = {}
        self._by_type: dict[str, list[WorldObject]] = {}
        self._queue_lengths: dict[str, int] = {}
        self._participants_counts: dict[str, int] = {}

    def add_object(self, obj: WorldObject) -> None:
        self._objects[obj.object_id] = obj
        self._by_type.setdefault(obj.type, []).append(obj)
        self._logger.debug(f"World object added: {obj.object_id} ({obj.type})")

    def remove_object(self, object_id: str) -> WorldObject | None:
        obj = self._objects.pop(object_id, None)
        if obj is not None:
            type_list = self._by_type.get(obj.type)
            if type_list:
                type_list[:] = [o for o in type_list if o.object_id != object_id]
                if not type_list:
                    del self._by_type[obj.type]
            self._queue_lengths.pop(object_id, None)
            self._participants_counts.pop(object_id, None)
        return obj

    def get(self, object_id: str) -> WorldObject | None:
        return self._objects.get(object_id)

    def get_by_type(self, object_type: str) -> list[WorldObject]:
        return list(self._by_type.get(object_type, []))

    def nearest_object(
        self,
        object_type: str,
        from_pose: Pose2D,
        exclude_full: bool = True,
    ) -> WorldObject | None:
        candidates = self._by_type.get(object_type, [])
        best: WorldObject | None = None
        best_dist = float("inf")

        for obj in candidates:
            if exclude_full:
                q_len = self._queue_lengths.get(obj.object_id, 0)
                if obj.capacity > 0 and q_len >= obj.capacity:
                    continue

            dx = obj.pose.x - from_pose.x
            dy = obj.pose.y - from_pose.y
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best = obj

        return best

    def object_pose(self, object_id: str) -> Pose2D | None:
        obj = self._objects.get(object_id)
        return obj.pose if obj else None

    def set_queue_length(self, object_id: str, length: int) -> None:
        self._queue_lengths[object_id] = length

    def queue_length(self, object_type: str) -> int:
        total = 0
        for obj in self._by_type.get(object_type, []):
            total += self._queue_lengths.get(obj.object_id, 0)
        return total

    def queue_length_for_object(self, object_id: str) -> int:
        return self._queue_lengths.get(object_id, 0)

    def set_participants_count(self, object_id: str, count: int) -> None:
        self._participants_counts[object_id] = count

    def participants_count_for_object(self, object_id: str) -> int:
        return self._participants_counts.get(object_id, 0)

    def clear(self) -> None:
        self._objects.clear()
        self._by_type.clear()
        self._queue_lengths.clear()
        self._participants_counts.clear()

    def __len__(self) -> int:
        return len(self._objects)

    def __bool__(self) -> bool:
        return bool(self._objects)
