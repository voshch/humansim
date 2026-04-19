from __future__ import annotations

import attrs

from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D


@attrs.define
class StepContext:
    target_pose: Pose2D | None = None
    target_object_id: str | None = None
    interaction_radius: float = DISTANCE_TOLERANCE
    advertised: bool = False
