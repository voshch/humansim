"""Reach envelopes shared with the arena gesture layer (asserted equal there at import)."""

import math

ARM_IN = math.radians(90)
ARM_OUT = math.radians(110)
HEAD_IN = math.radians(60)
HEAD_OUT = math.radians(70)
MIN_RESIDENCE_S = 0.5

_ENVELOPES = {"head": (HEAD_IN, HEAD_OUT)}


def reachable(slot: str, azimuth_rad: float, was_shown: bool) -> bool:
    """Hysteresis: IN to start showing, OUT to stop; arm slots share the arm envelope."""
    lo, hi = _ENVELOPES.get(slot, (ARM_IN, ARM_OUT))
    return abs(azimuth_rad) < (hi if was_shown else lo)
