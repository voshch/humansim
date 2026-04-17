from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.world_knowledge import FormationSpec
from arena_humansim.utils.types import Pose2D


@dataclass
class _AnchorCfg:
    kind: str = "object"
    ref: str | None = None
    pose: object | None = None


@dataclass
class _FormCfg:
    type: str = ""
    anchor: _AnchorCfg | None = None
    params: dict = field(default_factory=dict)


def test_from_config_returns_none_for_none_input() -> None:
    assert FormationSpec.from_config(None) is None


def test_from_config_returns_none_for_empty_type() -> None:
    assert FormationSpec.from_config(_FormCfg(type="")) is None


def test_from_config_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown formation type"):
        FormationSpec.from_config(_FormCfg(type="bogus"))


def test_from_config_rejects_unknown_anchor_kind() -> None:
    cfg = _FormCfg(type="line", anchor=_AnchorCfg(kind="weird"))
    with pytest.raises(ValueError, match="Unknown anchor kind"):
        FormationSpec.from_config(cfg)


def test_from_config_defaults_to_object_anchor() -> None:
    spec = FormationSpec.from_config(_FormCfg(type="line"))
    assert spec is not None
    assert spec.type == "line"
    assert spec.anchor_kind == "object"
    assert spec.anchor_ref is None


def test_from_config_carries_params() -> None:
    cfg = _FormCfg(type="line", params={"base_step": 1.5})
    spec = FormationSpec.from_config(cfg)
    assert spec is not None
    assert spec.params == {"base_step": 1.5}


def test_from_config_parses_pose_anchor() -> None:
    pose = Pose2D(x=1.0, y=2.0, theta=0.5)
    cfg = _FormCfg(type="cluster", anchor=_AnchorCfg(kind="pose", pose=pose))
    spec = FormationSpec.from_config(cfg)
    assert spec is not None
    assert spec.anchor_kind == "pose"
    assert spec.anchor_pose == pose


def test_from_config_centroid_anchor() -> None:
    cfg = _FormCfg(type="f_formation", anchor=_AnchorCfg(kind="centroid"))
    spec = FormationSpec.from_config(cfg)
    assert spec is not None
    assert spec.anchor_kind == "centroid"
    assert spec.anchor_ref is None


def test_from_config_agent_anchor_with_ref() -> None:
    cfg = _FormCfg(type="line", anchor=_AnchorCfg(kind="agent", ref="42"))
    spec = FormationSpec.from_config(cfg)
    assert spec is not None
    assert spec.anchor_kind == "agent"
    assert spec.anchor_ref == "42"
