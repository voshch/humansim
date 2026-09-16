"""gesture_mode: the control arm in which agents behave the same but publish no gesture."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("arena_humansim_msgs")
pytest.importorskip("rclpy")

from rclpy.parameter import Parameter

from arena_humansim.core import agent_manager
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_manager import CONTACT_ENABLED, GESTURE_DISABLED, GESTURE_ENABLED


class _InteractionManager:
    contact_mode = CONTACT_ENABLED

    def set_contact_mode(self, mode: str, distance: float | None) -> None:
        del mode, distance


class _Node:
    """Enough of AgentManager for the parameter callback."""

    def __init__(self) -> None:
        self._interaction_manager = _InteractionManager()
        self._gesture_mode = GESTURE_ENABLED


def _set(node: _Node, **params: object):
    return AgentManager._on_contact_params(node, [Parameter(name, value=value) for name, value in params.items()])


def test_gesture_mode_switches_and_rejects_unknown() -> None:
    node = _Node()
    assert _set(node, gesture_mode=GESTURE_DISABLED).successful
    assert node._gesture_mode == GESTURE_DISABLED
    assert _set(node, gesture_mode=GESTURE_ENABLED).successful
    assert node._gesture_mode == GESTURE_ENABLED

    result = _set(node, gesture_mode="off")
    assert not result.successful and "gesture_mode" in result.reason
    assert node._gesture_mode == GESTURE_ENABLED  # a rejected set leaves the arm as it was


def test_unrelated_parameters_pass_through() -> None:
    node = _Node()
    assert _set(node, contact_mode=CONTACT_ENABLED, locomotion_standing_distance=1.2).successful
    assert node._gesture_mode == GESTURE_ENABLED


def test_disabled_also_hides_the_interaction_id() -> None:
    """The id is what lets a costmap group pedestrians, so the control arm must not publish it either."""
    source = pathlib.Path(agent_manager.__file__).read_text()
    body = source.split("rendered = self._gesture_mode == GESTURE_ENABLED", 1)[1].split("return msg", 1)[0]
    assert "if rendered else []" in body
    assert "active_interaction(a.agent_id) if rendered else None" in body
