from __future__ import annotations

from collections.abc import Iterator

import pytest

from arena_humansim.utils.loggable import Loggable


@pytest.fixture(scope="session", autouse=True)
def _loggable_node(rclpy_context: object) -> Iterator[None]:  # noqa: ARG001
    import rclpy

    node = rclpy.create_node("contracts_loggable")
    Loggable.init_logging(node)
    yield
    node.destroy_node()
