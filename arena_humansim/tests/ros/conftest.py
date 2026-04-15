from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.ros._helpers import RosTestSystem, make_system


@pytest.fixture(scope="module")
def ros_system(rclpy_context) -> Iterator[RosTestSystem]:
    sys = make_system()
    try:
        yield sys
    finally:
        sys.shutdown()
