from functools import cached_property

import rclpy.node


class Loggable:
    _node = None

    @classmethod
    def init_logging(cls, node: rclpy.node.Node) -> None:
        cls._node = node

    @cached_property
    def _logger(self) -> rclpy.impl.rcutils_logger.RcutilsLogger:
        return self._node.get_logger().get_child(type(self).__qualname__)
