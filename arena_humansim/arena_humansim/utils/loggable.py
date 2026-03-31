from functools import cached_property


class Loggable:
    _node = None

    @classmethod
    def init_logging(cls, node):
        cls._node = node

    @cached_property
    def _logger(self):
        return self._node.get_logger().get_child(type(self).__qualname__)
