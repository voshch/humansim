from typing import Any

import py_trees


def compile_agent_behavior(*args: Any, **kwargs: Any) -> py_trees.trees.BehaviourTree | None:
    # Avoids circular import
    from .compiler import compile_agent_behavior as _compile

    return _compile(*args, **kwargs)


__all__ = ["compile_agent_behavior"]
