def compile_agent_behavior(*args, **kwargs):
    # Avoids circular import
    from .compiler import compile_agent_behavior as _compile

    return _compile(*args, **kwargs)


__all__ = ["compile_agent_behavior"]
