"""PX4/Gazebo environment exports; operational checks need only the stdlib."""
__all__ = ["RateControlEnv", "TaskConfig"]

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .env import RateControlEnv, TaskConfig
    exports = {"RateControlEnv": RateControlEnv, "TaskConfig": TaskConfig}
    globals().update(exports)
    return exports[name]
