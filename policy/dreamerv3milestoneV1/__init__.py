from .thick_context import SlowContextModule, ThickContextConfig

__all__ = ["DreamerV3PolicyFolder", "SlowContextModule", "ThickContextConfig"]


def __getattr__(name):
    if name == "DreamerV3PolicyFolder":
        from .dreamerv3_policy_impl import DreamerV3PolicyFolder

        return DreamerV3PolicyFolder
    raise AttributeError(name)
