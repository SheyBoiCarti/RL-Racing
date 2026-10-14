"""Held controls and discrete commands are intentionally separate."""

from racing.core.models import CarControl


def held_control(held, bindings):
    def active(action):
        return any(key in held for key in dict(bindings)[action])

    steering = float(active("right")) - float(active("left"))
    longitudinal = (
        -1.0
        if active("brake_override")
        else float(active("accelerate")) - float(active("brake"))
    )
    return CarControl(steering, longitudinal)


def key_bindings(config):
    import pygame

    resolved = {}
    for action, names in config.bindings:
        try:
            resolved[action] = tuple(pygame.key.key_code(name) for name in names)
        except ValueError as error:
            raise ValueError(f"bindings.{action}: unknown keyboard key") from error
    keys = [key for values in resolved.values() for key in values]
    if len(keys) != len(set(keys)):
        raise ValueError("bindings: conflicting keyboard aliases")
    return resolved
