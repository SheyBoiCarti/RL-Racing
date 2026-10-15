"""Headless Gymnasium environment (registration does not load a renderer)."""

from gymnasium.envs.registration import register, registry

if "Racing-v0" not in registry:
    register(id="Racing-v0", entry_point="racing.envs.racing_env:RacingEnv")
