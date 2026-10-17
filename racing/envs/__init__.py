"""Headless Gymnasium environment (registration does not load a renderer)."""

try:
    from gymnasium.envs.registration import register, registry
except ModuleNotFoundError as error:
    # Shared sensor/config primitives also serve model-free race playback.
    # Constructing RacingEnv itself still requires the optional Gym dependency.
    if not (error.name == "gymnasium" or error.name.startswith("gymnasium.")):
        raise
else:
    if "Racing-v0" not in registry:
        register(id="Racing-v0", entry_point="racing.envs.racing_env:RacingEnv")
