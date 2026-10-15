"""A minimal registered-environment smoke check."""

import gymnasium as gym

import racing.envs  # noqa: F401


def test_registered_environment_steps_without_rendering():
    with gym.make("Racing-v0") as env:
        env.reset(seed=123)
        observation, reward, terminated, truncated, info = env.step([0.0, 1.0])
        assert env.observation_space.contains(observation)
        assert info["tick"] == 1
        assert not terminated and not truncated
        assert isinstance(reward, float)
