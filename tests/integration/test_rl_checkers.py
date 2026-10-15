import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check

from racing.envs.config import RLConfig
from racing.envs.racing_env import RacingEnv


def test_checkers_and_registration():
    env = RacingEnv()
    check_env(env, skip_render_check=True)
    sb3_check(env)
    assert env._renderer is None
    registered = gym.make("Racing-v0")
    registered.reset(seed=1)
    registered.close()
    env.close()


def test_ten_thousand_headless_ticks():
    # Fresh process catches even indirect pygame imports, independently of GUI tests.
    script = """
import builtins
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name == 'pygame' or name.startswith('pygame.'):
        raise AssertionError('headless imported pygame')
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
import numpy as np
from racing.envs.racing_env import RacingEnv
env=RacingEnv()
env.reset(seed=4)
for i in range(10000):
    obs,reward,terminated,truncated,info=env.step([0,1])
    assert np.isfinite(obs).all() and np.isfinite(reward)
    assert env.observation_space.contains(obs)
    if terminated or truncated:
        env.reset()
env.close()
print('10000 finite headless ticks')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "10000 finite headless ticks" in result.stdout


def test_repeated_close_reset_has_no_episode_history():
    env = RacingEnv(RLConfig(no_progress_steps=1))
    for _ in range(200):
        env.reset(seed=1)
        _, _, _, truncated, info = env.step(np.zeros(2))
        assert truncated
        json.dumps(info, allow_nan=False)
        env.close()
        assert env._renderer is None
        assert env.snapshot.cars[0].completed_lap_times_s == ()
        assert env._steps == 1
