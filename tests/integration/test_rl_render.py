"""Optional rendering is read-only and releases its renderer on close."""

import gc
import weakref

import numpy as np

from racing.envs.racing_env import RacingEnv


def test_human_render_draws_sensor_rays_and_releases_renderer(pygame_env):
    pygame = pygame_env
    env = RacingEnv(render_mode="human")
    env.reset(seed=9000)
    before = env.snapshot
    for _ in range(3):
        env.render()
        assert env.snapshot == before
        pixels = pygame.surfarray.array3d(pygame.display.get_surface())
        assert np.count_nonzero(np.all(pixels == (100, 225, 240), axis=2)) > 100
        renderer = weakref.ref(env._renderer)
        env.close()
        env.close()
        gc.collect()
        assert renderer() is None
        assert pygame.display.get_surface() is None
    env.close()
