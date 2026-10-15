import importlib.util

import numpy as np
import pytest

from racing.envs.config import RLConfig, load_rl
from racing.envs.racing_env import RacingEnv


def test_environment_module_exists():
    assert importlib.util.find_spec("racing.envs") is not None


def test_reset_seed_draw_order_and_stream():
    env = RacingEnv()
    rng = np.random.default_rng(9000)
    obs, info = env.reset(seed=9000)
    assert info["reset_seed"] == 9000
    assert info["simulator_seed"] == int(rng.integers(0, 2**32, dtype=np.uint32))
    assert env.observation_space.contains(obs)
    _, info = env.reset()
    assert info["reset_seed"] is None
    assert info["simulator_seed"] == int(rng.integers(0, 2**32, dtype=np.uint32))
    assert load_rl() == RLConfig()


@pytest.mark.parametrize("action", [[0], [[0, 0]], [np.nan, 0], [0, np.inf], ["x", 0]])
def test_bad_action(action):
    env = RacingEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.snapshot.tick == 0


@pytest.mark.parametrize(
    "options", [{"extra": 1}, {"track_id": "missing"}, {"spawn_id": "missing"}, []]
)
def test_bad_options(options):
    with pytest.raises(ValueError):
        RacingEnv().reset(seed=1, options=options)


def test_lifecycle_clipping_reproducibility_and_reset():
    env = RacingEnv(RLConfig(no_progress_steps=2))
    first, _ = env.reset(seed=3)
    other = RacingEnv(RLConfig(no_progress_steps=2))
    other.reset(seed=3)
    a = env.step([2, 0])
    b = other.step([1, 0])
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]
    assert env.snapshot.tick == 1
    _, _, terminated, truncated, info = env.step([0, 0])
    assert not terminated and truncated and info["episode_reason"] == "no_progress"
    with pytest.raises(RuntimeError):
        env.step([0, 0])
    fresh, info = env.reset(seed=3)
    np.testing.assert_array_equal(first, fresh)
    assert info["no_progress_steps"] == 0
    assert all(v == 0 for v in info["reward_terms"].values())
    env.close()
    env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": True},
        {"no_progress_steps": 0},
        {"ray_range_m": 30},
        {"ray_angles_deg": (0,)},
        {"ray_angles_deg": None},
        {"progress_threshold_m": float("nan")},
        {"physics": {}},
        {"observation_version": "unknown"},
        {"allowed_split": ""},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RLConfig(**kwargs)


def test_split_restriction_and_invalid_render_mode():
    from racing.tracks.loader import Track

    track = Track.from_json("assets/tracks/demo.json")
    with pytest.raises(ValueError, match="split"):
        RacingEnv(track_catalog={"demo": (track, "test")}).reset(seed=1)
    with pytest.raises(ValueError, match="render_mode"):
        RacingEnv(render_mode="rgb_array")


def test_no_progress_strict_threshold(monkeypatch):
    from dataclasses import replace

    from racing.core.models import StepResult

    env = RacingEnv()
    env.reset(seed=1)
    world = env.snapshot
    deltas = iter((0.01, 0.010001))

    def step(controls):
        delta = next(deltas)
        env.simulation._world = replace(
            world, cars=(replace(world.cars[0], rewardable_delta_m=delta),)
        )
        return StepResult(env.snapshot, ())

    monkeypatch.setattr(env.simulation, "step", step)
    assert env.step([0, 0])[-1]["no_progress_steps"] == 1
    assert env.step([0, 0])[-1]["no_progress_steps"] == 0


def test_catalog_cannot_relabel_track_split(demo_track):
    demo_track.split = "test"
    env = RacingEnv(track_catalog={"demo": (demo_track, "demo")})
    with pytest.raises(ValueError, match="split"):
        env.reset(seed=1)


@pytest.mark.parametrize("track_id", ["demo", "unclassified"])
@pytest.mark.parametrize("label_as_demo", [False, True])
def test_catalog_rejects_missing_split_provenance(track_id, label_as_demo):
    import json

    from racing.core.config import project_path
    from racing.tracks.loader import Track

    data = json.loads(project_path("assets/tracks/demo.json").read_text())
    del data["split"]
    data["track_id"] = track_id
    track = Track(data)
    entry = (track, "demo") if label_as_demo else track
    with RacingEnv(track_catalog={track_id: entry}) as env:
        with pytest.raises(ValueError, match="split"):
            env.reset(seed=1, options={"track_id": track_id})


def test_boolean_ray_angles_are_not_numbers():
    with pytest.raises(ValueError, match="ray"):
        RLConfig(ray_angles_deg=(-90, -60, -30, False, 30, 60, 90))


def test_seeded_moving_terminal_trace_and_fresh_reset(demo_track):
    first = RacingEnv(track_catalog={"demo": demo_track})
    second = RacingEnv(track_catalog={"demo": demo_track})
    spawn, spawn_info = first.reset(seed=71)
    second.reset(seed=71)
    for _ in range(300):
        a = first.step([0.0, 1.0])
        b = second.step([0.0, 1.0])
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        if a[2] or a[3]:
            break
    assert a[2] and not a[3] and a[-1]["episode_reason"] == "off_track"
    assert not np.array_equal(a[0][:7], spawn[:7])
    assert first.snapshot.tick > 1
    fresh, info = first.reset(seed=71)
    np.testing.assert_array_equal(fresh, spawn)
    assert info == spawn_info
    first.close()
    second.close()
