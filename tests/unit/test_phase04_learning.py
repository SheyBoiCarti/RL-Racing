"""Track sampling and held-out validation contracts."""

from dataclasses import replace

import numpy as np
import pytest

from racing.envs.config import RLConfig
from racing.envs.racing_env import RacingEnv
from racing.tracks.loader import Track
from racing.training.config import TrainConfig


def catalog():
    demo = Track.from_json("assets/tracks/demo.json")
    result = {}
    for name in ("train_b", "train_a"):
        data = dict(demo.geometry_payload, track_id=name, name=name, split="train")
        result[name] = Track(data)
    return result


def test_uniform_sampling_uses_sorted_ids_and_seeded_rng():
    config = replace(RLConfig(), track_sampling="uniform_train", allowed_split="train")
    env = RacingEnv(config=config, track_catalog=catalog())
    expected_rng = np.random.default_rng(42)
    expected = []
    actual = []
    for index in range(8):
        expected.append(["train_a", "train_b"][expected_rng.integers(2)])
        expected_rng.integers(0, 2**32, dtype=np.uint32)
        _, info = env.reset(seed=42 if index == 0 else None)
        actual.append(info["track_id"])
    assert actual == expected
    with pytest.raises(ValueError, match="unknown|outside"):
        env.reset(options={"track_id": "test_000"})


def test_uniform_sampling_rejects_nontrain_split():
    with pytest.raises(ValueError, match="train"):
        replace(RLConfig(), track_sampling="uniform_train", allowed_split="validation")


def test_phase04_requires_manifest_and_blocks_test_evaluation():
    with pytest.raises(ValueError, match="manifest"):
        TrainConfig(track_sampling="uniform_train")
    with pytest.raises(ValueError, match="test|validation"):
        TrainConfig(evaluation_split="test", split_manifest="experiments/splits/v1.json")


def test_validation_test_embargo_precedes_model_loading(tmp_path):
    from racing.training.evaluation import evaluate_split

    with pytest.raises(ValueError, match="Phase 6|test"):
        evaluate_split("ppo", split="test", split_manifest="missing", model="missing", output=tmp_path / "eval")


def test_full_condition_configs_share_budget_and_learning_parameters():
    from dataclasses import asdict

    from racing.training.config import load_train

    single = asdict(load_train("experiments/phase04_single.yaml"))
    multi = asdict(load_train("experiments/phase04_multi.yaml"))
    assert single["total_timesteps"] == multi["total_timesteps"] == 1_000_000
    assert single["eval_seeds"] == multi["eval_seeds"] == [9000, 9001, 9002]
    for name in ("condition", "track_sampling"):
        single.pop(name)
        multi.pop(name)
    assert single == multi


def test_uniform_explicit_override_does_not_consume_selection_rng():
    env = RacingEnv(config=replace(RLConfig(), allowed_split="train", track_sampling="uniform_train"), track_catalog=catalog())
    _, info = env.reset(seed=42, options={"track_id": "train_b"})
    assert info["track_id"] == "train_b"
    assert info["simulator_seed"] == 383329928
