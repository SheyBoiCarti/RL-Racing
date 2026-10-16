"""Independent rejection and selection fixtures for trained artifacts."""

import hashlib
import json
from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path

import pytest


def test_config_rejects_invalid_batches_seeds_and_unknowns(tmp_path):
    from racing.training.config import TrainConfig, load_train

    for change in ({"batch_size": 63}, {"seed": 9000}, {"n_envs": 2},
                   {"learning_rate": float("nan")}, {"total_timesteps": True},
                   {"eval_seeds": [9000, 9000]}, {"algo": "sac"}):
        with pytest.raises(ValueError):
            replace(TrainConfig(), **change)
    path = tmp_path / "bad.yaml"
    path.write_text("typo: 42\n")
    with pytest.raises(ValueError, match="unknown"):
        load_train(path)


def test_selection_is_permutation_invariant_and_ignores_incomplete_laps():
    from racing.training.selection import choose_best

    def candidate(progress, step, digest):
        return dict(macro_completion=.5, macro_progress=progress,
                    all_success=False, macro_mean_lap_time_s=None,
                    checkpoint_step=step, update_count=1, model_sha256=digest * 64)

    a, b, c = candidate(.8, 100, "a"), candidate(.9, 200, "b"), candidate(.85, 300, "c")
    for order in permutations([a, b, c]):
        assert choose_best(order) == b
    early = dict(b, checkpoint_step=100)
    assert choose_best([b, early]) == early
    low_hash = dict(early, model_sha256="0" * 64)
    assert choose_best([early, low_hash]) == low_hash
    full = dict(a, macro_completion=1., macro_progress=1., all_success=True,
                macro_mean_lap_time_s=25.)
    faster = dict(full, macro_mean_lap_time_s=20., checkpoint_step=500)
    assert choose_best([full, faster]) == faster
    for change in ({"macro_progress": float("nan")}, {"macro_completion": None},
                   {"checkpoint_step": 0}, {"all_success": True}):
        with pytest.raises(ValueError):
            choose_best([dict(a, **change)])


def test_execution_requires_real_session_budget_and_writable_root(tmp_path):
    from racing.training.config import ExecutionConfig

    with pytest.raises(ValueError):
        ExecutionConfig(max_session_seconds=900)
    with pytest.raises(ValueError):
        ExecutionConfig(output_root="/kaggle/input/runs", max_session_seconds=1800)
    assert ExecutionConfig(output_root=str(tmp_path), max_session_seconds=1800).torch_threads == 1


def test_model_sidecar_rejects_semantic_changes_and_corruption(tmp_path):
    from racing.training.checkpoints import validate_metadata
    from racing.training.evaluation import canonical_hash

    model = tmp_path / "model.zip"
    model.write_bytes(b"test bytes, schema validation does not deserialize")
    original = json.loads(Path("docs/specs/examples/model.json").read_text())
    from racing.training.config import TrainConfig

    typed = asdict(TrainConfig())
    original["hyperparameters"] = dict(typed, activation_fn="torch.nn.Tanh", optimizer_class="torch.optim.Adam", optimizer_defaults={})
    original["config_hashes"]["training"] = canonical_hash(typed)
    original.update(fixture_only=False, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    sidecar = model.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(original))
    assert validate_metadata(model)["checkpoint_step"] == 2048
    for field, value in (("fixture_only", True), ("algorithm", "sac")):
        sidecar.write_text(json.dumps(dict(original, **{field: value})))
        with pytest.raises(ValueError):
            validate_metadata(model)
    for field, value in (("learning_rate", .1), ("n_steps", 15)):
        changed = json.loads(json.dumps(original))
        changed["hyperparameters"][field] = value
        sidecar.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="training"):
            validate_metadata(model)
    changed = json.loads(json.dumps(original))
    del changed["hyperparameters"]["seed"]
    sidecar.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="training"):
        validate_metadata(model)
    changed = json.loads(json.dumps(original))
    changed["observation_config"]["ray_angles_deg"].reverse()
    changed["config_hashes"]["observation"] = canonical_hash(changed["observation_config"])
    sidecar.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="observation"):
        validate_metadata(model)
    changed = json.loads(json.dumps(original))
    changed["env_config"]["lap_bonus"] = 75.
    changed["config_hashes"]["env"] = canonical_hash(changed["env_config"])
    sidecar.write_text(json.dumps(changed))
    assert validate_metadata(model)["env_config"]["lap_bonus"] == 75.
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        validate_metadata(model)
