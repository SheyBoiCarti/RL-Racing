"""Real PPO updates, isolation, exact-checkpoint resume and corruption checks."""

import copy
import random
from dataclasses import replace

import numpy as np
import pytest

from racing.envs.config import RLConfig
from racing.training.config import TrainConfig


def test_evaluation_restores_modes_and_global_rng_on_error():
    import torch

    from racing.training.callbacks import isolated_evaluation

    policy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    policy.train()
    policy[0].eval()
    modes = [m.training for m in policy.modules()]
    python = random.getstate()
    numpy = copy.deepcopy(np.random.get_state())
    cpu = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with isolated_evaluation(policy):
            policy.eval()
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("evaluation failed")
    assert [m.training for m in policy.modules()] == modes
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)


def test_post_update_smoke_resume_and_checkpoint_bound_ledger(tmp_path):
    from racing.agents.policy import PolicyController
    from racing.training.checkpoints import load_bundle, validate_metadata
    from racing.training.manifest import read_json
    from racing.training.train import train

    config = replace(TrainConfig(), n_steps=64, batch_size=32, n_epochs=1,
                     total_timesteps=128, eval_every_steps=64, eval_seeds=[9000])
    env = RLConfig(max_steps=8, no_progress_steps=8)
    parent = tmp_path / "parent"
    assert train(config, parent, env_config=env) == 0
    manifest = read_json(parent / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["actual_transitions"] == 128
    final = validate_metadata(parent / "final_model.zip", env)
    assert final["checkpoint_step"] == 128 and final["update_count"] == 2
    policy = PolicyController(parent / "best_model.zip", config=env)
    action = policy.act(np.zeros(11, dtype=np.float32), None, None)
    assert -1 <= action.steering <= 1 and -1 <= action.longitudinal <= 1
    earlier = parent / "checkpoints/step_000000064_update_000001/model.zip"
    _, _, state = load_bundle(earlier)
    assert [c["checkpoint_step"] for c in state["ledger"]] == [64]
    assert state["next_due"] == 128
    child = tmp_path / "child"
    assert train(config, child, resume=earlier, env_config=env) == 0
    child_manifest = read_json(child / "manifest.json")
    assert child_manifest["actual_transitions"] == 128
    assert child_manifest["additional_transitions"] == 64
    assert child_manifest["resume_mode"] == "episode_reset"
    _, _, child_state = load_bundle(child / "final_model.zip")
    assert len(child_state["ledger"]) == 2
    # Resume state belongs to the earlier checkpoint, not the parent's final state.
    assert child_state["ledger"][0]["model_sha256"] == state["ledger"][0]["model_sha256"]
    with pytest.raises(ValueError, match="final absolute"):
        train(replace(config, total_timesteps=64), tmp_path / "invalid", resume=earlier, env_config=env)
    state_file = earlier.parent / "resume/trainer_state.json"
    state_file.write_bytes(state_file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(earlier)


def test_controlled_stop_preserves_original_budget_without_extra_selection(tmp_path):
    from racing.training.checkpoints import load_bundle
    from racing.training.manifest import read_json
    from racing.training.train import RESUMABLE_STOP, train

    config = replace(TrainConfig(), n_steps=64, batch_size=32, n_epochs=1,
                     total_timesteps=192, eval_every_steps=128, eval_seeds=[9000])
    env = RLConfig(max_steps=8, no_progress_steps=8)
    root = tmp_path / "stop"
    assert train(config, root, env_config=env, stop_requested=lambda: True) == RESUMABLE_STOP
    manifest = read_json(root / "manifest.json")
    assert manifest["requested_transitions"] == 192
    assert manifest["status"] == "interrupted"
    latest = root / read_json(root / "checkpoints/committed.json")["latest"]
    _, _, state = load_bundle(latest)
    assert state["ledger"] == [] and state["next_due"] == 128
    assert not (root / "final_model.zip").exists()
    assert train(config, tmp_path / "resumed", resume=latest, env_config=env) == 0
    _, _, done = load_bundle(tmp_path / "resumed/final_model.zip")
    assert [c["checkpoint_step"] for c in done["ledger"]] == [128, 192]
    # Even a nonselectable parent checkpoint is required by resume provenance.
    from racing.training.export import export_runs
    child = tmp_path / "resumed"
    child_manifest = read_json(child / "manifest.json")
    ancestor_model = child / child_manifest["parent_checkpoint"]
    ancestor_manifest = ancestor_model.parents[2] / "manifest.json"
    original = ancestor_manifest.read_bytes()
    ancestor_manifest.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="ancestor|parent"):
        load_bundle(child / "final_model.zip")
    ancestor_manifest.write_bytes(original)
    ancestor_model.unlink()
    with pytest.raises(ValueError, match="ancestor|parent|Missing"):
        export_runs(child, tmp_path / "bad-export.zip")
