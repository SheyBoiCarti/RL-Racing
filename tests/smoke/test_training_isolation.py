"""Live evaluation must not advance training state or run twice per weight state."""

import copy
from dataclasses import replace

import numpy as np

from racing.envs.config import RLConfig
from racing.training.config import TrainConfig


def test_live_evaluation_preserves_training_state_and_deduplicates(tmp_path, monkeypatch):
    import torch

    from racing.training.callbacks import Coordinator
    from racing.training.train import train

    original = Coordinator.evaluate
    checked = []

    def audited(self, model, diagnostic=False):
        env = model.env.envs[0].unwrapped
        if diagnostic:
            return original(self, model, diagnostic=True)
        snapshot = env.snapshot
        worker_rng = copy.deepcopy(env.np_random.bit_generator.state)
        counters = (model.num_timesteps, model._n_updates, model._current_progress_remaining)
        logger = dict(model.logger.name_to_value)
        modes = [m.training for m in model.policy.modules()]
        weights = [p.detach().clone() for p in model.policy.parameters()]
        result = original(self, model)
        assert env.snapshot == snapshot
        assert env.np_random.bit_generator.state == worker_rng
        assert (model.num_timesteps, model._n_updates, model._current_progress_remaining) == counters
        assert model.logger.name_to_value == logger
        assert [m.training for m in model.policy.modules()] == modes
        assert all(torch.equal(p, before) for p, before in zip(model.policy.parameters(), weights))
        ledger_size = len(self.ledger)
        original(self, model)
        assert len(self.ledger) == ledger_size
        checked.append(counters[0])
        return result

    monkeypatch.setattr(Coordinator, "evaluate", audited)
    config = replace(TrainConfig(), n_steps=64, batch_size=32, n_epochs=1,
                     total_timesteps=64, eval_every_steps=32, eval_seeds=[9000])
    assert train(config, tmp_path / "isolated", env_config=RLConfig(max_steps=8)) == 0
    assert np.array_equal(checked, [64])
