"""SB3 2.9.0 adapter: selectable weights exist only after a completed update."""

from stable_baselines3 import PPO


class PostUpdatePPO(PPO):
    evaluation_coordinator = None

    def train(self):
        super().train()
        if self.evaluation_coordinator is not None:
            self.evaluation_coordinator.after_update(self)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["evaluation_coordinator"]
