"""Train the versioned PPO pipeline; exit 75 means safely resumable session stop."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing.training.config import load_execution, load_train
from racing.training.evaluation import canonical_hash
from racing.training.manifest import ROOT, now
from racing.training.train import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / "configs/train_ppo.yaml"))
    parser.add_argument("--execution-config")
    parser.add_argument("--algo", choices=["ppo"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int, help="Explicit controlled-stop acceptance experiment")
    args = parser.parse_args(argv)
    try:
        config = load_train(args.config, seed=args.seed, total_timesteps=args.total_timesteps, device=args.device)
        if args.algo is not None and args.algo != config.algo:
            raise ValueError("--algo contradicts config")
        execution = load_execution(args.execution_config) if args.execution_config else None
        root = Path(execution.output_root) if execution else ROOT / "runs"
        run_dir = args.run_dir or str(root / f"{now().replace(':', '').replace('+', '_')}-seed{config.seed}-{canonical_hash(config)[:8]}")
        if args.stop_after_updates is not None and args.stop_after_updates <= 0:
            raise ValueError("--stop-after-updates must be positive")
        calls = 0

        def stop_requested():
            nonlocal calls
            calls += 1
            return args.stop_after_updates is not None and calls >= args.stop_after_updates

        return train(config, run_dir, execution=execution, resume=args.resume, stop_requested=stop_requested)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
