"""Evaluate headless Phase 2 baseline controllers."""

import argparse
import json

from racing.training.evaluation import evaluate, evaluate_split


class EvaluationParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        options = super().parse_args(args, namespace)
        if options.agent != "ppo" and options.model is not None:
            self.error("--model is only valid with --agent ppo")
        if options.agent == "ppo" and options.model is None:
            self.error("--agent ppo requires --model")
        if options.split == "test":
            self.error("test inference requires Phase 6 frozen selection authorization")
        if options.split:
            if not options.split_manifest:
                self.error("--split requires --split-manifest")
            if options.episodes is not None or options.profile_steps is not None:
                self.error("split evaluation uses --episodes-per-track, without fixed/profile counts")
            options.episodes_per_track = 3 if options.episodes_per_track is None else options.episodes_per_track
        elif options.episodes_per_track is not None or options.split_manifest is not None:
            self.error("--episodes-per-track and --split-manifest require --split")
        return options


def parser():
    result = EvaluationParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--agent", required=True, choices=("random", "scripted", "ppo"))
    result.add_argument("--model", help="PPO zip; matching .metadata.json required")
    track = result.add_mutually_exclusive_group()
    track.add_argument("--split", choices=("validation", "test"))
    result.add_argument("--split-manifest")
    result.add_argument("--episodes-per-track", type=int)
    track.add_argument(
        "--track", help="Fixed track ID (default: demo)"
    )
    count = result.add_mutually_exclusive_group()
    count.add_argument(
        "--episodes", type=int, help="Positive complete-attempt count (default: 20)"
    )
    count.add_argument(
        "--profile-steps", type=int, help="Profile exactly this many simulation steps"
    )
    result.add_argument(
        "--seed", type=int, default=9000, help="Base reset seed (default: 9000)"
    )
    result.add_argument(
        "--output",
        required=True,
        help="New artifact directory; existing paths rejected",
    )
    return result


def main(argv=None):
    arguments = parser()
    options = arguments.parse_args(argv)
    try:
        if options.split:
            summary = evaluate_split(options.agent, split=options.split, split_manifest=options.split_manifest,
                                     episodes_per_track=options.episodes_per_track, seed=options.seed,
                                     output=options.output, model=options.model)
        else:
            summary = evaluate(
                options.agent,
                track_id=options.track or "demo",
                episodes=options.episodes,
                seed=options.seed,
                output=options.output,
                profile_steps=options.profile_steps,
                model=options.model,
            )
    except (ValueError, OSError) as error:
        arguments.error(str(error))
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
