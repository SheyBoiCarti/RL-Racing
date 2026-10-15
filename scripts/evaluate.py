"""Evaluate headless Phase 2 baseline controllers."""

import argparse
import json

from racing.training.evaluation import evaluate


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--agent", required=True, choices=("random", "scripted"))
    result.add_argument(
        "--track", default="demo", help="Fixed track ID (default: demo)"
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
        summary = evaluate(
            options.agent,
            track_id=options.track,
            episodes=options.episodes,
            seed=options.seed,
            output=options.output,
            profile_steps=options.profile_steps,
        )
    except (ValueError, OSError) as error:
        arguments.error(str(error))
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
