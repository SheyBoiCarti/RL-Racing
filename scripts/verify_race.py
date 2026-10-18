"""Drive and reconstruct a complete three-lap race; no training or test tracks."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from racing.agents.scripted import ScriptedAgent
from racing.core.config import project_path
from racing.race.config import RaceRules
from racing.race.replay import ReplayRecorder, load_replay, verification_state
from racing.race.replay_state import plain, state_hash
from racing.race.session import RaceSession
from racing.tracks import Track


class ScriptedCar:
    """Use the existing privileged diagnostic driver for one explicitly named car."""

    def __init__(self, car_id):
        self.car_id, self.driver = car_id, ScriptedAgent()

    def reset(self, seed):
        self.driver.reset(seed)

    def act(self, observation, snapshot, track):
        car = next(car for car in snapshot.cars if car.car_id == self.car_id)
        return self.driver.act(observation, replace(snapshot, cars=(car,)), track)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Optional real one-lap PPO checkpoint for AI")
    parser.add_argument("--output", type=Path, default=Path("runs/replays/verified-race.jsonl"))
    parser.add_argument("--swap-lanes", action="store_true")
    parser.add_argument("--rules", choices=("legacy", "recovery"), default="legacy")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default=None)
    args = parser.parse_args(argv)
    if args.rules == "legacy" and args.difficulty is not None:
        parser.error("--difficulty requires recovery rules")
    rules = RaceRules(difficulty=args.difficulty or "medium") if args.rules == "recovery" else None
    controllers = {car_id: ScriptedCar(car_id) for car_id in ("human", "ai")}
    label, digest = "Two scripted diagnostic controllers", None
    if args.model:
        from racing.agents.policy import PolicyController

        controllers["ai"] = PolicyController(project_path(args.model))
        digest = controllers["ai"].metadata["model_sha256"]
        label = str(args.model)
    session = RaceSession(Track.from_json("assets/tracks/demo.json"), controllers=controllers,
                          lane_swap=args.swap_lanes, rules=rules)
    output = project_path(args.output)
    recorder = ReplayRecorder(output, session, model_sha256=digest, model_label=label)
    session.start()
    while session.state == "COUNTDOWN":
        session.countdown_tick()
    while session.state == "RUNNING":
        session.step()
    if not recorder.closed:
        raise RuntimeError("Race did not finalize recording")
    replay = load_replay(output)
    if verification_state(replay.session) != verification_state(session):
        raise RuntimeError("Reconstructed race differs from driven result")
    results = plain(session.results)
    if rules is None:
        for value in results.values():
            value.pop("recovery_count")
            value.pop("recovery_held_ticks")
    report = dict(replay=str(args.output), replay_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  verified=True, strict=True, model_sha256=digest, model_label=label,
                  target_laps=session.config.target_laps, lane_swap=args.swap_lanes,
                  tick_count=session.snapshot().tick, winner=session.winner,
                  abort_reason=session.abort_reason, results=results,
                  final_state_sha256=state_hash(verification_state(session)),
                  verification_frames=len(replay.frames),
                  interpretation="Engineering lifecycle diagnostic on demo; not a policy generalization evaluation")
    if rules is not None:
        report["race_rules"] = plain(rules)
    output.with_suffix(".results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
