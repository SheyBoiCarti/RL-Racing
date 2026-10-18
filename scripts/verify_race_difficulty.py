"""Reproducible entertainment-race diagnostics on a declared public sample."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from racing.core.config import PhysicsConfig, project_path
from racing.race.config import RaceConfig, RaceRules
from racing.race.replay import ReplayRecorder, load_replay
from racing.race.replay_state import plain, state_hash, verification_state
from racing.race.session import RaceSession
from racing.tracks import Track
from racing.tracks.splits import load_split_catalog
from scripts.verify_race import ScriptedCar

TIERS = ("easy", "medium", "hard")


def load_policy(model):
    from racing.agents.policy import PolicyController
    return PolicyController(model)


def public_sample():
    tracks = [Track.from_json("assets/tracks/demo.json")]
    for split in ("train", "validation"):
        catalog = load_split_catalog("experiments/splits/v1.json", split)
        ids = sorted(catalog)
        if len(ids) < 3:
            raise ValueError(f"{split}: three public tracks required")
        tracks.extend(catalog[key] for key in ids[:3])
    return tracks


def run_race(track, controller, difficulty, path, seed):
    session = RaceSession(track, config=RaceConfig(countdown_seconds=0),
        rules=RaceRules(difficulty=difficulty), seed=seed,
        controllers={"human": ScriptedCar("human"), "ai": controller})
    recorder = ReplayRecorder(path, session, model_sha256=controller.metadata["model_sha256"],
                              model_label="Validated saved PPO; scripted privileged human diagnostic")
    session.start()
    while session.state == "RUNNING":
        session.step()
    if not recorder.closed or session.abort_reason:
        raise RuntimeError(f"incomplete diagnostic: {session.abort_reason}")
    replay = load_replay(path)
    if verification_state(replay.session) != verification_state(session):
        raise RuntimeError("incomplete diagnostic: replay divergence")
    return dict(tick_count=session.snapshot().tick, winner=session.winner,
        replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        final_state_sha256=state_hash(verification_state(session)), results=plain(session.results))


def run_diagnostics(model: Path, output: Path) -> dict:
    model, output = Path(model), Path(output)
    if output.exists():
        raise FileExistsError(f"Diagnostic evidence already exists: {output}")
    controller = load_policy(model)
    tracks = public_sample()
    if len(tracks) != 7 or any(track.split not in (None, "demo", "train", "validation") for track in tracks):
        raise ValueError("diagnostics: exactly seven public tracks required")
    output.mkdir(parents=True)
    config = RaceConfig(countdown_seconds=0)
    report = dict(complete=False, model=str(model), model_sha256=controller.metadata["model_sha256"],
        model_file_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), simulator_seed=0,
        race_config=plain(config), race_config_sha256=state_hash(plain(config)),
        physics_config=plain(PhysicsConfig()), human_acceptance=False,
        human_controller="ScriptedCar: privileged diagnostic controller, not human acceptance",
        sample=[dict(track_id=t.track_id, split=t.split or "demo",
                     geometry_sha256=state_hash(t.geometry_payload)) for t in tracks], races=[])
    csv_rows = []
    for track in tracks:
        for difficulty in TIERS:
            path = output / f"{track.track_id}-{difficulty}.jsonl"
            row = dict(track_id=track.track_id, split=track.split or "demo",
                track_geometry_sha256=state_hash(track.geometry_payload), difficulty=difficulty,
                race_rules=plain(RaceRules(difficulty=difficulty)), seed=0,
                model_sha256=report["model_sha256"], replay=path.name)
            try:
                result = run_race(track, controller, difficulty, path, 0)
                if set(result["results"]) != {"human", "ai"}:
                    raise RuntimeError("incomplete diagnostic: missing car results")
                row.update(result)
                for car_id, metrics in result["results"].items():
                    csv_rows.append(dict(track_id=track.track_id, split=row["split"],
                        track_geometry_sha256=row["track_geometry_sha256"], difficulty=difficulty,
                        rules="race-recovery-v1", model_sha256=row["model_sha256"], seed=0,
                        car_id=car_id, status=metrics["status"], reason=metrics["reason"],
                        completed_laps=metrics["completed_laps"], recovery_count=metrics["recovery_count"],
                        held_ticks=metrics["recovery_held_ticks"],
                        successful_lap_times_s=json.dumps(metrics["lap_times_s"]),
                        elapsed_race_ticks=result["tick_count"], replay_sha256=result["replay_sha256"]))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            report["races"].append(row)
            (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["tiers"] = {}
    for tier in TIERS:
        rows = [r for r in report["races"] if r["difficulty"] == tier and "error" not in r]
        ai = [r["results"]["ai"] for r in rows]
        laps = [time for car in ai for time in car["lap_times_s"]]
        report["tiers"][tier] = dict(races=len(rows), ai_finishes=sum(c["status"] == "finished" for c in ai),
            terminal_failures=sum(c["status"] in ("timed_out", "aborted", "disqualified") for c in ai),
            completed_laps=sum(c["completed_laps"] for c in ai), lap_time_coverage=len(laps),
            mean_successful_lap_s=mean(laps) if laps else None,
            recoveries=sum(c["recovery_count"] for c in ai),
            held_ticks=sum(c["recovery_held_ticks"] for c in ai))
    findings = []
    for easier, harder in zip(TIERS, TIERS[1:]):
        a, b = report["tiers"][easier], report["tiers"][harder]
        if a["terminal_failures"] > b["terminal_failures"]:
            findings.append(f"{easier} is less reliable than {harder}: more terminal failures")
        if a["mean_successful_lap_s"] is None or b["mean_successful_lap_s"] is None:
            findings.append(f"{easier}/{harder}: insufficient successful-lap coverage to judge pace separation")
        elif a["mean_successful_lap_s"] < b["mean_successful_lap_s"] * 1.02:
            findings.append(f"{easier}/{harder}: aggregate lap pace lacks useful ordered separation")
    report["findings"] = findings
    report["complete"] = all("error" not in r for r in report["races"]) and len(report["races"]) == 21
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if csv_rows:
        with (output / "races.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    if not report["complete"]:
        raise RuntimeError(f"incomplete diagnostics; see {output / 'inventory.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_diagnostics(project_path(args.model), project_path(args.output))
    print(json.dumps({"complete": report["complete"], "tiers": report["tiers"], "findings": report["findings"]}, indent=2))


if __name__ == "__main__":
    main()
