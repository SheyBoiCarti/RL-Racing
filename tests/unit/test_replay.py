"""Record/reconstruct actual race actions, including between-tick events."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from racing.core.models import CarControl
from racing.race.config import RaceConfig
from racing.race.replay import ReplayRecorder, load_replay, verification_state
from racing.race.session import RaceSession
from racing.tracks import Track


@pytest.fixture(scope="module")
def track():
    return Track.from_json("assets/tracks/demo.json")


def recorded(track, path, *, timeout=4, lane_swap=False, rules=None):
    race = RaceSession(track, config=RaceConfig(countdown_seconds=0, race_max_steps=timeout),
                       lane_swap=lane_swap, rules=rules)
    recorder = ReplayRecorder(path, race)
    race.start()
    return race, recorder


def write_records(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_timeout_roundtrip_schema_hashes_and_no_model(track, tmp_path):
    path = tmp_path / "race.jsonl"
    race, recorder = recorded(track, path, timeout=104, lane_swap=True)
    for _ in range(104):
        race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    assert recorder.closed
    playback = load_replay(path)
    assert playback.session.snapshot() == race.snapshot()
    assert verification_state(playback.session) == verification_state(race)
    assert playback.session.winner == "no_winner"
    assert len(playback.frames) == 105
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["fixture_only"] is False
    assert records[0]["hash_interval"] == 100
    assert "verification_state" in records[100]
    assert records[-1]["record_type"] == "footer"
    assert records[-1]["tick_count"] == 104
    assert load_replay(path, tolerant=True).session.snapshot() == race.snapshot()


@pytest.mark.parametrize("rules_enabled", [False, True])
@pytest.mark.parametrize("pause", [False, True])
@pytest.mark.parametrize("reason", ["inference_error", "user_exit", "restart"])
def test_zero_tick_and_paused_abort(track, tmp_path, pause, reason, rules_enabled):
    path = tmp_path / "abort.jsonl"
    from racing.race.config import RaceRules
    race, recorder = recorded(track, path, rules=RaceRules() if rules_enabled else None)
    if pause:
        race.pause()
    race.abort(reason)
    assert recorder.closed
    replay = load_replay(path)
    assert replay.session.snapshot().tick == 0
    assert replay.session.abort_reason == reason
    assert verification_state(replay.session) == verification_state(race)


def test_pause_resume_does_not_advance_and_reconstructs(track, tmp_path):
    path = tmp_path / "paused.jsonl"
    race, _ = recorded(track, path, timeout=2)
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    race.pause()
    race.resume()
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    assert load_replay(path).session.snapshot() == race.snapshot()


@pytest.mark.parametrize("corrupt", ["action", "seq", "tick", "hash", "footer", "version", "config", "spawns"])
def test_corruption_rejected(track, tmp_path, corrupt):
    path = tmp_path / "original.jsonl"
    race, _ = recorded(track, path)
    for _ in range(4):
        race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if corrupt == "action":
        records[1]["controls"]["human"]["longitudinal"] = .9
    elif corrupt == "seq":
        records[2]["seq"] += 1
    elif corrupt == "tick":
        records.pop(1)
        for i, record in enumerate(records):
            record["seq"] = i
    elif corrupt == "hash":
        records[-1]["final_state_sha256"] = "0" * 64
    elif corrupt == "footer":
        records.pop()
    elif corrupt == "version":
        records[0]["versions"]["physics"] = "physics-v99"
    elif corrupt == "config":
        records[0]["physics_config"]["max_speed_mps"] += 1
    else:
        records[0]["spawns"][1] = copy.deepcopy(records[0]["spawns"][0])
    bad = tmp_path / "corrupt.jsonl"
    write_records(bad, records)
    with pytest.raises(ValueError):
        load_replay(bad)


def test_missing_scheduled_snapshot_and_illegal_events_rejected(track, tmp_path):
    path = tmp_path / "race.jsonl"
    race, _ = recorded(track, path, timeout=101)
    for _ in range(101):
        race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[100].pop("verification_state")
    records[100].pop("state_sha256")
    write_records(path, records)
    with pytest.raises(ValueError, match="scheduled"):
        load_replay(path)


def test_tolerant_checks_stored_float_but_exact_discrete_fields(track, tmp_path):
    from racing.race.replay import state_hash

    path = tmp_path / "race.jsonl"
    race, _ = recorded(track, path, timeout=1)
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    state = records[-1]["final_state"]
    car = state["world"]["cars"][0]
    car["state"]["x_m"] = (float.fromhex(car["state"]["x_m"]) + 1e-8).hex()
    records[-1]["final_state_sha256"] = state_hash(state)
    write_records(path, records)
    with pytest.raises(ValueError, match="x_m"):
        load_replay(path)
    load_replay(path, tolerant=True)
    state["world"]["cars"][0]["completed_laps"] += 1
    records[-1]["final_state_sha256"] = state_hash(state)
    write_records(path, records)
    with pytest.raises(ValueError, match="completed_laps"):
        load_replay(path, tolerant=True)


def test_reject_duplicate_json_fields_and_nonfinite(track, tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"seq":0,"seq":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_replay(path)
    path.write_text('{"seq":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_replay(path)


def test_recorder_never_overwrites_and_rejects_mid_race_attach(track, tmp_path):
    path = tmp_path / "race.jsonl"
    race, recorder = recorded(track, path)
    with pytest.raises((ValueError, FileExistsError)):
        ReplayRecorder(path, race)
    race.abort("user_exit")
    assert recorder.closed
    race = RaceSession(track, config=replace(RaceConfig(), countdown_seconds=0))
    race.start()
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    with pytest.raises(ValueError, match="tick|start"):
        ReplayRecorder(tmp_path / "late.jsonl", race)


def test_embedded_track_retains_generator_provenance(tmp_path):
    from racing.tracks.schema import read_track

    source_path = next(Path("assets/tracks/generated").glob("train*.json"))
    data = read_track(source_path)
    track = Track(data)
    path = tmp_path / "generated.jsonl"
    race, _ = recorded(track, path, timeout=1)
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    header = json.loads(path.read_text().splitlines()[0])
    assert header["track"]["generator"] == data["generator"]
    assert header["track"] == data
    load_replay(path)


@pytest.mark.parametrize("mutation", ["duplicate_pause", "unmatched_resume", "tick_paused", "after_abort", "event_tick"])
def test_illegal_session_order(track, tmp_path, mutation):
    path = tmp_path / "events.jsonl"
    race, _ = recorded(track, path)
    race.pause()
    race.resume()
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    race.abort("user_exit")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate_pause":
        records.insert(2, copy.deepcopy(records[1]))
    elif mutation == "unmatched_resume":
        records.pop(1)
    elif mutation == "tick_paused":
        records.pop(2)
    elif mutation == "after_abort":
        records.insert(-1, copy.deepcopy(records[3]))
    else:
        records[1]["after_tick"] = 1
    for index, record in enumerate(records):
        record["seq"] = index
    write_records(path, records)
    with pytest.raises(ValueError):
        load_replay(path)


def test_tolerance_does_not_apply_to_discrete_hex_looking_labels():
    from racing.race.replay_state import compare_state

    with pytest.raises(ValueError, match="track_id"):
        compare_state({"track_id": "0x1.0000000000001p+0"}, {"track_id": "0x1.0000000000000p+0"},
                      tick=0, tolerant=True)


def test_hex_float_overflow_is_clear_divergence():
    from racing.race.replay_state import compare_state

    with pytest.raises(ValueError, match="time_s"):
        compare_state({"time_s": "0x1.0p+999999"}, {"time_s": "0x1.0p+0"}, tick=0, tolerant=True)


def test_playback_imports_no_training_or_optional_rl_library(track, tmp_path):
    path = tmp_path / "portable.jsonl"
    race, _ = recorded(track, path, timeout=1)
    race.step({"human": CarControl(0., .1), "ai": CarControl(0., .1)})
    script = """
import sys
sys.modules.update(gymnasium=None, torch=None, stable_baselines3=None)
from racing.race.replay import load_replay
replay = load_replay(sys.argv[1])
assert replay.session.snapshot().tick == 1
assert 'racing.agents.policy' not in sys.modules
from racing.core.config import load_game
from racing.render.race import RaceApplication
from racing.render.race_view import RaceView
import pygame
pygame.init()
app = RaceApplication(load_game(), None, replay_path=sys.argv[1])
app.debug = True
RaceView(pygame.Surface((480,360)), app.game, app.physics).draw(app)
pygame.quit()
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True,
                            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"})
    assert result.returncode == 0, result.stderr


def test_legacy_retained_race_still_matches_original_hash():
    from racing.core.config import project_path
    from racing.race.replay_state import state_hash
    replay = load_replay(project_path("docs/specs/fixtures/automated-race.jsonl"))
    assert replay.session.snapshot().tick == 1656
    assert len(replay.frames) == 1657
    assert replay.session.winner == "ai"
    assert state_hash(verification_state(replay.session)) == (
        "95ff4f10de28772b7ed80fc42baec80bf11371ba3b7a3647598aabdee1442c90")


def recovery_recording(track, path):
    from racing.race.config import RaceRules
    race = RaceSession(track, config=RaceConfig(countdown_seconds=0), rules=RaceRules())
    recorder = ReplayRecorder(path, race)
    race.start()
    for _ in range(200):
        race.step({"human": CarControl(1, 1), "ai": CarControl(0, 0)})
        if race.recovery_by_car["human"].count:
            break
    assert race.recovery_by_car["human"].count == 1
    race.pause()
    race.resume()
    for _ in range(21):
        race.step({"human": CarControl(0, 0), "ai": CarControl(0, 0)})
    for _ in range(100):
        active = [c.car_id for c in race.snapshot().cars if c.status.value == "active"]
        race.step({key: CarControl(1, 1) if key == "human" else CarControl(0, 0)
                   for key in active})
        if race.recovery_by_car["human"].count == 2:
            break
    assert race.recovery_by_car["human"].count == 2
    race.abort("user_exit")
    assert recorder.closed
    return race


def test_recovery_v2_roundtrip(track, tmp_path):
    from racing.race.replay_state import plain, state_hash
    path = tmp_path / "recovery.jsonl"
    race = recovery_recording(track, path)
    replay = load_replay(path)
    assert replay.header["schema_version"] == "replay-v2"
    assert replay.header["race_rules"] == plain(race.rules)
    assert replay.header["race_rules_sha256"] == state_hash(plain(race.rules))
    assert verification_state(replay.session) == verification_state(race)
    recovery = [frame.recovery_by_car["human"] for frame in replay.frames]
    assert any(r.remaining_ticks == 20 and r.held_ticks == 0 for r in recovery)
    assert any(r.remaining_ticks == 0 and r.held_ticks == 20 and
               r.last_release_tick is not None for r in recovery)
    assert recovery[0].count == 0
    with pytest.raises(TypeError):
        replay.frames[-1].recovery_by_car["human"] = recovery[0]


@pytest.mark.parametrize("mutation", ["missing_rules", "misspelled_rules", "extra_version",
    "rules_hash", "safe_anchor", "remaining", "release_future", "held_control",
    "missing_footer", "mixed_version", "state_hash"])
def test_recovery_v2_corruption(track, tmp_path, mutation):
    from racing.race.replay_state import state_hash
    path = tmp_path / "corrupt-v2.jsonl"
    recovery_recording(track, path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    header = records[0]
    if mutation == "missing_rules":
        header["versions"].pop("rules")
    elif mutation == "misspelled_rules":
        header["versions"]["rule"] = header["versions"].pop("rules")
    elif mutation == "extra_version":
        header["versions"]["extra"] = "unknown"
    elif mutation == "rules_hash":
        header["race_rules_sha256"] = "0" * 64
    elif mutation in ("safe_anchor", "remaining", "release_future"):
        recovery = header["initial_state"]["session"]["recovery_by_car"]["human"]
        if mutation == "safe_anchor":
            recovery["safe"]["state"]["x_m"] = float(100000).hex()
        elif mutation == "remaining":
            recovery["remaining_ticks"] = 21
        else:
            recovery["last_release_tick"] = 100000
        header["initial_state_sha256"] = state_hash(header["initial_state"])
    elif mutation == "held_control":
        pause_index = next(i for i, r in enumerate(records) if r.get("event") == "pause")
        records[pause_index + 2]["controls"]["human"]["longitudinal"] = 1.
    elif mutation == "missing_footer":
        records.pop()
    elif mutation == "mixed_version":
        records[1]["schema_version"] = "replay-v1"
    else:
        records[-1]["final_state_sha256"] = "0" * 64
    write_records(path, records)
    with pytest.raises(ValueError):
        load_replay(path)


@pytest.mark.parametrize("rules_enabled", [False, True])
def test_playback_does_not_import_optional_rl(track, tmp_path, rules_enabled):
    path = tmp_path / "ml-free.jsonl"
    if rules_enabled:
        recovery_recording(track, path)
    else:
        race, _ = recorded(track, path)
        race.abort("user_exit")
    code = '''
import sys
class BlockRL:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "stable_baselines3"):
            raise RuntimeError("Optional ML import: " + fullname)
sys.meta_path.insert(0, BlockRL())
from racing.race.replay import load_replay
replay = load_replay(sys.argv[1])
assert replay.session.state == "RESULTS"
'''
    completed = subprocess.run([sys.executable, "-c", code, str(path)],
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_recovery_failure_records_tick_before_footer(track, tmp_path, monkeypatch):
    from racing.core.simulation import Simulation
    from racing.race.config import RaceRules
    def failure(*args, **kwargs):
        raise ValueError("invalid safe anchor")
    monkeypatch.setattr(Simulation, "recover_car", failure)
    race = RaceSession(track, rules=RaceRules(), config=RaceConfig(countdown_seconds=0))
    path = tmp_path / "fault.jsonl"
    recorder = ReplayRecorder(path, race)
    race.start()
    for _ in range(200):
        if race.state != "RUNNING":
            break
        race.step({"human": CarControl(1, 1), "ai": CarControl(0, 0)})
    assert race.abort_reason == "recovery_error" and recorder.closed
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["record_type"] for r in records[-2:]] == ["tick", "footer"]
    assert not any(r["record_type"] == "session_event" for r in records)
    assert verification_state(load_replay(path).session) == verification_state(race)
