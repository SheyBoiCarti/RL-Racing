"""Validation cannot hide incomplete coverage or inconsistent deterministic repeats."""

import json
from copy import deepcopy
from pathlib import Path

import pytest


def rows():
    return [dict(track_id=track, reset_seed=seed, success=False, reason="max_steps",
                 completed_laps=0, lap_time_s=None, progress_fraction=.2,
                 episode_steps=1200, episode_return=2., sim_duration_s=60.)
            for track in ("validation_000", "validation_001") for seed in (9000, 9001)]


def test_validation_repeats_and_exact_coverage():
    from racing.training.evaluation import validate_split_attempts

    attempts = rows()
    validate_split_attempts(attempts, ["validation_000", "validation_001"], [9000, 9001], deterministic=True)
    changed = deepcopy(attempts)
    changed[1]["progress_fraction"] = .3
    with pytest.raises(ValueError, match="reproducibility"):
        validate_split_attempts(changed, ["validation_000", "validation_001"], [9000, 9001], deterministic=True)
    validate_split_attempts(changed, ["validation_000", "validation_001"], [9000, 9001], deterministic=False)
    with pytest.raises(ValueError, match="coverage"):
        validate_split_attempts(attempts[:-1], ["validation_000", "validation_001"], [9000, 9001], deterministic=True)


@pytest.mark.parametrize("extra", [
    ["--split", "validation", "--track", "demo"],
    ["--split", "validation", "--episodes", "2"],
    ["--split", "validation", "--profile-steps", "20"],
    ["--episodes-per-track", "2"],
])
def test_split_cli_rejects_ambiguous_protocol_at_parse_time(extra):
    from scripts.evaluate import parser

    with pytest.raises(SystemExit):
        parser().parse_args(["--agent", "random", "--output", "unused", *extra])


@pytest.mark.parametrize("track_id", ["../../escaped", "train_000/escape", "validation_000", "train_999"])
def test_manifest_track_ids_cannot_escape_or_misrepresent_split(tmp_path, track_id):
    from racing.tracks.splits import _read_manifest, manifest_hash

    data = json.loads(Path("experiments/splits/v1.json").read_text())
    data["entries"][0]["track_id"] = track_id
    data["manifest_sha256"] = manifest_hash(data)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="track_id"):
        _read_manifest(manifest)


def test_matching_track_and_manifest_unsafe_id_is_rejected():
    from racing.tracks.splits import _validated_content

    manifest = json.loads(Path("experiments/splits/v1.json").read_text())
    entry = manifest["entries"][0]
    track = json.loads(Path(entry["path"]).read_text())
    entry["track_id"] = track["track_id"] = "../../escaped"
    with pytest.raises(ValueError, match="track_id"):
        _validated_content(json.dumps(track), json.dumps(manifest["generator_config"]),
                           json.dumps(entry), manifest["generator_config_hash"])
