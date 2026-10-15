"""Artifact and counting contracts for headless evaluation."""

import csv
import hashlib
import json

import pytest


def test_canonical_hash_normalizes_signed_zero_and_rejects_nonfinite():
    from racing.core.config import PhysicsConfig
    from racing.training.evaluation import canonical_hash

    assert canonical_hash({"x": -0.0}) == canonical_hash({"x": 0.0})
    assert canonical_hash(PhysicsConfig(max_speed_mps=22)) == canonical_hash(
        PhysicsConfig()
    )
    with pytest.raises(ValueError):
        canonical_hash({"x": float("nan")})


def test_summary_null_and_linear_quartiles():
    from racing.training.evaluation import summarize

    rows = [
        dict(
            success=False,
            reason="no_progress",
            progress_fraction=0.2,
            lap_time_s=None,
            episode_steps=3,
        )
    ]
    result = summarize(rows, stepping_duration_s=2.0, wall_duration_s=4.0)
    assert result["lap_time_median_s"] is None
    assert result["lap_time_iqr_s"] is None
    assert result["steps_per_second"] == 1.5
    rows = [
        dict(
            success=True,
            reason="success",
            progress_fraction=1.0,
            lap_time_s=t,
            episode_steps=3,
        )
        for t in [1.0, 2.0, 5.0, 10.0]
    ]
    result = summarize(rows, stepping_duration_s=2.0, wall_duration_s=4.0)
    assert result["lap_time_median_s"] == 3.5
    assert result["lap_time_iqr_s"] == 4.5


def test_real_evaluation_columns_seeds_and_no_overwrite(tmp_path):
    from racing.envs.config import RLConfig
    from racing.training.evaluation import evaluate

    output = tmp_path / "run"
    result = evaluate(
        "random", episodes=2, seed=42, output=output, config=RLConfig(max_steps=2)
    )
    with (output / "attempts.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == (
            "evaluation_version,agent_id,training_seed,track_id,track_hash,reset_seed,"
            "simulator_seed,agent_seed,success,reason,completed_laps,lap_time_s,"
            "progress_fraction,episode_steps,sim_duration_s,episode_return,wall_duration_s"
        ).split(",")
        rows = list(reader)
    assert [r["reset_seed"] for r in rows] == ["42", "43"]
    assert [r["agent_seed"] for r in rows] == ["42", "43"]
    assert all(r["success"] == "false" and r["lap_time_s"] == "" for r in rows)
    assert result["attempted"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        evaluate("random", episodes=1, output=output)


def test_profile_exact_budget_excludes_partial_attempt(tmp_path):
    from racing.envs.config import RLConfig
    from racing.training.evaluation import evaluate

    result = evaluate(
        "random",
        profile_steps=5,
        output=tmp_path / "profile",
        config=RLConfig(max_steps=2),
    )
    assert result["total_steps"] == 5
    assert result["attempted"] == 2
    assert result["partial_episode"]["episode_steps"] == 1
    assert result["partial_episode"]["reason"] == "profile_budget"


@pytest.mark.parametrize(
    "args",
    [
        ["--episodes", "1", "--profile-steps", "5"],
        ["--split", "test"],
        ["--model", "missing.zip"],
        ["--episodes-per-track", "2"],
        ["--selection-manifest", "missing.json"],
    ],
)
def test_cli_rejects_incompatible_or_future_phase_options(args):
    from scripts.evaluate import parser

    with pytest.raises(SystemExit) as error:
        parser().parse_args(["--agent", "random", "--output", "unused", *args])
    assert error.value.code == 2


def test_track_hash_is_exact_source_geometry(demo_track):
    from racing.core.config import project_path
    from racing.training.evaluation import canonical_hash, track_geometry

    source = json.loads(project_path("assets/tracks/demo.json").read_text())
    payload = {
        k: source[k]
        for k in ("schema_version", "centerline", "width_m", "checkpoints", "spawns")
    }
    assert canonical_hash(track_geometry(demo_track)) == canonical_hash(payload)


@pytest.mark.parametrize(
    "failure,status",
    [(RuntimeError("broken"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_failure_manifest_and_environment_cleanup(tmp_path, failure, status):
    from racing.training.evaluation import evaluate

    class BrokenEnvironment:
        closed = False

        def __init__(self, config):
            pass

        def reset(self, **kwargs):
            raise failure

        def close(self):
            type(self).closed = True

    output = tmp_path / status
    with pytest.raises(type(failure)):
        evaluate("random", output=output, env_factory=BrokenEnvironment)
    assert BrokenEnvironment.closed
    assert json.loads((output / "manifest.json").read_text())["status"] == status
