"""Headless baseline evaluator and versioned, atomically published artifacts.

Only finished attempts enter performance statistics. A profiling run's final
unfinished episode is reported separately and still counts toward throughput.
"""

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

EVALUATION_VERSION = "evaluation-v1"
ATTEMPT_COLUMNS = tuple(
    (
        "evaluation_version,agent_id,training_seed,track_id,"
        "track_hash,reset_seed,simulator_seed,agent_seed,success,reason,completed_laps,"
        "lap_time_s,progress_fraction,episode_steps,sim_duration_s,episode_return,"
        "wall_duration_s"
    ).split(",")
)


def _plain(value):
    if is_dataclass(value):
        result = {}
        for field in fields(value):
            child = getattr(value, field.name)
            if field.type in (float, "float"):
                if isinstance(child, bool):
                    raise ValueError(
                        f"{field.name}: boolean is not a numeric config value"
                    )
                child = float(child)
            result[field.name] = _plain(child)
        return result
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite artifact value")
        return 0.0 if value == 0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported artifact type: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(
        _plain(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def track_geometry(track):
    """Use original schema geometry; reprojection cannot recover exact stations."""
    return track.geometry_payload


def summarize(rows, *, stepping_duration_s, wall_duration_s, total_steps=None):
    count = len(rows)
    successful = [r for r in rows if r["success"]]
    laps = [r["lap_time_s"] for r in successful if r["lap_time_s"] is not None]
    off_track = sum(r["reason"] == "off_track" for r in rows)
    timeout = sum(r["reason"] in ("max_steps", "no_progress") for r in rows)
    steps = (
        sum(r["episode_steps"] for r in rows) if total_steps is None else total_steps
    )
    return dict(
        attempted=count,
        completed=len(successful),
        completion_rate=len(successful) / count if count else None,
        off_track_count=off_track,
        off_track_rate=off_track / count if count else None,
        timeout_count=timeout,
        timeout_rate=timeout / count if count else None,
        mean_progress_fraction=statistics.mean(r["progress_fraction"] for r in rows)
        if rows
        else None,
        successful_lap_count=len(laps),
        lap_time_median_s=statistics.median(laps) if laps else None,
        lap_time_iqr_s=float(
            np.quantile(laps, 0.75, method="linear")
            - np.quantile(laps, 0.25, method="linear")
        )
        if laps
        else None,
        lap_time_std_s=statistics.stdev(laps) if len(laps) > 1 else None,
        total_steps=steps,
        stepping_duration_s=stepping_duration_s,
        steps_per_second=steps / stepping_duration_s
        if stepping_duration_s > 0
        else None,
        wall_duration_s=wall_duration_s,
    )


def _atomic(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path, value):
    _atomic(path, canonical_json(value) + "\n")


def _write_csv(path, columns, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            "true"
            if row[k] is True
            else "false"
            if row[k] is False
            else ""
            if row[k] is None
            else _plain(row[k])
            for k in columns
        )
    _atomic(path, output.getvalue())


def validate_split_attempts(rows, track_ids, reset_seeds, *, deterministic):
    """Require exact paired coverage and reproducible deterministic outcomes."""
    expected = [(track_id, seed) for track_id in sorted(track_ids) for seed in reset_seeds]
    if [(row["track_id"], row["reset_seed"]) for row in rows] != expected:
        raise ValueError("validation attempt coverage mismatch")
    if deterministic:
        fields = ("success", "reason", "completed_laps", "lap_time_s", "progress_fraction",
                  "episode_steps", "episode_return", "sim_duration_s")
        reference = {}
        for row in rows:
            outcome = tuple(row[key] for key in fields)
            if reference.setdefault(row["track_id"], outcome) != outcome:
                raise ValueError(f"deterministic reproducibility failure: {row['track_id']}")


def evaluate_split(agent_id, *, split="validation", split_manifest, output,
                   episodes_per_track=3, seed=9000, reset_seeds=None,
                   config=None, agent=None, model=None):
    """Evaluate sorted held-out IDs explicitly; final tests remain embargoed."""
    from dataclasses import replace

    from racing.envs.config import load_rl
    from racing.envs.racing_env import RacingEnv
    if split != "validation":
        raise ValueError("Only validation inference is authorized; test requires Phase 6 frozen selection")
    from racing.tracks.splits import load_manifest, load_split_catalog

    manifest_data = load_manifest(split_manifest)
    catalog = load_split_catalog(split_manifest, split)
    if not catalog:
        raise ValueError("empty validation catalog")
    config = replace(config or load_rl(), allowed_split=split, track_sampling="fixed", split_manifest=None)
    if agent_id == "ppo" and agent is None:
        from racing.agents.policy import PolicyController

        if model is None:
            raise ValueError("ppo evaluation requires --model")
        agent = PolicyController(model, config=config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, tracks, stepping = [], [], 0.
    metadata = dict(evaluation_version=EVALUATION_VERSION, status="running", split=split,
                    split_manifest_sha256=manifest_data["manifest_sha256"],
                    track_ids=sorted(catalog), reset_seeds=reset_seeds or list(range(seed, seed + episodes_per_track)),
                    reproducibility_note="Deterministic reset repeats are not independent trials.")
    _write_json(output / "manifest.json", metadata)
    try:
        for track_id in sorted(catalog):
            folder = output / track_id
            summary = evaluate(agent_id, track_id=track_id, episodes=episodes_per_track,
                               seed=seed, reset_seeds=reset_seeds, output=folder,
                               config=replace(config, default_track_id=track_id), agent=agent, model=model,
                               env_factory=lambda config: RacingEnv(config=config, track_catalog=catalog))
            with (folder / "attempts.csv").open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    rows.append({key: (row[key] == "true" if key == "success" else
                                      None if row[key] == "" else
                                      float(row[key]) if key in ("lap_time_s", "progress_fraction", "episode_return", "sim_duration_s", "wall_duration_s") else
                                      int(row[key]) if key in ("training_seed", "reset_seed", "simulator_seed", "agent_seed", "completed_laps", "episode_steps") else row[key])
                                 for key in ATTEMPT_COLUMNS})
            tracks.append(dict(track_id=track_id, track_hash=canonical_hash(track_geometry(catalog[track_id])), **summary))
            stepping += summary["stepping_duration_s"]
        validate_split_attempts(rows, sorted(catalog), reset_seeds if reset_seeds is not None
                                else list(range(seed, seed + episodes_per_track)), deterministic=agent_id != "random")
        summary = summarize(rows, stepping_duration_s=stepping, wall_duration_s=time.perf_counter() - started)
        summary.update(macro_completion=statistics.mean(t["completion_rate"] for t in tracks),
                       macro_progress=statistics.mean(t["mean_progress_fraction"] for t in tracks),
                       track_count=len(tracks), mode="validation")
        _write_csv(output / "attempts.csv", ATTEMPT_COLUMNS, rows)
        columns = ("track_id", "track_hash", "attempted", "completed", "completion_rate", "off_track_count", "timeout_count",
                   "mean_progress_fraction", "successful_lap_count", "lap_time_median_s", "lap_time_iqr_s", "lap_time_std_s",
                   "stepping_duration_s", "wall_duration_s")
        _write_csv(output / "tracks.csv", columns, tracks)
        _write_json(output / "summary.json", summary)
        metadata.update(status="completed", artifact_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                        for name in ("attempts.csv", "tracks.csv", "summary.json")})
        _write_json(output / "manifest.json", metadata)
        return summary
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(output / "manifest.json", metadata)
        raise


def evaluate(
    agent_id,
    *,
    track_id="demo",
    episodes=None,
    seed=9000,
    output,
    config=None,
    profile_steps=None,
    agent=None,
    env_factory=None,
    model=None,
    reset_seeds=None,
):
    """Evaluate complete fixed-track attempts, or an exact profiling step budget."""
    from racing.agents.random import RandomAgent
    from racing.agents.scripted import ScriptedAgent
    from racing.envs.config import load_rl
    from racing.envs.racing_env import RacingEnv
    from racing.tracks import schema as track_schema

    if agent_id not in ("random", "scripted", "ppo"):
        raise ValueError("agent_id: expected random, scripted or ppo")
    if agent_id == "ppo" and agent is None:
        from racing.agents.policy import PolicyController

        if model is None:
            raise ValueError("ppo evaluation requires --model")
        agent = PolicyController(model, config=config)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed: expected uint32")
    if profile_steps is not None and episodes is not None:
        raise ValueError("profile_steps is mutually exclusive with episodes")
    if profile_steps is not None and (
        type(profile_steps) is not int or profile_steps <= 0
    ):
        raise ValueError("profile_steps: expected positive integer")
    episodes = 20 if episodes is None else episodes
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes: expected positive integer")
    if seed + (profile_steps or episodes) - 1 >= 2**32:
        raise ValueError("attempt seeds exceed uint32")
    if reset_seeds is not None and (
        profile_steps is not None or len(reset_seeds) != episodes
        or len(set(reset_seeds)) != episodes
        or any(type(s) is not int or not 0 <= s < 2**32 for s in reset_seeds)
    ):
        raise ValueError("reset_seeds: exact distinct episode seed coverage required")
    started = time.perf_counter()
    config = config if config is not None else load_rl()
    agent = (
        agent
        if agent is not None
        else (
            RandomAgent()
            if agent_id == "random"
            else ScriptedAgent(physics=config.physics)
        )
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        evaluation_version=EVALUATION_VERSION,
        status="running",
        agent_id=agent_id,
        agent_settings=_plain(getattr(agent, "config", {})),
        privileged_track_access=agent_id == "scripted",
        reproducibility_note="Fixed-track deterministic scripted attempts are duplicates, "
        "not independent evidence of robustness.",
        config=_plain(config),
        config_hash=canonical_hash(config),
        physics_hash=canonical_hash(config.physics),
        track_hashes={},
        base_reset_seed=seed,
        requested_episodes=None if profile_steps else episodes,
        requested_profile_steps=profile_steps,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_versions={
            name: version(name) for name in ("numpy", "gymnasium", "PyYAML")
        },
        schema_versions=dict(
            track="track-v1",
            physics="physics-v1",
            observation=config.observation_version,
            action=config.action_version,
            reward=config.reward_version,
        ),
        schema_hashes={
            "track": hashlib.sha256(
                Path(track_schema.__file__).with_name("track.schema.json").read_bytes()
            ).hexdigest()
        },
        agent_settings_hash=canonical_hash(getattr(agent, "config", {})),
    )
    if agent_id == "ppo":
        manifest["model_metadata"] = getattr(agent, "metadata", None)
        manifest["reproducibility_note"] = "Fixed-track deterministic attempts are duplicates, not independent samples."
    manifest["reset_seeds"] = reset_seeds
    _write_json(output / "manifest.json", manifest)
    env = None
    rows, partial, step_duration, total_steps = [], None, 0.0, 0
    try:
        env = (env_factory or RacingEnv)(config=config)
        index = 0
        while (
            total_steps < profile_steps
            if profile_steps is not None
            else index < episodes
        ):
            attempt_started = time.perf_counter()
            attempt_seed = reset_seeds[index] if reset_seeds is not None else seed + index
            observation, info = env.reset(
                seed=attempt_seed, options={"track_id": track_id}
            )
            agent.reset(attempt_seed)
            geometry_hash = canonical_hash(track_geometry(env.track))
            manifest["track_hashes"][track_id] = geometry_hash
            episode_return, episode_steps, finished = 0.0, 0, False
            while not finished and (
                profile_steps is None or total_steps < profile_steps
            ):
                control = agent.act(observation, env.snapshot, env.track)
                action = np.asarray(
                    [control.steering, control.longitudinal], dtype=np.float32
                )
                before = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                step_duration += time.perf_counter() - before
                total_steps += 1
                episode_steps += 1
                episode_return += float(reward)
                finished = terminated or truncated
            row = dict(
                evaluation_version=EVALUATION_VERSION,
                agent_id=agent_id,
                training_seed=getattr(agent, "training_seed", None),
                track_id=track_id,
                track_hash=geometry_hash,
                reset_seed=attempt_seed,
                simulator_seed=int(info["simulator_seed"]),
                agent_seed=attempt_seed if agent_id == "random" else None,
                success=bool(info["success"]) if finished else False,
                reason=info["episode_reason"] if finished else "profile_budget",
                completed_laps=int(info["completed_laps"]),
                lap_time_s=float(info["lap_time_s"])
                if finished and info["success"] and info["lap_time_s"] is not None
                else None,
                progress_fraction=float(info["progress_fraction"]),
                episode_steps=episode_steps,
                sim_duration_s=float(env.snapshot.time_s),
                episode_return=episode_return,
                wall_duration_s=time.perf_counter() - attempt_started,
            )
            if finished:
                rows.append(row)
            else:
                partial = row
            index += 1
        expected = reset_seeds if reset_seeds is not None else list(range(seed, seed + len(rows)))
        if [r["reset_seed"] for r in rows] != expected or any(
            r["track_id"] != track_id for r in rows
        ):
            raise ValueError("attempt coverage mismatch")
        summary = summarize(
            rows,
            stepping_duration_s=step_duration,
            wall_duration_s=time.perf_counter() - started,
            total_steps=total_steps,
        )
        summary.update(
            evaluation_version=EVALUATION_VERSION,
            partial_episode=partial,
            mode="profile" if profile_steps is not None else "evaluation",
        )
        _write_csv(output / "attempts.csv", ATTEMPT_COLUMNS, rows)
        track_summary = dict(track_id=track_id, track_hash=geometry_hash, **summary)
        track_columns = (
            "track_id",
            "track_hash",
            "attempted",
            "completed",
            "completion_rate",
            "off_track_count",
            "timeout_count",
            "mean_progress_fraction",
            "successful_lap_count",
            "lap_time_median_s",
            "lap_time_iqr_s",
        )
        _write_csv(output / "tracks.csv", track_columns, [track_summary])
        env.close()
        env = None
        summary["wall_duration_s"] = time.perf_counter() - started
        _write_json(output / "summary.json", summary)
        manifest["status"] = "completed"
        manifest["artifact_sha256"] = {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("attempts.csv", "tracks.csv", "summary.json")
        }
        _write_json(output / "manifest.json", manifest)
        return summary
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        _write_json(output / "manifest.json", manifest)
        raise
    finally:
        if env is not None:
            env.close()
