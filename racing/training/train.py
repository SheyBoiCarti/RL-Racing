"""Reproducible fixed-track PPO orchestration and episode-reset continuation."""

import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from racing.envs.config import load_rl
from racing.training.callbacks import (
    RESUMABLE_STOP,
    ControlledStop,
    Coordinator,
    EpisodeMetrics,
)
from racing.training.checkpoints import (
    load_bundle,
    publish_alias,
    restore_rng,
)
from racing.training.config import ExecutionConfig, resume_identity
from racing.training.evaluation import canonical_hash, track_geometry
from racing.training.manifest import (
    ROOT,
    contracts,
    dependencies,
    file_hash,
    now,
    read_json,
    source_identity,
    versions,
    write_json,
)


def train(config, run_dir, *, execution=None, resume=None, env_config=None, stop_requested=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from racing.envs.racing_env import RacingEnv
    from racing.training.algorithms import PostUpdatePPO

    execution = execution or ExecutionConfig(device="cuda" if config.device == "cuda" else "cpu")
    if config.device != "auto" and execution.device != config.device:
        raise ValueError("training/execution device mismatch")
    if execution.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    env_config = env_config or load_rl()
    env_config = replace(env_config, target_laps=config.target_laps, default_track_id=config.track_id,
                         track_sampling=config.track_sampling, split_manifest=config.split_manifest if config.track_sampling == "uniform_train" else None,
                         allowed_split="train" if config.track_sampling == "uniform_train" else "demo")
    split_hash = None
    if config.split_manifest:
        from racing.tracks.splits import load_manifest

        split_hash = load_manifest(config.split_manifest)["manifest_sha256"]
    worker_seeds = ([int(child.generate_state(1, dtype=np.uint32)[0])
                     for child in np.random.SeedSequence(config.seed).spawn(config.n_envs)]
                    if config.evaluation_split else [config.seed])
    profile = dict(platform=execution.platform, device=execution.device, torch_threads=execution.torch_threads,
                   python_version=dependencies()["python"], packages=dependencies()["packages"], source_sha256=source_identity())
    lock_hash = file_hash(ROOT / ("requirements-kaggle.lock.txt" if execution.platform == "kaggle" else "requirements-rl.lock.txt"))
    if config.evaluation_split and execution.platform == "kaggle" and not config.execution_profile_registration:
        raise ValueError("Phase 4 Kaggle requires execution_profile_registration")
    registration = None
    if config.execution_profile_registration:
        registration = read_json(config.execution_profile_registration)
        if registration != dict(profile, lockfile_sha256=lock_hash):
            raise ValueError("registered execution profile/source/lock mismatch")
    run_dir = Path(run_dir).resolve()
    if execution.platform == "kaggle" and not run_dir.is_relative_to(Path(execution.output_root).resolve()):
        raise ValueError("Kaggle run-dir must be inside execution output_root")
    if run_dir.exists():
        raise ValueError(f"run-dir already exists: {run_dir}")
    parent, parent_metadata, saved_state = None, None, None
    if resume is not None:
        parent, parent_metadata, saved_state = load_bundle(resume)
        if parent_metadata["checkpoint_step"] >= config.total_timesteps:
            raise ValueError("resume requires final absolute total greater than saved transitions")
        previous_config = {k: parent_metadata["hyperparameters"].get(k, v) for k, v in asdict(config).items()}
        if {k: v for k, v in previous_config.items() if k not in ("total_timesteps", "execution_profile_registration")} != resume_identity(config):
            raise ValueError("resume hyperparameters/seed/worker/track mismatch")
        if {k: parent_metadata[f"{k}_config"] for k in contracts(env_config)} != contracts(env_config):
            raise ValueError("resume environment/physics/reward mismatch")
        parent_manifest = read_json(parent.parents[2] / "manifest.json")
        if config.evaluation_split and parent_manifest["requested_transitions"] != config.total_timesteps:
            raise ValueError("Phase 4 resume cannot change preregistered transition budget/candidates")
        if parent_manifest.get("split_manifest_sha256") != split_hash:
            raise ValueError("resume split manifest hash mismatch")
        if parent_manifest.get("execution_profile_registration") != registration:
            raise ValueError("resume registered execution profile mismatch")
        if parent_manifest["execution_profile"] != dict(
            platform=execution.platform, device=execution.device, torch_threads=execution.torch_threads,
            python_version=dependencies()["python"], packages=dependencies()["packages"],
            source_sha256=source_identity()):
            raise ValueError("resume execution profile/runtime/source mismatch")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics").mkdir()
    (run_dir / "tracks").mkdir()
    starting = parent_metadata["checkpoint_step"] if parent else 0
    manifest = dict(run_id=run_dir.name, status="running", started_at=now(), finished_at=None,
                    requested_transitions=config.total_timesteps, actual_transitions=starting,
                    starting_transitions=starting, additional_transitions=config.total_timesteps - starting,
                    training_config=asdict(config), config_hash=canonical_hash(config),
                    env_config=asdict(env_config), config_hashes={k: canonical_hash(v) for k, v in contracts(env_config).items()},
                    execution=asdict(execution), dependency_versions=dependencies(), versions=versions(env_config),
                    worker_seeds=worker_seeds, evaluation_seeds=config.eval_seeds, selection_rule="selection-v1",
                    split_manifest_sha256=split_hash, execution_profile_registration=registration,
                    resume_mode="episode_reset" if parent else None, parent_checkpoint=None,
                    execution_profile=dict(platform=execution.platform, device=execution.device,
                        torch_threads=execution.torch_threads, python_version=dependencies()["python"],
                        packages=dependencies()["packages"], source_sha256=source_identity()),
                    lockfile_sha256=file_hash(ROOT / ("requirements-kaggle.lock.txt" if execution.platform == "kaggle" else "requirements-rl.lock.txt")),
                    artifacts=dict(checkpoints="checkpoints/committed.json", train="metrics/train.csv", evaluations="metrics/evaluations.csv"),
                    hardware=dict(cpu=__import__("platform").processor(), cuda=torch.version.cuda,
                                  gpu=torch.cuda.get_device_name(0) if execution.device == "cuda" else None))
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(dict(training=asdict(config), environment=asdict(env_config), execution=asdict(execution))), encoding="utf-8")
    vector = None
    coordinator, model = None, None
    try:
        torch.set_num_threads(execution.torch_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        vector = DummyVecEnv([lambda: Monitor(RacingEnv(config=env_config),
                     info_keywords=("success", "progress_fraction", "episode_reason"))])
        vector.seed(worker_seeds[0])
        catalog = vector.envs[0].unwrapped.track_catalog
        track_hashes = []
        for track_id in sorted(catalog):
            entry = catalog[track_id]
            track = entry[0] if isinstance(entry, tuple) else entry
            track_hashes.append(canonical_hash(track_geometry(track)))
            write_json(run_dir / f"tracks/{track_id}.json", track.geometry_payload)
        track_hash = track_hashes if config.evaluation_split else track_hashes[0]
        manifest.update(train_track_hashes=track_hashes)
        write_json(run_dir / "manifest.json", manifest)
        coordinator = Coordinator(config, env_config, run_dir, track_hash, execution, stop_requested)
        if parent:
            # Keep a self-contained immutable ancestor tree. References always
            # resolve relative to the owning run, even after an export moves.
            owner = parent.parents[2]
            prefix = f"parents/{file_hash(owner / 'manifest.json')[:16]}"
            shutil.copytree(owner, run_dir / prefix, ignore=shutil.ignore_patterns("*.staging", ".*staging"))
            for candidate in saved_state["ledger"]:
                candidate["path"] = f"{prefix}/{candidate['path']}"
            coordinator.restore(saved_state)
            manifest.update(parent_checkpoint=f"{prefix}/{parent.relative_to(owner).as_posix()}",
                            parent_manifest_sha256=file_hash(owner / "manifest.json"),
                            parent_status=read_json(owner / "manifest.json")["status"])
            write_json(run_dir / "manifest.json", manifest)
            model = PostUpdatePPO.load(parent, env=vector, device=execution.device, tensorboard_log=str(run_dir / "tensorboard"))
            import torch as _torch

            rng = _torch.load(parent.parent / "resume/rng_state.pt", map_location="cpu", weights_only=False)
            restore_rng(rng, model)
            for monitor, state in zip(vector.envs, saved_state["monitor"], strict=True):
                for name, value in state.items():
                    setattr(monitor, name, value)
            # Prevent SB3's seeded reset from replacing the restored worker stream.
            vector._reset_seeds()
        else:
            model = PostUpdatePPO(config.policy, vector, seed=config.seed, device=execution.device,
                n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
                learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
                clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                policy_kwargs={**config.policy_kwargs, "activation_fn": torch.nn.Tanh},
                tensorboard_log=str(run_dir / "tensorboard"), verbose=1)
            vector.seed(worker_seeds[0])
            coordinator.evaluate(model, diagnostic=True)
        model.evaluation_coordinator = coordinator
        model.learn(total_timesteps=config.total_timesteps - starting,
                    callback=EpisodeMetrics(coordinator), reset_num_timesteps=not bool(parent),
                    tb_log_name="ppo")
        # after_update always evaluates final trained weights; never save a
        # pre-update callback snapshot as the final policy.
        publish_alias(run_dir, "final_model", coordinator.last_checkpoint["path"])
        manifest = read_json(run_dir / "manifest.json")
        manifest.update(status="completed", finished_at=now(), actual_transitions=int(model.num_timesteps))
        write_json(run_dir / "manifest.json", manifest)
        return 0
    except BaseException as error:
        manifest = read_json(run_dir / "manifest.json")
        manifest.update(status="interrupted" if isinstance(error, (ControlledStop, KeyboardInterrupt)) else "failed",
                        finished_at=now(), error=f"{type(error).__name__}: {error}")
        if model is not None:
            manifest["actual_transitions"] = int(model.num_timesteps)
        write_json(run_dir / "manifest.json", manifest)
        if isinstance(error, ControlledStop):
            return RESUMABLE_STOP
        if isinstance(error, KeyboardInterrupt):
            return 130
        raise
    finally:
        if vector is not None:
            vector.close()
