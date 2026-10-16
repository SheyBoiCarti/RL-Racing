"""Schema-valid model pairs and checkpoint-bound, hash-verified resume bundles."""

import copy
import os
import random
import shutil
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from racing.envs.config import load_rl
from racing.training.config import TrainConfig
from racing.training.evaluation import _plain, canonical_hash
from racing.training.manifest import (
    ROOT,
    contracts,
    dependencies,
    file_hash,
    read_json,
    versions,
    write_json,
)


def validate_metadata(model_path, runtime=None):
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Missing model or sidecar: {model_path}")
    data = read_json(metadata_path)
    schema_root = ROOT / "docs/specs/schemas"
    common = read_json(schema_root / "common.schema.json")
    registry = Registry().with_resource(common["$id"], Resource.from_contents(common))
    errors = list(Draft202012Validator(read_json(schema_root / "model.schema.json"), registry=registry).iter_errors(data))
    if errors:
        raise ValueError(f"{metadata_path}: invalid model schema: {errors[0].message}")
    if data["fixture_only"] or data["algorithm"] != "ppo":
        raise ValueError("Expected runnable PPO model, not a fixture or unsupported algorithm")
    if file_hash(model_path) != data["model_sha256"]:
        raise ValueError("model hash mismatch")
    try:
        new_fields = {"track_sampling", "split_manifest", "evaluation_split", "condition", "execution_profile_registration"}
        if any(field.name not in data["hyperparameters"] for field in fields(TrainConfig) if field.name not in new_fields):
            raise ValueError("missing required historical training field")
        training = TrainConfig(**{field.name: data["hyperparameters"][field.name] for field in fields(TrainConfig)
                                 if field.name in data["hyperparameters"]})
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"training: missing or invalid typed hyperparameters: {error}") from error
    typed_training = asdict(training)
    typed_training = {k: v for k, v in typed_training.items() if k in data["hyperparameters"]}
    if canonical_hash(typed_training) != data["config_hashes"]["training"]:
        raise ValueError("training: config hash mismatch")
    if training.seed != data["training_seed"] or data["hyperparameters"].get("activation_fn") != "torch.nn.Tanh" or data["hyperparameters"].get("optimizer_class") != "torch.optim.Adam":
        raise ValueError("training: incompatible resolved policy/seed")
    expected = contracts(runtime or load_rl())
    for name in ("physics", "observation", "action", "env"):
        value = data[f"{name}_config"]
        if canonical_hash(value) != data["config_hashes"][name]:
            raise ValueError(f"{name}: sidecar config hash mismatch")
        if name != "env" and value != expected[name]:
            raise ValueError(f"{name}: incompatible runtime semantics/order/version")
    expected_versions = versions(runtime or load_rl())
    if data["versions"] != expected_versions or data["normalization"] != "none":
        raise ValueError("incompatible model versions or normalization")
    if data["selection_metrics"]["checkpoint_step"] != data["checkpoint_step"] or data["selection_metrics"]["update_count"] != data["update_count"]:
        raise ValueError("selection identity differs from model")
    # A convenience pair must still match its inventory association, if present.
    if model_path.name in ("best_model.zip", "final_model.zip"):
        inventory_path = model_path.parent / "checkpoints/committed.json"
        if inventory_path.exists():
            inventory = read_json(inventory_path)
            reference = inventory.get(model_path.stem)
            if reference is None:
                raise ValueError("convenience model has no committed association")
            committed = model_path.parent / reference
            if file_hash(committed) != data["model_sha256"] or file_hash(committed.with_suffix(".metadata.json")) != file_hash(metadata_path):
                raise ValueError("convenience pair differs from committed model/sidecar")
    return data


def capture_rng(model):
    import torch

    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                workers=[copy.deepcopy(env.unwrapped.np_random.bit_generator.state) for env in model.env.envs])


def restore_rng(state, model=None):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if model is not None:
        for env, rng in zip(model.env.envs, state["workers"], strict=True):
            env.unwrapped.np_random.bit_generator.state = rng


def make_metadata(model, train_config, env_config, metrics, track_hash):
    configs = contracts(env_config)
    hyperparameters = _plain(train_config)
    hyperparameters.update(activation_fn="torch.nn.Tanh", optimizer_class="torch.optim.Adam",
                           optimizer_defaults=_plain(model.policy.optimizer.defaults))
    return dict(schema_version="model-v1", fixture_only=False, model_sha256="0" * 64,
                algorithm="ppo", policy="MlpPolicy", checkpoint_step=int(model.num_timesteps),
                update_count=metrics["update_count"], training_seed=train_config.seed,
                versions=versions(env_config), normalization="none",
                **{f"{k}_config": v for k, v in configs.items()},
                config_hashes={**{k: canonical_hash(v) for k, v in configs.items()}, "training": canonical_hash(train_config)},
                hyperparameters=hyperparameters, train_track_hashes=track_hash if isinstance(track_hash, list) else [track_hash], dependencies=dependencies(),
                selection_metrics={k: v for k, v in metrics.items() if k not in ("model_sha256", "path")})


def save_checkpoint(model, coordinator, metrics, *, selectable):
    """Publish complete pairs + bundles only; staged files never enter inventory."""
    import torch

    root = coordinator.run_dir
    relative = f"checkpoints/step_{model.num_timesteps:09d}_update_{coordinator.update_count:06d}"
    destination = root / relative
    if destination.exists():
        raise ValueError(f"immutable checkpoint already exists: {destination}")
    if shutil.disk_usage(root).free < 128 * 1024 * 1024:
        raise OSError("Insufficient free disk space for safe checkpoint publication")
    stage = destination.with_name("." + destination.name + ".staging")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        model.save(stage / "model.zip")
        metadata = make_metadata(model, coordinator.config, coordinator.env_config, metrics, coordinator.track_hash)
        metadata["model_sha256"] = file_hash(stage / "model.zip")
        write_json(stage / "model.metadata.json", metadata)
        validate_metadata(stage / "model.zip", coordinator.env_config)
        entry = dict(metrics, model_sha256=metadata["model_sha256"], path=f"{relative}/model.zip")
        if selectable:
            coordinator.ledger.append(entry)
        resume = stage / "resume"
        resume.mkdir()
        state = coordinator.state()
        state.update(checkpoint_step=int(model.num_timesteps), model_sha256=metadata["model_sha256"],
                     monitor=[dict(episode_returns=e.episode_returns, episode_lengths=e.episode_lengths,
                                   episode_times=e.episode_times, total_steps=e.total_steps) for e in model.env.envs])
        write_json(resume / "trainer_state.json", state)
        # Torch serialization preserves Python/NumPy/Torch RNG types. Only load
        # bundles produced by this project from a trusted source (like SB3 zips).
        torch.save(capture_rng(model), resume / "rng_state.pt")
        write_json(resume / "bundle.json", dict(checkpoint_step=int(model.num_timesteps), update_count=coordinator.update_count,
                   model_sha256=metadata["model_sha256"], metadata_sha256=file_hash(stage / "model.metadata.json"),
                   files={p.name: file_hash(p) for p in resume.iterdir()}))
        os.replace(stage, destination)
        inventory_path = root / "checkpoints/committed.json"
        inventory = read_json(inventory_path) if inventory_path.exists() else {"checkpoints": []}
        inventory["checkpoints"].append(entry)
        inventory["latest"] = entry["path"]
        write_json(inventory_path, inventory)
        return entry
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def resolve_checkpoint(path):
    path = Path(path).resolve()
    validate_metadata(path)
    if path.name in ("best_model.zip", "final_model.zip"):
        inventory_path = path.parent / "checkpoints/committed.json"
        if not inventory_path.is_file():
            raise ValueError("resume: missing committed inventory for convenience model")
        path = path.parent / read_json(inventory_path)[path.stem]
    return path


def load_bundle(path, _ancestors=None):
    path = resolve_checkpoint(path)
    metadata = validate_metadata(path)
    owner = path.parents[2]
    ancestors = set() if _ancestors is None else set(_ancestors)
    if owner.resolve() in ancestors:
        raise ValueError("resume ancestor cycle")
    ancestors.add(owner.resolve())
    inventory_path = owner / "checkpoints/committed.json"
    if not inventory_path.is_file():
        raise ValueError("resume: missing committed checkpoint inventory")
    inventory = read_json(inventory_path)
    matches = [entry for entry in inventory.get("checkpoints", []) if (owner / entry["path"]).resolve() == path.resolve()]
    if len(matches) != 1 or matches[0].get("model_sha256") != metadata["model_sha256"]:
        raise ValueError("resume: checkpoint absent or ambiguous in committed inventory")
    manifest = read_json(owner / "manifest.json")
    reference = manifest.get("parent_checkpoint")
    if reference:
        parent = (owner / reference).resolve()
        if not parent.is_relative_to(owner.resolve()) or parent == path.resolve() or not parent.is_file():
            raise ValueError("resume: missing or unsafe parent checkpoint")
        parent_manifest = parent.parents[2] / "manifest.json"
        if not parent_manifest.is_file() or file_hash(parent_manifest) != manifest.get("parent_manifest_sha256"):
            raise ValueError("resume: ancestor manifest hash mismatch")
        load_bundle(parent, _ancestors=ancestors)
    bundle_path = path.parent / "resume/bundle.json"
    if not bundle_path.is_file():
        raise ValueError(f"resume: missing bundle: {bundle_path}")
    bundle = read_json(bundle_path)
    for key in ("checkpoint_step", "update_count", "model_sha256"):
        if bundle.get(key) != metadata[key]:
            raise ValueError(f"resume bundle {key} mismatch")
    if bundle.get("metadata_sha256") != file_hash(path.with_suffix(".metadata.json")):
        raise ValueError("resume metadata hash mismatch")
    if not {"rng_state.pt", "trainer_state.json"} <= bundle.get("files", {}).keys():
        raise ValueError("resume bundle: missing state files")
    for name, digest in bundle["files"].items():
        target = (bundle_path.parent / name).resolve()
        if not target.is_relative_to(bundle_path.parent.resolve()) or not target.is_file() or file_hash(target) != digest:
            raise ValueError(f"resume bundle file hash/path mismatch: {name}")
    state = read_json(bundle_path.parent / "trainer_state.json")
    for key in ("checkpoint_step", "update_count", "model_sha256"):
        if state.get(key) != bundle[key]:
            raise ValueError(f"resume trainer {key} mismatch")
    if state["next_due"] <= state["checkpoint_step"]:
        raise ValueError("resume evaluation cadence is stale")
    seen = set()
    for candidate in state["ledger"]:
        if (candidate["checkpoint_step"], candidate["update_count"]) > (state["checkpoint_step"], state["update_count"]):
            raise ValueError("resume ledger contains a later checkpoint")
        model_file = path.parents[2] / candidate["path"]
        if not model_file.resolve().is_relative_to(path.parents[2].resolve()) or not model_file.is_file():
            raise ValueError("resume ledger has missing or unsafe ancestor model")
        candidate_metadata = validate_metadata(model_file)
        if candidate_metadata["model_sha256"] != candidate["model_sha256"]:
            raise ValueError("resume ledger model hash mismatch")
        if any(candidate.get(k) != v for k, v in candidate_metadata["selection_metrics"].items()):
            raise ValueError("resume ledger selection metrics mismatch")
        pair = (candidate["checkpoint_step"], candidate["update_count"])
        if pair in seen:
            raise ValueError("resume ledger has duplicate transition/update pair")
        seen.add(pair)
    return path, metadata, state


def publish_alias(root, name, checkpoint_path):
    root, source = Path(root), Path(root) / checkpoint_path
    # Pair replacement is individually atomic; the loader rejects a mixed pair.
    for suffix in (".zip", ".metadata.json"):
        destination = root / (name + suffix)
        temporary = root / ("." + name + suffix)
        shutil.copyfile(source.with_suffix(suffix), temporary)
        os.replace(temporary, destination)
    inventory_path = root / "checkpoints/committed.json"
    inventory = read_json(inventory_path)
    inventory[name] = checkpoint_path
    write_json(inventory_path, inventory)
