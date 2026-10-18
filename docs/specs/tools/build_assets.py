"""Deterministically build specification schemas/examples, not application artifacts."""
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ai-racing.local/specs/"


def canonical(value):
    def normalize_zero(item):
        if isinstance(item, float) and item == 0:
            return 0.0
        if isinstance(item, dict):
            return {k: normalize_zero(v) for k, v in item.items()}
        if isinstance(item, list):
            return [normalize_zero(v) for v in item]
        return item
    return json.dumps(normalize_zero(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n", encoding="utf-8")


def obj(properties, optional=()):
    return {"type": "object", "properties": properties,
            "required": [k for k in properties if k not in optional],
            "additionalProperties": False}


def arr(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


def ref(name):
    return {"$ref": BASE + "common.schema.json#/$defs/" + name}


def nullable(value):
    return {"anyOf": [value, {"type": "null"}]}


def enum(*values):
    return {"enum": list(values)}


def schema(name, body):
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": BASE + name + ".schema.json", **body}


N = {"type": "number"}
F = {"type": "number", "minimum": 0}
POS = {"type": "number", "exclusiveMinimum": 0}
I = {"type": "integer", "minimum": 0}
PI = {"type": "integer", "minimum": 1}
B = {"type": "boolean"}
S = {"type": "string", "minLength": 1}
H = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
U32 = {"type": "integer", "minimum": 0, "maximum": 4294967295}
RATE = {"type": "number", "minimum": 0, "maximum": 1}
CONTROL = {"type": "number", "minimum": -1, "maximum": 1}
HEX = {"type": "string", "pattern": r"^-?0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+$"}
MAPPING = lambda value: {"type": "object", "additionalProperties": value}

physics = obj({"dt_s": POS, "max_speed_mps": POS, "max_steer_rad": POS,
               "wheelbase_m": POS, "acceleration_mps2": POS,
               "braking_mps2": POS, "rolling_drag_mps2": F,
               "steering_slew_radps": POS, "car_radius_m": POS})
observation = obj({"version": enum("obs-rays3-v1", "obs-rays7-v1", "obs-rays15-v1"),
                   "ray_angles_deg": arr(N, 3), "ray_range_m": POS,
                   "dtype": enum("float32"), "dimension": PI})
action = obj({"version": enum("action-v1"), "dimension": {"const": 2},
              "dtype": enum("float32"), "order": {"const": ["steering", "longitudinal"]}})
env = obj({"target_laps": PI, "max_steps": PI, "no_progress_steps": PI,
           "progress_threshold_m": F, "reward_version": enum("reward-v1", "reward-no-steer-v1"),
           "steering_penalty": F, "progress_coefficient": POS, "step_cost": F,
           "off_track_penalty": {"type": "number", "maximum": 0}, "lap_bonus": F})
deps = obj({"python": S, "platform": S, "packages": MAPPING(S),
            "git_commit": nullable(S)})
versions = obj({k: S for k in ["physics", "track", "observation", "action", "reward", "evaluation", "selection"]})
metrics = obj({"macro_completion": RATE, "macro_progress": RATE, "all_success": B,
               "macro_mean_lap_time_s": nullable(POS), "checkpoint_step": I,
               "update_count": I})
state = obj({k: (F if k == "speed_mps" else N) for k in
             ["x_m", "y_m", "heading_rad", "speed_mps", "steering_rad"]})
projection = obj({"segment_index": I, "segment_t": RATE, "s_m": F,
                  "lateral_m": N, "tangent_heading": N, "distance_m": F})
car = obj({"car_id": S, "state": ref("car_state"),
           "status": enum("active", "finished", "disqualified", "timed_out", "aborted"),
           "projection": ref("projection"), "projection_discontinuity": B,
           "signed_delta_m": N, "rewardable_delta_m": F, "net_travel_m": N,
           "high_water_m": F, "approach_distance_m": F, "completed_laps": I,
           "finish_armed": B, "next_checkpoint": I, "lap_start_tick": nullable(I),
           "completed_lap_times_s": arr(POS)})
world = obj({"tick": I, "time_s": F, "track_id": S, "cars": arr(ref("car_snapshot"), 1)})
session = obj({"no_progress_by_car": MAPPING(I), "finish_tick_by_car": MAPPING(nullable(I)),
               "session_state": enum("RUNNING", "PAUSED", "RESULTS"),
               "winner": nullable(enum("human", "ai", "draw", "no_winner")),
               "abort_reason": nullable(enum("inference_error", "user_exit", "restart"))})
defs = {"physics_config": physics, "observation_config": observation, "action_config": action,
        "env_config": env, "dependencies": deps, "versions": versions,
        "selection_metrics": metrics, "car_state": state, "projection": projection,
        "car_snapshot": car, "world_snapshot": world, "session_state": session}


def hexify(node):
    if isinstance(node, list):
        return [hexify(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node and node["$ref"].startswith(BASE + "common.schema.json#/$defs/"):
        return hexify(defs[node["$ref"].split("/")[-1]])
    if node.get("type") == "number":
        return deepcopy(HEX)
    return {key: hexify(value) for key, value in node.items()}


def build_recovery_assets():
    """Extend retained v1 contracts without rewriting any historical assets."""
    old_common = json.loads((ROOT / "schemas/common.schema.json").read_text())
    old_replay = json.loads((ROOT / "schemas/replay.schema.json").read_text())
    rules = obj({"version": {"const": "race-recovery-v1"},
                 "difficulty": enum("easy", "medium", "hard"),
                 "recovery_hold_ticks": {"const": 20, "type": "integer"},
                 "pace_adapter": {"const": "pace-governor-v1"}})
    recovery = obj({"safe": old_common["$defs"]["verification_state"]["properties"]["world"]["properties"]["cars"]["items"],
                    "remaining_ticks": {"type": "integer", "minimum": 0, "maximum": 20},
                    "count": I, "held_ticks": I, "last_release_tick": nullable(I)})
    verification = deepcopy(old_common["$defs"]["verification_state"])
    verification["properties"]["session"]["properties"]["race_rules"] = rules
    verification["properties"]["session"]["properties"]["recovery_by_car"] = obj({key: recovery for key in ("ai", "human")})
    verification["properties"]["session"]["required"] += ["race_rules", "recovery_by_car"]
    verification["properties"]["session"]["properties"]["abort_reason"] = nullable(
        enum("inference_error", "user_exit", "restart", "recovery_error"))
    versions_v2 = deepcopy(old_common["$defs"]["versions"])
    versions_v2["properties"]["rules"] = {"const": "race-recovery-v1"}
    versions_v2["required"].append("rules")
    common = schema("common-v2", {"$defs": {"race_rules": rules,
        "recovery_state": recovery, "verification_state": verification,
        "versions": versions_v2}})
    def rewrite(node):
        if isinstance(node, list):
            return [rewrite(v) for v in node]
        if not isinstance(node, dict):
            return node
        result = {key: rewrite(value) for key, value in node.items()}
        if result.get("$ref") in (ref("verification_state")["$ref"], ref("versions")["$ref"]):
            result["$ref"] = result["$ref"].replace("common.schema", "common-v2.schema")
        if result.get("enum") == ["inference_error", "user_exit", "restart"]:
            result["enum"].append("recovery_error")
        return result
    replay = rewrite(old_replay)
    replay["$id"] = BASE + "replay-v2.schema.json"
    header = replay["oneOf"][0]
    header["properties"]["schema_version"] = {"const": "replay-v2"}
    header["properties"]["race_rules"] = {"$ref": BASE + "common-v2.schema.json#/$defs/race_rules"}
    header["properties"]["race_rules_sha256"] = H
    header["required"] += ["race_rules", "race_rules_sha256"]
    write("schemas/common-v2.schema.json", common)
    write("schemas/replay-v2.schema.json", replay)
    records = [json.loads(line) for line in (ROOT / "examples/replay_abort.jsonl").read_text().splitlines()]
    rules_value = dict(version="race-recovery-v1", difficulty="medium", recovery_hold_ticks=20,
                       pace_adapter="pace-governor-v1")
    safe = {c["car_id"]: c for c in records[0]["initial_state"]["world"]["cars"]}
    for record in records:
        for key in ("initial_state", "verification_state", "final_state"):
            if key in record:
                value = record[key]
                value["session"]["race_rules"] = rules_value
                value["session"]["recovery_by_car"] = {car_id: dict(safe=car, remaining_ticks=0,
                    count=0, held_ticks=0, last_release_tick=None) for car_id, car in safe.items()}
                digest_key = {"initial_state": "initial_state_sha256",
                              "verification_state": "state_sha256",
                              "final_state": "final_state_sha256"}[key]
                record[digest_key] = sha(value)
    records[0].update(schema_version="replay-v2", race_rules=rules_value,
                      race_rules_sha256=sha(rules_value))
    records[0]["versions"]["rules"] = "race-recovery-v1"
    target = ROOT / "fixtures/replay_v2/abort.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(canonical(r).decode() + "\n" for r in records), encoding="utf-8")


if "--recovery-only" in sys.argv:
    build_recovery_assets()
    raise SystemExit(0)


defs["verification_state"] = obj({"world": hexify(world), "session": session})
write("schemas/common.schema.json", schema("common", {"$defs": defs}))

track = obj({"schema_version": enum("track-v1"), "track_id": S, "name": S,
             "split": enum("demo", "train", "validation", "test"), "width_m": POS,
             "centerline": arr({"type": "array", "items": N, "minItems": 2, "maxItems": 2}, 8),
             "checkpoints": arr(obj({"index": I, "s_m": F}), 4),
             "spawns": arr(obj({"spawn_id": S, "s_m": F, "lateral_m": N}), 3),
             "generator": nullable(obj({"version": S, "seed": U32, "attempt": I, "config_hash": H}))})
write("schemas/track.schema.json", schema("track", track))
model = obj({"schema_version": enum("model-v1"), "fixture_only": B,
             "model_sha256": H, "algorithm": enum("ppo", "sac"), "policy": enum("MlpPolicy"),
             "checkpoint_step": I, "update_count": I, "training_seed": U32,
             "versions": ref("versions"), "physics_config": ref("physics_config"),
             "observation_config": ref("observation_config"), "action_config": ref("action_config"),
             "env_config": ref("env_config"), "normalization": enum("none"),
             "config_hashes": obj({k: H for k in ["physics", "observation", "action", "env", "training"]}),
             "hyperparameters": MAPPING({}), "train_track_hashes": arr(H, 1),
             "dependencies": ref("dependencies"), "selection_metrics": ref("selection_metrics")})
write("schemas/model.schema.json", schema("model", model))
entry = obj({"track_id": S, "split": enum("train", "validation", "test"), "seed": U32,
             "attempt": I, "path": S, "geometry_hash": H, "length_m": POS, "max_curvature": F})
split = obj({"schema_version": enum("split-v1"), "fixture_only": B, "generator_version": S,
             "generator_config": MAPPING({}), "generator_config_hash": H,
             "counts": obj({k: PI for k in ["train", "validation", "test"]}),
             "entries": arr(entry, 3), "manifest_sha256": H})
write("schemas/split.schema.json", schema("split", split))
condition = obj({"condition_id": enum("ppo_multi_7", "sac_multi_7", "ppo_multi_3", "ppo_multi_15",
                                      "ppo_multi_no_steer_penalty", "ppo_single_7"),
                 "algorithm": enum("ppo", "sac"), "track_sampling": enum("fixed_demo", "uniform_train"),
                 "observation_version": S, "reward_version": S, "config_path": S, "config_sha256": H})
protocol = enum("full", "reduced", "smoke")
evaluation = obj({"version": enum("evaluation-v1"), "selection_version": enum("selection-v1"),
                  "eval_every_steps": PI, "base_reset_seed": U32, "episodes_per_track": PI,
                  "deterministic": {"const": True}, "evaluator_sha256": H, "config_sha256": H})
baseline = obj({"agent_id": enum("random", "scripted"), "config": MAPPING({}), "config_sha256": H})
experiment = obj({"schema_version": enum("experiment-v1"), "fixture_only": B, "protocol": protocol,
                  "training_seeds": {**arr(U32, 1), "uniqueItems": True},
                  "requested_final_timesteps": PI, "n_envs": PI, "split_manifest_sha256": H,
                  "conditions": {**arr(condition, 6), "maxItems": 6}, "evaluation": evaluation,
                  "baselines": {**arr(baseline, 2), "maxItems": 2},
                  "runs": arr(obj({"condition_id": S, "training_seed": U32, "run_path": S}), 6)})
write("schemas/experiment.schema.json", schema("experiment", experiment))
selected = obj({"condition_id": S, "training_seed": U32, "model_path": S, "metadata_path": S,
                "model_sha256": H, "metadata_sha256": H, "config_sha256": H,
                "checkpoint_step": PI, "update_count": PI,
                "selection_metrics": ref("selection_metrics")})
selection = obj({"schema_version": enum("selection-v1"), "fixture_only": B, "protocol": protocol,
                 "experiment_sha256": H, "split_manifest_sha256": H, "evaluation": evaluation,
                 "models": arr(selected, 6), "baselines": {**arr(baseline, 2), "maxItems": 2},
                 "manifest_sha256": H})
write("schemas/selection.schema.json", schema("selection", selection))
race_config = obj({"target_laps": PI, "countdown_seconds": F, "race_max_steps": PI,
                   "no_progress_steps": PI, "progress_threshold_m": F, "ghost": {"const": True}})
header = obj({"record_type": enum("header"), "seq": {"const": 0}, "schema_version": enum("replay-v1"),
              "fixture_only": B, "physics_config": ref("physics_config"), "race_config": race_config,
              "versions": ref("versions"), "config_hashes": obj({"physics": H, "race": H}),
              "track": {"$ref": BASE + "track.schema.json"}, "track_geometry_sha256": H,
              "simulator_seed": U32, "spawns": arr(obj({"car_id": S, "spawn_id": S}), 2),
              "model_sha256": nullable(H), "model_label": nullable(S), "dependencies": ref("dependencies"),
              "hash_interval": PI, "initial_state": ref("verification_state"), "initial_state_sha256": H})
tick = obj({"record_type": enum("tick"), "seq": PI, "tick": PI,
            "controls": MAPPING(obj({"steering": CONTROL, "longitudinal": CONTROL})),
            "verification_state": ref("verification_state"), "state_sha256": H},
           optional=("verification_state", "state_sha256"))
tick["dependentRequired"] = {"verification_state": ["state_sha256"], "state_sha256": ["verification_state"]}
external = obj({"record_type": enum("session_event"), "seq": PI, "after_tick": I,
                "event": enum("abort", "pause", "resume"),
                "reason": nullable(enum("inference_error", "user_exit", "restart")),
                "verification_state": ref("verification_state"), "state_sha256": H},
               optional=("verification_state", "state_sha256"))
external["dependentRequired"] = deepcopy(tick["dependentRequired"])
footer = obj({"record_type": enum("footer"), "seq": PI, "status": enum("completed", "aborted"),
              "tick_count": I, "winner": enum("human", "ai", "draw", "no_winner"),
              "abort_reason": nullable(enum("inference_error", "user_exit", "restart")),
              "final_state": ref("verification_state"), "final_state_sha256": H})
write("schemas/replay.schema.json", schema("replay", {"oneOf": [header, tick, external, footer]}))


def make_track(scale=1.0, track_id="reference_demo", split_name="demo"):
    # Frozen hand-designed harmonic source; this is NOT the Phase 4 generator.
    theta = np.arange(4096) * (2 * math.pi / 4096)
    radius = scale * (90 + 14 * np.cos(3 * theta))
    dense = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    edges = np.roll(dense, -1, axis=0) - dense
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    count = math.ceil(cumulative[-1] / 4)
    wanted = np.arange(count) * cumulative[-1] / count
    idx = np.searchsorted(cumulative, wanted, side="right") - 1
    points = dense[idx] + ((wanted-cumulative[idx])/lengths[idx])[:, None] * edges[idx]
    points = np.round(points, 6)
    total = float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).sum())
    gates = math.ceil(total / 60)
    return {"schema_version": "track-v1", "track_id": track_id, "name": "Reference fixture (manual)",
            "split": split_name, "width_m": 12.0, "centerline": points.tolist(),
            "checkpoints": [{"index": i, "s_m": float(i * total/gates)} for i in range(gates)],
            "spawns": [{"spawn_id": sid, "s_m": float(total-5), "lateral_m": lateral}
                       for sid, lateral in [("solo", 0.0), ("grid_left", -1.5), ("grid_right", 1.5)]],
            "generator": None}


def track_hash(track_value):
    return sha({k: track_value[k] for k in ["schema_version", "centerline", "width_m", "checkpoints", "spawns"]})


reference = make_track()
write("fixtures/reference_demo.json", reference)
entries = []
for scale, split_name, seed in [(1.0, "train", 10000), (1.03, "validation", 20000), (.97, "test", 30000)]:
    t = make_track(scale, "fixture_" + split_name, split_name)
    path = "fixtures/" + t["track_id"] + ".json"
    write(path, t)
    points = np.array(t["centerline"])
    edges = np.roll(points, -1, axis=0)-points
    lens = np.linalg.norm(edges, axis=1)
    headings = np.arctan2(edges[:, 1], edges[:, 0])
    turns = (headings-np.roll(headings, 1)+math.pi) % (2*math.pi)-math.pi
    entries.append({"track_id": t["track_id"], "split": split_name, "seed": seed, "attempt": 0,
                    "path": path, "geometry_hash": track_hash(t), "length_m": float(lens.sum()),
                    "max_curvature": float(np.max(np.abs(turns)/((lens+np.roll(lens, 1))/2)))})
split_example = {"schema_version": "split-v1", "fixture_only": True,
                 "generator_version": "manual-fixture-v1", "generator_config": {"source": "harmonic_reference", "dense_samples": 4096},
                 "counts": {"train": 1, "validation": 1, "test": 1}, "entries": entries}
split_example["generator_config_hash"] = sha(split_example["generator_config"])
split_example["manifest_sha256"] = sha(split_example)
write("examples/split.json", split_example)

physics_config = dict(zip(physics["properties"], [.05, 22.0, .55, 2.6, 6.0, 10.0, .25, 2.5, .8]))
obs_config = {"version": "obs-rays7-v1", "ray_angles_deg": [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0],
              "ray_range_m": 40.0, "dtype": "float32", "dimension": 11}
action_config = {"version": "action-v1", "dimension": 2, "dtype": "float32", "order": ["steering", "longitudinal"]}
env_config = dict(zip(env["properties"], [1, 1200, 120, .01, "reward-v1", .01, 2.0, .002, -10.0, 50.0]))
dependency_example = {"python": "fixture-runtime", "platform": "fixture-platform", "packages": {"purpose": "specification-only"}, "git_commit": None}
versions_example = dict(zip(versions["properties"], ["physics-v1", "track-v1", "obs-rays7-v1", "action-v1", "reward-v1", "evaluation-v1", "selection-v1"]))
metric = {"macro_completion": .5, "macro_progress": .9, "all_success": False,
          "macro_mean_lap_time_s": None, "checkpoint_step": 2048, "update_count": 1}
model_bytes = b"SPECIFICATION FIXTURE ONLY - NOT AN SB3 MODEL\n"
(ROOT / "fixtures/model_bytes.fixture").write_bytes(model_bytes)
model_hash = hashlib.sha256(model_bytes).hexdigest()
hyperparameters = {"learning_rate": .0003, "n_steps": 2048, "batch_size": 64, "n_epochs": 10}
model_example = {"schema_version": "model-v1", "fixture_only": True, "model_sha256": model_hash,
                 "algorithm": "ppo", "policy": "MlpPolicy", "checkpoint_step": 2048, "update_count": 1,
                 "training_seed": 42, "versions": versions_example, "physics_config": physics_config,
                 "observation_config": obs_config, "action_config": action_config, "env_config": env_config,
                 "normalization": "none", "config_hashes": {"physics": sha(physics_config), "observation": sha(obs_config),
                 "action": sha(action_config), "env": sha(env_config), "training": sha(hyperparameters)},
                 "hyperparameters": hyperparameters, "train_track_hashes": [entries[0]["geometry_hash"]],
                 "dependencies": dependency_example, "selection_metrics": metric}
write("examples/model.json", model_example)
evaluation_example = {"version": "evaluation-v1", "selection_version": "selection-v1", "eval_every_steps": 50000,
                      "base_reset_seed": 9000, "episodes_per_track": 3, "deterministic": True,
                      "evaluator_sha256": hashlib.sha256(b"fixture evaluator").hexdigest(),
                      "config_sha256": sha({"deterministic": True, "target_laps": 1})}
baselines = [{"agent_id": a, "config": {"profile": "fixture"}, "config_sha256": sha({"profile": "fixture"})}
             for a in ["random", "scripted"]]
ids = ["ppo_multi_7", "sac_multi_7", "ppo_multi_3", "ppo_multi_15", "ppo_multi_no_steer_penalty", "ppo_single_7"]
condition_models = {}
for cid in ids:
    variant = deepcopy(model_example)
    variant["algorithm"] = "sac" if cid.startswith("sac") else "ppo"
    if cid.endswith("_3") or cid.endswith("_15"):
        count = 3 if cid.endswith("_3") else 15
        variant["observation_config"].update(version=f"obs-rays{count}-v1",
                                              ray_angles_deg=np.linspace(-90, 90, count).tolist(), dimension=count+4)
        variant["versions"]["observation"] = variant["observation_config"]["version"]
    if cid.endswith("penalty"):
        variant["env_config"].update(reward_version="reward-no-steer-v1", steering_penalty=0.0)
        variant["versions"]["reward"] = "reward-no-steer-v1"
    if variant["algorithm"] == "sac":
        variant["hyperparameters"] = {"learning_rate": .0003, "learning_starts": 128,
                                       "buffer_size": 4096, "batch_size": 64, "gradient_steps": 1}
    for key in ["physics", "observation", "action", "env"]:
        variant["config_hashes"][key] = sha(variant[key+"_config"])
    variant["config_hashes"]["training"] = sha(variant["hyperparameters"])
    path = "examples/models/"+cid+".json"
    write(path, variant)
    condition_models[cid] = variant
conditions = [{"condition_id": cid, "algorithm": "sac" if cid.startswith("sac") else "ppo",
               "track_sampling": "fixed_demo" if cid == "ppo_single_7" else "uniform_train",
               "observation_version": "obs-rays3-v1" if cid.endswith("_3") else "obs-rays15-v1" if cid.endswith("_15") else "obs-rays7-v1",
               "reward_version": "reward-no-steer-v1" if cid.endswith("penalty") else "reward-v1",
               "config_path": "examples/models/"+cid+".json", "config_sha256": sha(condition_models[cid])} for cid in ids]
experiment_example = {"schema_version": "experiment-v1", "fixture_only": True, "protocol": "smoke", "training_seeds": [42],
                      "requested_final_timesteps": 2048, "n_envs": 1, "split_manifest_sha256": split_example["manifest_sha256"],
                      "conditions": conditions, "evaluation": evaluation_example, "baselines": baselines,
                      "runs": [{"condition_id": cid, "training_seed": 42, "run_path": "fixture-unexecuted/" + cid} for cid in ids]}
write("examples/experiment.json", experiment_example)
selection_example = {"schema_version": "selection-v1", "fixture_only": True, "protocol": "smoke",
                     "experiment_sha256": sha(experiment_example), "split_manifest_sha256": split_example["manifest_sha256"],
                     "evaluation": evaluation_example, "baselines": baselines,
                     "models": [{"condition_id": cid, "training_seed": 42, "model_path": "fixtures/model_bytes.fixture",
                     "metadata_path": "examples/models/"+cid+".json", "model_sha256": model_hash, "metadata_sha256": sha(condition_models[cid]),
                     "config_sha256": sha(condition_models[cid]), "checkpoint_step": 2048, "update_count": 1,
                     "selection_metrics": metric} for cid in ids]}
selection_example["manifest_sha256"] = sha(selection_example)
write("examples/selection.json", selection_example)

# Replay fixture is an actual zero-tick external-abort case, not a fabricated race.
points = np.array(reference["centerline"])
edges = np.roll(points, -1, axis=0)-points
lens = np.linalg.norm(edges, axis=1)
cumulative = np.r_[0.0, np.cumsum(lens)]


def initial_car(cid, lateral):
    s = float(cumulative[-1]-5)
    index = int(np.searchsorted(cumulative, s, side="right")-1)
    t = float((s-cumulative[index])/lens[index])
    tangent = edges[index]/lens[index]
    normal = np.array([-tangent[1], tangent[0]])
    pose = points[index]+t*edges[index]+lateral*normal
    heading = float(math.atan2(tangent[1], tangent[0]))
    return {"car_id": cid, "state": {"x_m": float(pose[0]), "y_m": float(pose[1]), "heading_rad": heading,
            "speed_mps": 0.0, "steering_rad": 0.0}, "status": "active", "projection": {"segment_index": index,
            "segment_t": t, "s_m": s, "lateral_m": lateral, "tangent_heading": heading, "distance_m": abs(lateral)},
            "projection_discontinuity": False, "signed_delta_m": 0.0, "rewardable_delta_m": 0.0,
            "net_travel_m": 0.0, "high_water_m": 0.0, "approach_distance_m": 5.0,
            "completed_laps": 0, "finish_armed": False, "next_checkpoint": 1, "lap_start_tick": None,
            "completed_lap_times_s": []}


def encode_floats(value):
    if isinstance(value, float):
        return (0.0 if value == 0 else value).hex()
    if isinstance(value, dict):
        return {k: encode_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode_floats(v) for v in value]
    return value


initial = encode_floats({"world": {"tick": 0, "time_s": 0.0, "track_id": reference["track_id"],
                        "cars": [initial_car("ai", 1.5), initial_car("human", -1.5)]},
                        "session": {"no_progress_by_car": {"ai": 0, "human": 0},
                        "finish_tick_by_car": {"ai": None, "human": None}, "session_state": "RUNNING",
                        "winner": None, "abort_reason": None}})
race_config_example = {"target_laps": 3, "countdown_seconds": 3.0, "race_max_steps": 7200,
                       "no_progress_steps": 120, "progress_threshold_m": .01, "ghost": True}
header_example = {"record_type": "header", "seq": 0, "schema_version": "replay-v1", "fixture_only": True,
                  "physics_config": physics_config, "race_config": race_config_example, "versions": versions_example,
                  "config_hashes": {"physics": sha(physics_config), "race": sha(race_config_example)},
                  "track": reference, "track_geometry_sha256": track_hash(reference), "simulator_seed": 42,
                  "spawns": [{"car_id": "ai", "spawn_id": "grid_right"}, {"car_id": "human", "spawn_id": "grid_left"}],
                  "model_sha256": None, "model_label": None, "dependencies": dependency_example,
                  "hash_interval": 100, "initial_state": initial, "initial_state_sha256": sha(initial)}
final = deepcopy(initial)
for car_value in final["world"]["cars"]:
    car_value["status"] = "aborted"
final["session"].update(session_state="RESULTS", winner="no_winner", abort_reason="inference_error")
event_example = {"record_type": "session_event", "seq": 1, "after_tick": 0, "event": "abort",
                 "reason": "inference_error", "verification_state": final, "state_sha256": sha(final)}
footer_example = {"record_type": "footer", "seq": 2, "status": "aborted", "tick_count": 0,
                  "winner": "no_winner", "abort_reason": "inference_error", "final_state": final,
                  "final_state_sha256": sha(final)}
(ROOT / "examples/replay_abort.jsonl").write_text("".join(canonical(r).decode()+"\n" for r in
                                                         [header_example, event_example, footer_example]), encoding="utf-8")
write("fixtures/numeric_cases.json", {"projection_switch": {"previous": [-2.08, 5.0], "next": [-2.06, 5.0],
      "segments": [[[-10.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [10.0, 10.0]]],
      "expected_signed_delta_m": 2.08 + 2.94/math.sqrt(2), "expected_displacement_m": .02},
      "gate_corner": {"width_m": 12.0, "turn_rad": math.pi/4,
      "positive_endpoint": [-6.0, 6.0], "negative_endpoint": [6/math.sqrt(2), -6/math.sqrt(2)]},
      "selection": {"candidates": [{"id": "A", "completion": .5, "progress": .8, "step": 100},
      {"id": "B", "completion": .5, "progress": .9, "step": 200},
      {"id": "C", "completion": .5, "progress": .85, "step": 300}], "winner": "B"},
      "bootstrap": {"A": [[.1, .2, .3], [.4, .5, .6]], "B": [[.3, .4, .5], [.6, .7, .8]],
      "expected_B_minus_A": .2, "rng_seed": 12345, "replicates": 2000}})
print("Built 7 schemas, fixture tracks, numeric cases, and fixture-only examples.")


build_recovery_assets()
