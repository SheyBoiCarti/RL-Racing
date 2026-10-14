"""Verify specification assets; does not test or implement the future racing app."""
from copy import deepcopy
from itertools import permutations
from pathlib import Path
import hashlib
import json
import math
import re
import sys

import numpy as np
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ai-racing.local/specs/"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def canonical(value):
    def normalize_zero(item):
        if isinstance(item, float) and item == 0:
            return 0.0
        if isinstance(item, dict):
            return {k: normalize_zero(v) for k, v in item.items()}
        if isinstance(item, list):
            return [normalize_zero(v) for v in item]
        return item
    return json.dumps(normalize_zero(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


schemas = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "schemas").glob("*.json")}
registry = Registry().with_resources((s["$id"], Resource.from_contents(s)) for s in schemas.values())
validators = {name: Draft202012Validator(s, registry=registry) for name, s in schemas.items()}
for value in schemas.values():
    Draft202012Validator.check_schema(value)


def validate(kind, value):
    canonical(value)  # also rejects nonfinite values
    validators[kind+".schema.json"].validate(value)


def fails(call, label):
    try:
        call()
    except (AssertionError, ValueError, TypeError):
        return
    except Exception as error:
        if error.__class__.__name__ == "ValidationError":
            return
        raise
    raise AssertionError("Negative case incorrectly accepted: " + label)


def cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def project(point, a, b):
    edge = b-a
    t = float(np.clip(np.dot(point-a, edge)/np.dot(edge, edge), 0, 1))
    foot = a+t*edge
    return foot, t, float(np.linalg.norm(point-foot))


def hit(a, b, c, d):
    edge, other = b-a, d-c
    denominator = cross(edge, other)
    if abs(denominator) < 1e-9:
        if abs(cross(c-a, edge)) > 1e-8:
            return False
        denominator2 = np.dot(edge, edge)
        t0, t1 = sorted([np.dot(c-a, edge)/denominator2, np.dot(d-a, edge)/denominator2])
        return max(t0, 0) <= min(t1, 1)+1e-9
    t, u = cross(c-a, other)/denominator, cross(c-a, edge)/denominator
    return -1e-9 <= t <= 1+1e-9 and -1e-9 <= u <= 1+1e-9


def segment_distance(a, b, c, d):
    if hit(a, b, c, d):
        return 0.0
    return min(project(a, c, d)[2], project(b, c, d)[2],
               project(c, a, b)[2], project(d, a, b)[2])


def inside(point, polygon):
    result = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] > point[1]) != (b[1] > point[1]):
            cut = a[0]+(point[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if point[0] < cut:
                result = not result
    return result


def no_self_intersections(polygon, label):
    count = len(polygon)
    for i in range(count):
        for j in range(i+1, count):
            if j == i+1 or (i == 0 and j == count-1):
                continue
            demand(not hit(polygon[i], polygon[(i+1) % count], polygon[j], polygon[(j+1) % count]), label)


def point_at(points, lengths, cumulative, station):
    station %= cumulative[-1]
    index = int(np.searchsorted(cumulative, station, side="right")-1)
    direction = (points[(index+1) % len(points)]-points[index])/lengths[index]
    center = points[index]+(station-cumulative[index])*direction
    return center, direction


def gate(center, tangent, loops):
    normal = np.array([-tangent[1], tangent[0]])
    intersections = []
    for loop_id, loop in enumerate(loops):
        for a, b in zip(loop, np.roll(loop, -1, axis=0)):
            edge = b-a
            denominator = cross(normal, edge)
            if abs(denominator) < 1e-9:
                demand(abs(cross(a-center, normal)) > 1e-8, "Ambiguous collinear gate")
                continue
            t, u = cross(a-center, edge)/denominator, cross(a-center, normal)/denominator
            if -1e-9 <= u <= 1+1e-9:
                intersections.append((t, loop_id))
    positive = min((item for item in intersections if item[0] > 1e-6), key=lambda item: item[0])
    negative = max((item for item in intersections if item[0] < -1e-6), key=lambda item: item[0])
    demand(positive[1] != negative[1], "Gate hit wrong boundary loops")
    return center+negative[0]*normal, center+positive[0]*normal


def track_hash(track):
    payload = {key: deepcopy(track[key]) for key in ["schema_version", "centerline", "width_m", "checkpoints", "spawns"]}
    payload["centerline"] = [[float(x), float(y)] for x, y in payload["centerline"]]
    payload["width_m"] = float(payload["width_m"])
    for checkpoint in payload["checkpoints"]:
        checkpoint["s_m"] = float(checkpoint["s_m"])
    for spawn in payload["spawns"]:
        spawn["s_m"], spawn["lateral_m"] = float(spawn["s_m"]), float(spawn["lateral_m"])
    return sha(payload)


def validate_track(track):
    validate("track", track)
    points = np.array(track["centerline"])
    edges = np.roll(points, -1, axis=0)-points
    lengths = np.linalg.norm(edges, axis=1)
    demand(np.all(lengths >= .1), "Short centerline segment")
    total = float(lengths.sum())
    demand(400 <= total <= 800, "Reference length")
    tangent = edges/lengths[:, None]
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    previous = np.roll(normals, 1, axis=0)
    denominator = 1+np.sum(previous*normals, axis=1)
    demand(np.all(denominator > 1e-9), "Reversing miter")
    offset = (track["width_m"]/2)*(normals+previous)/denominator[:, None]
    demand(np.max(np.linalg.norm(offset, axis=1)) <= track["width_m"], "Miter limit")
    loops = [points+offset, points-offset]
    for polygon in [points, *loops]:
        no_self_intersections(polygon, "Self intersection")
    for a, b in zip(loops[0], np.roll(loops[0], -1, axis=0)):
        for c, d in zip(loops[1], np.roll(loops[1], -1, axis=0)):
            demand(not hit(a, b, c, d), "Boundary loop intersection")
    if inside(loops[0][0], loops[1]):
        inner, outer = loops
    else:
        outer, inner = loops
    demand(all(inside(p, outer) for p in inner), "Loops not fully nested")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    headings = np.arctan2(edges[:, 1], edges[:, 0])
    turns = (headings-np.roll(headings, 1)+math.pi) % (2*math.pi)-math.pi
    demand(np.any(turns > 1e-5) and np.any(turns < -1e-5), "Both turn directions required")
    demand(np.max(np.abs(turns)) <= .35, "Reference turn limit")
    curvature = np.abs(turns)/((lengths+np.roll(lengths, 1))/2)
    demand(np.max(curvature) <= .10, "Reference curvature")
    midpoint_s = cumulative[:-1]+lengths/2
    minimum_clearance = math.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            separation = abs(midpoint_s[i]-midpoint_s[j])
            if min(separation, total-separation) <= 2*track["width_m"]:
                continue
            distance = segment_distance(points[i], points[(i+1) % len(points)],
                                        points[j], points[(j+1) % len(points)])
            minimum_clearance = min(minimum_clearance, distance)
            demand(distance >= track["width_m"]+2-1e-6, "Nonlocal clearance")
    gate_stations = [checkpoint["s_m"] for checkpoint in track["checkpoints"]]
    demand(track["checkpoints"][0] == {"index": 0, "s_m": 0.0}, "Finish station")
    demand([g["index"] for g in track["checkpoints"]] == list(range(len(gate_stations))), "Gate indices")
    demand(all(b-a >= 20 for a, b in zip(gate_stations, gate_stations[1:]+[total])), "Gate spacing")
    gate_lengths = []
    for station in gate_stations:
        center, direction = point_at(points, lengths, cumulative, station)
        a, b = gate(center, direction, loops)
        demand(min(project(a, p, q)[2] for loop in loops for p, q in zip(loop, np.roll(loop, -1, axis=0))) < 1e-6, "Gate endpoint A")
        demand(min(project(b, p, q)[2] for loop in loops for p, q in zip(loop, np.roll(loop, -1, axis=0))) < 1e-6, "Gate endpoint B")
        for t in np.linspace(.001, .999, max(4, math.ceil(np.linalg.norm(b-a)/.25))):
            point = a+t*(b-a)
            demand(inside(point, outer) and not inside(point, inner), "Gate outside road")
        for loop in loops:
            for p, q in zip(loop, np.roll(loop, -1, axis=0)):
                if hit(a, b, p, q):
                    demand(project(a, p, q)[2] < 1e-6 or project(b, p, q)[2] < 1e-6,
                           "Intermediate boundary crosses gate")
        gate_lengths.append(float(np.linalg.norm(b-a)))
    spawns = {}
    for spawn in track["spawns"]:
        demand(abs(spawn["s_m"]-(total-5)) < 1e-6, "Spawn station")
        center, direction = point_at(points, lengths, cumulative, spawn["s_m"])
        position = center+spawn["lateral_m"]*np.array([-direction[1], direction[0]])
        demand(inside(position, outer) and not inside(position, inner), "Spawn outside road")
        clearance = min(project(position, p, q)[2] for loop in loops for p, q in zip(loop, np.roll(loop, -1, axis=0)))
        demand(clearance >= .8-1e-6, "Spawn disk exceeds road")
        spawns[spawn["spawn_id"]] = position
    demand(np.linalg.norm(spawns["grid_left"]-spawns["grid_right"]) > 1.6, "Grid overlap")
    return {"track_id": track["track_id"], "points": len(points), "length_m": total,
            "max_curvature": float(curvature.max()), "min_nonlocal_centerline_distance_m": minimum_clearance,
            "gate_length_range_m": [min(gate_lengths), max(gate_lengths)], "geometry_sha256": track_hash(track)}


def validate_model(value):
    validate("model", value)
    obs = value["observation_config"]
    count = int(obs["version"].split("rays")[1].split("-")[0])
    expected = [-90, -60, -30, 0, 30, 60, 90] if count == 7 else np.linspace(-90, 90, count).tolist()
    demand(obs["ray_angles_deg"] == expected and obs["dimension"] == count+4, "Observation semantics")
    for key in ["physics", "observation", "action", "env"]:
        config = deepcopy(value[key+"_config"])
        if key == "physics":
            config = {k: float(v) for k, v in config.items()}
        elif key == "observation":
            config["ray_range_m"] = float(config["ray_range_m"])
            config["ray_angles_deg"] = [float(angle) for angle in config["ray_angles_deg"]]
        elif key == "env":
            for field in ["progress_threshold_m", "steering_penalty", "progress_coefficient", "step_cost", "off_track_penalty", "lap_bonus"]:
                config[field] = float(config[field])
        demand(value["config_hashes"][key] == sha(config), "Config hash: " + key)
    demand(value["config_hashes"]["training"] == sha(value["hyperparameters"]), "Hyperparameter hash")
    for key in ["observation", "action"]:
        demand(value["versions"][key] == value[key+"_config"]["version"], "Version mismatch")
    demand(value["versions"]["reward"] == value["env_config"]["reward_version"], "Reward version")
    metric = value["selection_metrics"]
    demand(metric["all_success"] == (metric["macro_completion"] == 1), "Selection all-success")
    demand((metric["macro_mean_lap_time_s"] is not None) == metric["all_success"], "Selection time eligibility")


def examples_check():
    demand(canonical({"zero": -0.0}) == canonical({"zero": 0.0}), "Signed-zero canonicalization")
    demand(canonical({"b": 2, "a": 1}) == canonical({"a": 1, "b": 2}), "Sorted canonical keys")
    for kind in ["model", "split", "experiment", "selection"]:
        example = load("examples/"+kind+".json")
        validate(kind, example)
        extra = deepcopy(example)
        extra["unexpected"] = 1
        fails(lambda: validate(kind, extra), kind+" extra field")
    models = [load("examples/model.json"), *[load(p.relative_to(ROOT)) for p in (ROOT / "examples/models").glob("*.json")]]
    for model in models:
        validate_model(model)
        demand(model["model_sha256"] == hashlib.sha256((ROOT / "fixtures/model_bytes.fixture").read_bytes()).hexdigest(), "Model fixture hash")
    malformed = deepcopy(models[0])
    malformed["observation_config"]["dimension"] += 1
    fails(lambda: validate_model(malformed), "Dimension semantics")
    numeric_type_variant = deepcopy(models[0])
    numeric_type_variant["physics_config"]["max_speed_mps"] = 22
    validate_model(numeric_type_variant)  # declared float hashes normalize int spelling
    malformed = deepcopy(models[0])
    malformed["physics_config"]["dt_s"] = float("nan")
    fails(lambda: validate_model(malformed), "Nonfinite config")
    split = load("examples/split.json")
    demand(split["manifest_sha256"] == sha({k: v for k, v in split.items() if k != "manifest_sha256"}), "Split self hash")
    demand(split["generator_config_hash"] == sha(split["generator_config"]), "Generator config hash")
    entries = split["entries"]
    for field in ["track_id", "seed", "geometry_hash"]:
        demand(len({e[field] for e in entries}) == len(entries), "Duplicate split " + field)
    for name, count in split["counts"].items():
        demand(count == sum(e["split"] == name for e in entries), "Split count")
    for entry in entries:
        demand(track_hash(load(entry["path"])) == entry["geometry_hash"], "Split geometry hash")
    experiment = load("examples/experiment.json")
    demand(len({c["condition_id"] for c in experiment["conditions"]}) == 6, "Condition coverage")
    for condition in experiment["conditions"]:
        sidecar = load(condition["config_path"])
        demand(sha(sidecar) == condition["config_sha256"], "Condition config hash")
        demand(sidecar["algorithm"] == condition["algorithm"], "Condition algorithm")
        demand(sidecar["observation_config"]["version"] == condition["observation_version"], "Condition observation")
        demand(sidecar["env_config"]["reward_version"] == condition["reward_version"], "Condition reward")
    selected = load("examples/selection.json")
    demand(selected["experiment_sha256"] == sha(experiment), "Frozen experiment hash")
    demand(selected["manifest_sha256"] == sha({k: v for k, v in selected.items() if k != "manifest_sha256"}), "Selection self hash")
    for candidate in selected["models"]:
        demand(sha(load(candidate["metadata_path"])) == candidate["metadata_sha256"], "Frozen sidecar hash")
    demand(selected["fixture_only"] and experiment["fixture_only"], "Examples must not authorize science")


def numeric_check():
    cases = load("fixtures/numeric_cases.json")
    # Gate x=0, endpoints (0,-1)/(0,1), forward tangent (1,0).
    # Independently specified trajectories exercise sign and finite-span rules.
    crossing_cases = [
        ("near-zero approach", [(-.01, 0), (-.0000005, 0), (.01, 0)], [False, True]),
        ("exact-zero arrival", [(-.01, 0), (0, 0), (.01, 0)], [True, False]),
        ("tiny forward", [(-1e-8, 0), (1e-8, 0)], [True]),
        ("stationary", [(0, 0), (0, 0)], [False]),
        ("backward", [(.01, 0), (-.01, 0)], [False]),
        ("outside finite gate", [(-.01, 1.001), (.01, 1.001)], [False]),
        ("finite endpoint", [(-.01, 1), (.01, 1)], [True]),
    ]
    for label, positions, expected_crossings in crossing_cases:
        actual = []
        for previous, current in zip(positions, positions[1:]):
            old, new = previous[0], current[0]
            crossed = False
            if old < 0 <= new and new-old > 0:
                fraction = old/(old-new)
                y = previous[1]+fraction*(current[1]-previous[1])
                crossed = 0 < fraction <= 1 and -1-1e-6 <= y <= 1+1e-6
            actual.append(crossed)
        demand(actual == expected_crossings, "Gate crossing: " + label)
    switch = cases["projection_switch"]
    segments = [(np.array(a), np.array(b)) for a, b in switch["segments"]]
    old, new = np.array(switch["previous"]), np.array(switch["next"])
    old_results = [project(old, a, b) for a, b in segments]
    new_results = [project(new, a, b) for a, b in segments]
    demand(np.argmin([r[2] for r in old_results]) == 0, "Old nearest segment")
    demand(np.argmin([r[2] for r in new_results]) == 1, "New nearest segment")
    advancement = 10*(1-old_results[0][1])+math.sqrt(200)*new_results[1][1]
    distance = float(np.linalg.norm(new-old))
    bound = (distance+old_results[0][2]+new_results[1][2])/math.cos(math.pi/8)+1e-6
    demand(abs(advancement-switch["expected_signed_delta_m"]) < 1e-10, "Projected numeric advancement")
    demand(advancement > 1.2 and advancement <= bound and distance <= 1.100001, "Legal corner continuity")
    demand(1.2 > 22*.05+1e-6, "Corrupt physical jump rejected")
    tangent = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    miter = np.array([-6*math.tan(math.pi/8), 6])
    inner_edge = np.array([[-10, 6], miter])
    outer_miter = -miter
    outer_edge = np.array([outer_miter, outer_miter+10*tangent])
    negative, positive = gate(np.zeros(2), tangent, [inner_edge, outer_edge])
    expected = cases["gate_corner"]
    demand(np.allclose(positive, expected["positive_endpoint"], atol=1e-10), "Corner gate inner endpoint")
    demand(np.allclose(negative, expected["negative_endpoint"], atol=1e-10), "Corner gate outer endpoint")
    wrong = 6*np.array([-tangent[1], tangent[0]])
    demand(project(wrong, *inner_edge)[2] > .1, "Half-width endpoint must fail")
    for order in permutations(cases["selection"]["candidates"]):
        winner = max(order, key=lambda x: (x["completion"], x["progress"], -x["step"]))
        demand(winner["id"] == cases["selection"]["winner"], "Order-dependent selection")
    boot = cases["bootstrap"]
    a, b = np.array(boot["A"]), np.array(boot["B"])
    rng = np.random.default_rng(boot["rng_seed"])
    differences = []
    for _ in range(boot["replicates"]):
        rows = rng.integers(0, len(a), size=len(a))
        cols = rng.integers(0, a.shape[1], size=a.shape[1])
        aa, bb = a[rows[:, None], cols[None, :]], b[rows[:, None], cols[None, :]]
        demand(float((aa-aa).mean()) == 0.0, "Identical paired bootstrap")
        differences.append(float(bb.mean()-aa.mean()))
    demand(np.allclose(differences, boot["expected_B_minus_A"], atol=1e-15, rtol=0), "Constant-offset bootstrap")
    return {"corner_projected_advancement_m": advancement, "corner_physical_displacement_m": distance,
            "continuity_bound_m": bound, "gate_crossing_cases": len(crossing_cases),
            "selection_permutations": 6, "bootstrap_replicates": len(differences)}


def replay_check():
    records = [json.loads(line) for line in (ROOT / "examples/replay_abort.jsonl").read_text().splitlines()]
    for record in records:
        validate("replay", record)
    demand([r["seq"] for r in records] == list(range(len(records))), "Replay sequence")
    header, event, footer = records
    demand(header["record_type"] == "header" and footer["record_type"] == "footer", "Replay framing")
    demand(event["event"] == "abort" and event["after_tick"] == footer["tick_count"] == 0, "Zero-tick abort")
    reconstructed = deepcopy(header["initial_state"])
    demand(sha(reconstructed) == header["initial_state_sha256"], "Initial state hash")
    for car in reconstructed["world"]["cars"]:
        demand(car["status"] == "active", "Initial car status")
        car["status"] = "aborted"
    reconstructed["session"].update(session_state="RESULTS", winner="no_winner", abort_reason=event["reason"])
    demand(reconstructed == event["verification_state"] == footer["final_state"], "Replay abort reconstruction")
    demand(sha(reconstructed) == event["state_sha256"] == footer["final_state_sha256"], "Replay final hash")
    demand(track_hash(header["track"]) == header["track_geometry_sha256"], "Replay track hash")
    def hex_values(value):
        if isinstance(value, dict):
            for child in value.values():
                hex_values(child)
        elif isinstance(value, list):
            for child in value:
                hex_values(child)
        elif isinstance(value, str) and value.startswith(("0x", "-0x")):
            demand(math.isfinite(float.fromhex(value)), "Replay hex finite")
    hex_values(reconstructed)


def markdown_check():
    count = 0
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8-sig")
        demand(text.startswith("# "), "Missing title " + str(path))
        demand("\ufffd" not in text, "Encoding replacement character")
        demand(len(re.findall(r"(?m)^```", text)) % 2 == 0, "Unbalanced code fences " + str(path))
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "app://")):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            demand(resolved.exists(), "Broken local link " + str(path) + " -> " + target)
        count += 1
    return count


def main():
    examples_check()
    reports = [validate_track(load(p.relative_to(ROOT))) for p in sorted((ROOT / "fixtures").glob("*.json"))
               if p.name != "numeric_cases.json"]
    numeric = numeric_check()
    replay_check()
    markdown_count = markdown_check()
    print(json.dumps({"status": "PASS", "schemas": len(schemas), "markdown_documents": markdown_count,
                      "reference_tracks": reports, "numeric_checks": numeric,
                      "scope": "specification assets only; no application/GUI/training executed"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
