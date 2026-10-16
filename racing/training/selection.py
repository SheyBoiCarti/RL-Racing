"""selection-v1 total order, independent of returns and inventory ordering."""

import math
import re


def selection_key(candidate):
    for name in ("macro_completion", "macro_progress"):
        value = candidate.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name}: finite fraction required")
    for name in ("checkpoint_step", "update_count"):
        if type(candidate.get(name)) is not int or candidate[name] <= 0:
            raise ValueError(f"{name}: positive trained checkpoint required")
    complete = candidate["macro_completion"] == 1
    if type(candidate.get("all_success")) is not bool or candidate["all_success"] != complete:
        raise ValueError("all_success: inconsistent completion")
    lap = candidate.get("macro_mean_lap_time_s")
    if complete and (type(lap) not in (float, int) or not math.isfinite(lap) or lap <= 0):
        raise ValueError("macro_mean_lap_time_s: positive full-success lap required")
    if not complete and lap is not None:
        raise ValueError("macro_mean_lap_time_s: null required without full success")
    if not re.fullmatch("[0-9a-f]{64}", candidate.get("model_sha256", "")):
        raise ValueError("model_sha256: invalid digest")
    return (candidate["macro_completion"], candidate["macro_progress"],
            -lap if complete else 0., -candidate["checkpoint_step"], -candidate["update_count"])


def choose_best(candidates):
    candidates = list(candidates)
    if not candidates:
        raise ValueError("no trained candidates")
    return min(candidates, key=lambda c: (tuple(-x for x in selection_key(c)), c["model_sha256"]))
