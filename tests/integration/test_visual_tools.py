"""Capture CLI must produce reproducible, explicitly labeled artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pygame


def test_capture_suite_manifest(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / "scripts/capture_debug.py"),
                             "--suite", "--output", str(tmp_path)],
                            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    expected = {"menu", "driving", "paused", "failed", "finished", "debug_overview",
                "menu_compact", "driving_compact"}
    assert set(manifest) == expected
    for name, item in manifest.items():
        size = (480, 360) if name.endswith("compact") else (1280, 720)
        assert pygame.image.load(tmp_path / f"{name}.png").get_size() == size
        assert tuple(item["viewport"]) == size
    for name in ("failed", "finished"):
        assert "synthetic" in manifest[name]["provenance"]


def test_profile_moving_frames_and_validation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "scripts/profile_render.py")]
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    invalid = subprocess.run(command + ["--frames", "0"], env=env, capture_output=True, timeout=30)
    assert invalid.returncode == 2
    output = tmp_path / "profile.json"
    result = subprocess.run(command + ["--frames", "8", "--output", str(output)],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    profile = json.loads(output.read_text())
    assert profile["frames"] == 8
    assert profile["last_tick"] > profile["first_tick"]
    assert profile["last_position"] != profile["first_position"]
    assert profile["median_draw_ms"] > 0
    assert profile["cache"]["tiles"] <= 32
    assert profile["video_driver"] == "dummy"
