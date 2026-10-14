from dataclasses import FrozenInstanceError

import pytest
import yaml

from racing.core.config import load_game, load_physics
from racing.core.models import CarControl


def test_defaults_and_immutability():
    physics, game = load_physics(), load_game()
    assert physics.dt_s == 0.05 and physics.car_radius_m == 0.8
    assert game.catch_up_limit == 5 and game.debug is False
    with pytest.raises(FrozenInstanceError):
        physics.dt_s = 0.1
    with pytest.raises(FrozenInstanceError):
        CarControl(0, 0).steering = 1
    assert isinstance(game.bindings, tuple)


@pytest.mark.parametrize(
    "content,field",
    [
        ("dt_s: true", "dt_s"),
        ("dt_s: 0", "dt_s"),
        ("wheelbase_m: -1", "wheelbase_m"),
        ("car_radius_m: .nan", "car_radius_m"),
        ("max_speed_mps: .inf", "max_speed_mps"),
        ("oops: 3", "oops"),
        ("dt_s: .05\ndt_s: .1", "dt_s"),
    ],
)
def test_invalid_physics(tmp_path, content, field):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError) as caught:
        load_physics(path)
    assert str(path) in str(caught.value) and field in str(caught.value)


def test_paths_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_physics().max_speed_mps == 22
    assert load_game().width_px == 1280


@pytest.mark.parametrize(
    "field,value",
    [
        ("render_fps", True),
        ("width_px", 0),
        ("pixels_per_metre", float("nan")),
        ("pixels_per_metre", True),
        ("debug", 1),
        ("follow_camera", "true"),
        ("catch_up_limit", 6),
        ("oops", 2),
    ],
)
def test_invalid_game(tmp_path, field, value):
    from racing.core.config import project_path

    data = yaml.safe_load(project_path("configs/game.yaml").read_text())
    data[field] = value
    path = tmp_path / "game.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError) as caught:
        load_game(path)
    assert str(path) in str(caught.value) and field in str(caught.value)


def test_binding_conflicts(tmp_path):
    from racing.core.config import project_path

    data = yaml.safe_load(project_path("configs/game.yaml").read_text())
    data["bindings"]["reset"] = ["esc"]
    path = tmp_path / "conflict.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="conflicting key") as caught:
        load_game(path)
    assert str(path) in str(caught.value)
