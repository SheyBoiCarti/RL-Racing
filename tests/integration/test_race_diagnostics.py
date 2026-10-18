"""Public sample and diagnostic failures must remain explicit and reproducible."""
from types import SimpleNamespace

import pytest


def test_public_sample_selects_only_declared_sorted_ids(monkeypatch):
    from scripts import verify_race_difficulty as diagnostic
    calls = []
    def catalog(path, split):
        calls.append(split)
        return {f"{split}_{i:03}": SimpleNamespace(track_id=f"{split}_{i:03}")
                for i in (4, 2, 0, 3, 1)}
    monkeypatch.setattr(diagnostic, "load_split_catalog", catalog)
    monkeypatch.setattr(diagnostic.Track, "from_json", lambda path: SimpleNamespace(track_id="demo"))
    assert [t.track_id for t in diagnostic.public_sample()] == ["demo", "train_000", "train_001",
        "train_002", "validation_000", "validation_001", "validation_002"]
    assert calls == ["train", "validation"]


@pytest.mark.parametrize("fail", [False, True])
def test_diagnostics_inventory_and_zero_success(tmp_path, monkeypatch, fail):
    from scripts import verify_race_difficulty as diagnostic
    model = tmp_path / "model.zip"
    model.write_bytes(b"test-model")
    tracks = [SimpleNamespace(track_id=key, split=split, geometry_payload={"id": key})
              for split, keys in [("demo", ["demo"]), ("train", [f"train_{i:03}" for i in range(3)]),
                                   ("validation", [f"validation_{i:03}" for i in range(3)])]
              for key in keys]
    monkeypatch.setattr(diagnostic, "public_sample", lambda: tracks)
    monkeypatch.setattr(diagnostic, "load_policy", lambda model: SimpleNamespace(metadata={"model_sha256": "a" * 64}))
    seen = []
    def run(track, controller, difficulty, path, seed):
        seen.append((track.track_id, difficulty, path, seed, controller))
        if fail and track.track_id == "train_001" and difficulty == "medium":
            raise RuntimeError("incomplete run")
        return dict(tick_count=10, replay_sha256="b" * 64, final_state_sha256="c" * 64,
                    results={key: dict(status="timed_out", reason="race_max_steps", completed_laps=0,
                        lap_times_s=[], recovery_count=2, recovery_held_ticks=40)
                        for key in ("ai", "human")})
    monkeypatch.setattr(diagnostic, "run_race", run)
    output = tmp_path / "diagnostics"
    if fail:
        with pytest.raises(RuntimeError, match="incomplete"):
            diagnostic.run_diagnostics(model, output)
        import json
        report = json.loads((output / "inventory.json").read_text())
        assert not report["complete"]
        assert len(report["races"]) == 21
        assert sum("error" in row for row in report["races"]) == 1
    else:
        report = diagnostic.run_diagnostics(model, output)
        assert report["complete"] and len(report["races"]) == 21
        assert all(row["mean_successful_lap_s"] is None for row in report["tiers"].values())
        assert (output / "races.csv").is_file()
    assert len(seen) == 21 and len({row[2] for row in seen}) == 21
    assert {row[3] for row in seen} == {0}
    assert len({id(row[4]) for row in seen}) == 1
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostics(model, output)


@pytest.mark.parametrize("module", ["verify_race", "capture_race"])
def test_legacy_rules_reject_explicit_difficulty_before_loading(module, tmp_path, capsys):
    import importlib
    script = importlib.import_module("scripts." + module)
    args = ["--rules", "legacy", "--difficulty", "easy", "--output", str(tmp_path / "new")]
    if module == "capture_race":
        args += ["--model", str(tmp_path / "missing.zip")]
    with pytest.raises(SystemExit) as error:
        script.main(args)
    assert error.value.code == 2
    assert "requires recovery" in capsys.readouterr().err
