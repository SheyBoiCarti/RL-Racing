"""Frozen Phase 4 inputs and condition-specific hosted launch contracts."""

import ast
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest


def pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", Path("scripts/kaggle_pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_package_contains_frozen_experiment_protocol(tmp_path):
    module = pipeline()
    stage = module.package_source(tmp_path / "source")
    with zipfile.ZipFile(stage / "source.zip") as archive:
        assert "experiments/phase04.yaml" in archive.namelist()
        assert "experiments/splits/v1.json" in archive.namelist()
    manifest = json.loads((stage / "source-manifest.json").read_text())
    assert manifest["split_manifest_sha256"]


@pytest.mark.parametrize("mode", ["phase04-smoke", "phase04-full"])
def test_phase04_launch_binds_condition_and_frozen_hash(tmp_path, monkeypatch, mode):
    module = pipeline()
    monkeypatch.setattr(module, "kaggle_command", lambda: ["kaggle"])
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout="ok", stderr=""))
    destination = tmp_path / mode
    module.main(["launch", "--directory", str(destination), "--dataset", "owner/frozen-source",
                 "--mode", mode, "--condition", "multi", "--split-hash", "a" * 64])
    notebook = json.loads((destination / "kaggle_train.ipynb").read_text())
    parameters = "".join(notebook["cells"][1]["source"])
    assert "CONDITION = 'multi'" in parameters
    assert "SPLIT_MANIFEST_SHA256 = '" + "a" * 64 in parameters
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["enable_gpu"]


def test_phase04_launch_rejects_missing_frozen_hash_before_upload(tmp_path, monkeypatch):
    module = pipeline()
    monkeypatch.setattr(module, "kaggle_command", lambda: ["kaggle"])
    with pytest.raises(SystemExit):
        module.main(["launch", "--directory", str(tmp_path / "launch"), "--dataset", "owner/source",
                     "--mode", "phase04-full", "--condition", "single"])
    assert not (tmp_path / "launch").exists()


def test_download_uses_utf8_and_larger_output_pages(tmp_path, monkeypatch):
    module = pipeline()
    monkeypatch.setattr(module, "kaggle_command", lambda: ["kaggle"])
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    module.main(["download", "--directory", str(tmp_path / "download")])
    args, kwargs = calls[-1]
    assert "--page-size" in args[0] and "200" in args[0]
    assert kwargs["env"]["PYTHONUTF8"] == "1"


def test_notebook_cells_compile_and_phase04_stop_still_exports(tmp_path):
    notebook = json.loads(Path("notebooks/kaggle_train.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    calls = []
    namespace = dict(ROOT=tmp_path, MODE="phase04-full", CONDITION="multi", RESUME_ARCHIVE=None,
                     EXECUTION=tmp_path / "execution.yaml", SMOKE=tmp_path / "phase04.session.yaml",
                     run=lambda *a, **kw: calls.append((a, kw)) or 75)
    exec("".join(notebook["cells"][9]["source"]), namespace)
    args, kwargs = calls[-1]
    assert args[0] == "scripts/train.py"
    assert tmp_path / "phase04.session.yaml" in args
    assert namespace["active_run"] == tmp_path / "runs/phase04-multi"
    assert kwargs["allowed"] == (0, 75)


def test_notebook_phase04_final_evaluation_uses_validation_only(tmp_path):
    notebook = json.loads(Path("notebooks/kaggle_train.ipynb").read_text(encoding="utf-8"))
    active = tmp_path / "runs/phase04-single"
    active.mkdir(parents=True)
    (active / "manifest.json").write_text('{"status":"completed"}')
    calls = []
    namespace = dict(ROOT=tmp_path, MODE="phase04-smoke", CONDITION="single", active_run=active,
                     json=json, run=lambda *a, **kw: calls.append(a))
    exec("".join(notebook["cells"][11]["source"]), namespace)
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index("--split") + 1] == "validation"
    assert "demo" not in args and "test" not in args
