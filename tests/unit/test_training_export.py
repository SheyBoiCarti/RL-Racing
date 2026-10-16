"""Export must survive relocation and reject corruption and path escapes."""

import json
from pathlib import Path

import pytest


def test_export_verifies_fresh_root_and_detects_corrupt_files(tmp_path):
    from racing.training.export import export_tree, import_export, verify_export

    source = tmp_path / "run"
    source.mkdir()
    (source / "manifest.json").write_text('{"status":"interrupted"}')
    (source / "parents").mkdir()
    (source / "parents/model.zip").write_bytes(b"ancestor")
    (source / ".model.staging").write_bytes(b"incomplete")
    archive = tmp_path / "export.zip"
    export_tree(source, archive)
    destination = tmp_path / "fresh"
    import_export(archive, destination)
    assert verify_export(destination)
    assert (destination / "parents/model.zip").read_bytes() == b"ancestor"
    assert not (destination / ".model.staging").exists()
    (destination / "parents/model.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash|size"):
        verify_export(destination)


def test_inventory_and_archive_paths_cannot_escape_root(tmp_path):
    import zipfile

    from racing.training.export import import_export, verify_export

    root = tmp_path / "bad"
    root.mkdir()
    (root / "inventory.json").write_text(json.dumps({"files": {"../outside": {"sha256": "0" * 64, "size": 0}}}))
    with pytest.raises(ValueError, match="path"):
        verify_export(root)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="path"):
        import_export(archive, tmp_path / "import")
    assert not (tmp_path / "outside").exists()


def test_packaged_sources_exclude_credentials_and_verify_bytes(tmp_path):
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location("pipeline", Path("scripts/kaggle_pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage = module.package_source(tmp_path / "package")
    manifest = json.loads((stage / "source-manifest.json").read_text())
    with zipfile.ZipFile(stage / "source.zip") as archive:
        names = set(archive.namelist())
        assert "scripts/train.py" in names and "assets/tracks/demo.json" in names
        assert "notebooks/kaggle_train.ipynb" in names
        assert not any("kaggle.json" in n or ".venv/" in n or "runs/" in n for n in names)
        import hashlib
        assert hashlib.sha256((stage / "source.zip").read_bytes()).hexdigest() == manifest["archive_sha256"]


def test_training_export_requires_checkpoint_inventory(tmp_path):
    from racing.training.export import export_runs

    root = tmp_path / "empty-run"
    root.mkdir()
    (root / "manifest.json").write_text('{"run_id":"empty-run", "status":"interrupted"}')
    with pytest.raises(ValueError, match="checkpoint"):
        export_runs(root, tmp_path / "bad.zip")


def test_full_export_resume_resolves_explicit_active_run_after_relocation(tmp_path):
    from racing.training.export import (
        export_tree,
        import_export,
        select_resume_checkpoint,
    )

    runs = tmp_path / "runs"
    (runs / "ppo-demo-seed42/checkpoints/step_000000064_update_000001").mkdir(parents=True)
    model = runs / "ppo-demo-seed42/checkpoints/step_000000064_update_000001/model.zip"
    model.write_bytes(b"selection fixture")
    (runs / "active-run.json").write_text(json.dumps({"run_directory": "ppo-demo-seed42"}))
    (model.parents[1] / "committed.json").write_text(json.dumps({"latest": "checkpoints/step_000000064_update_000001/model.zip"}))
    archive = tmp_path / "training-export.zip"
    export_tree(runs, archive)
    fresh = tmp_path / "new-session"
    import_export(archive, fresh)
    assert select_resume_checkpoint(fresh) == fresh / "ppo-demo-seed42/checkpoints/step_000000064_update_000001/model.zip"
    (fresh / "active-run.json").write_text(json.dumps({"run_directory": "../escape"}))
    with pytest.raises(ValueError, match="path"):
        select_resume_checkpoint(fresh)


def test_notebook_accepts_kaggle_expanded_source_and_checks_file_hashes(tmp_path):
    import ast
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location("pipeline", Path("scripts/kaggle_pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage = module.package_source(tmp_path / "input")
    with zipfile.ZipFile(stage / "source.zip") as archive:
        archive.extractall(stage / "source")
    (stage / "source.zip").unlink()  # Kaggle expands ZIP inputs automatically.
    prior = stage / "previous-notebook/runs"
    prior.mkdir(parents=True)
    (prior / "source-manifest.json").write_text('{"copied_run_provenance":true}')
    notebook = json.loads(Path("notebooks/kaggle_train.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("/kaggle/input", str(stage)).replace("/kaggle/working/ai-racing", str(tmp_path / "working"))
    exec(compile(tree, "<notebook-input-verification>", "exec"), {})
    assert (tmp_path / "working/scripts/train.py").is_file()


def test_launcher_surfaces_rejected_kernel_inputs_even_when_cli_exits_zero(tmp_path, monkeypatch):
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location("pipeline", Path("scripts/kaggle_pipeline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "kaggle_command", lambda: ["kaggle"])
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, stdout="The following are not valid kernel sources and could not be added to the kernel", stderr=""))
    with pytest.raises(ValueError, match="source"):
        module.main(["launch", "--directory", str(tmp_path / "launch"), "--dataset", "owner/source",
                     "--mode", "resume-acceptance", "--resume-kernel", "owner/prior-run"])
