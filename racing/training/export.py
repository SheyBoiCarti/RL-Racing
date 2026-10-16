"""Portable whole-lineage exports with byte inventories and safe extraction."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from racing.training.manifest import file_hash, read_json, write_json


def safe_path(root, name):
    relative = PurePosixPath(name)
    target = (root / name).resolve()
    if "\\" in name or relative.is_absolute() or ".." in relative.parts or ":" in name or not target.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe inventory/archive path: {name}")
    return target


def verify_export(root):
    root = Path(root)
    inventory = read_json(root / "inventory.json")
    if not isinstance(inventory.get("files"), dict) or not inventory["files"]:
        raise ValueError("export inventory: missing files")
    for name, record in inventory["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or path.stat().st_size != record["size"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"export file hash/size mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "inventory.json"}
    if actual != set(inventory["files"]):
        raise ValueError("export inventory does not cover exact file set")
    return inventory


def export_tree(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("export destination must be new and outside source tree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()
                   and not any(part.startswith(".") or part.endswith(".staging") for part in p.relative_to(root).parts))
    inventory = dict(schema_version="export-v1", files={p.relative_to(root).as_posix(): dict(size=p.stat().st_size, sha256=file_hash(p)) for p in files})
    temporary = destination.with_name("." + destination.name + ".staging")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
            import json

            archive.writestr("inventory.json", json.dumps(inventory, sort_keys=True))
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            extracted = Path(directory) / "verify"
            import_export(temporary, extracted)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return inventory


def import_export(archive_path, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("import destination must be new; parent artifacts are immutable")
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            safe_path(destination, entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive path is a symbolic link")
        destination.mkdir(parents=True)
        try:
            archive.extractall(destination)
            return verify_export(destination)
        except BaseException:
            shutil.rmtree(destination)
            raise


def export_runs(root, destination):
    """Validate every committed bundle before exporting its full ancestor tree."""
    from racing.training.checkpoints import load_bundle

    root = Path(root)
    inventories = list(root.rglob("committed.json"))
    if not inventories:
        raise ValueError("training export: no committed checkpoint inventory")
    training_manifests = [p for p in root.rglob("manifest.json") if "run_id" in read_json(p)]
    for manifest in training_manifests:
        inventory = manifest.parent / "checkpoints/committed.json"
        if not inventory.is_file() or not read_json(inventory).get("checkpoints"):
            raise ValueError(f"training export: missing checkpoint inventory: {manifest.parent}")
    for inventory in inventories:
        owner = inventory.parent.parent
        for checkpoint in read_json(inventory)["checkpoints"]:
            load_bundle(owner / checkpoint["path"])
    result = export_tree(root, destination)
    write_json(Path(destination).with_suffix(".inventory.json"), result)
    return result


def select_resume_checkpoint(root):
    """Resolve the declared active run in either a single-run or runs export."""
    root = Path(root).resolve()
    if (root / "checkpoints/committed.json").is_file():
        owner = root
    else:
        selector = root / "active-run.json"
        if not selector.is_file():
            raise ValueError("export: missing explicit active-run.json selector")
        owner = safe_path(root, read_json(selector)["run_directory"])
    inventory = owner / "checkpoints/committed.json"
    if not inventory.is_file():
        raise ValueError("export active run: missing checkpoint inventory")
    path = safe_path(owner, read_json(inventory)["latest"])
    if not path.is_file():
        raise ValueError("export active run: missing latest checkpoint")
    return path
