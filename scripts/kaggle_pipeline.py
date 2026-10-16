"""Package a private Kaggle source dataset and launch reproducible notebooks."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing.training.manifest import ROOT, file_hash, source_identity, write_json


def package_source(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    for directory in ("racing", "scripts", "configs", "assets", "experiments", "tests", "notebooks", "docs/specs"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and
                     "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo") and
                     not any(part.startswith(".") for part in p.relative_to(ROOT).parts))
    paths.extend(ROOT.glob("requirements*lock.txt"))
    paths.extend([p for p in [ROOT / "pyproject.toml", ROOT / "README.md"] if p.is_file()])
    archive = destination / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as file:
        for path in sorted(paths):
            file.write(path, path.relative_to(ROOT).as_posix())
    manifest = dict(
        schema_version="source-v1", archive_sha256=file_hash(archive), source_sha256=source_identity(),
        lockfile_sha256=file_hash(ROOT / "requirements-kaggle.lock.txt"),
        files={p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)})
    split = ROOT / "experiments/splits/v1.json"
    if split.is_file():
        manifest["split_manifest_sha256"] = json.loads(split.read_text(encoding="utf-8"))["manifest_sha256"]
    write_json(destination / "source-manifest.json", manifest)
    return destination


def kaggle_command():
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    # The CLI may live under a different Python than the project runtime.
    candidates = list((Path.home() / "AppData/Roaming/Python").glob("Python*/Scripts/kaggle.exe"))
    if candidates:
        return [str(sorted(candidates)[-1])]
    raise ValueError("Install Kaggle CLI and run kaggle auth login first")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["package", "launch", "status", "download"])
    parser.add_argument("--directory", required=True)
    parser.add_argument("--username", default="sheyboicarti")
    parser.add_argument("--slug", default="rl-racing-phase03")
    parser.add_argument("--dataset", help="Versioned private source dataset, username/slug")
    parser.add_argument("--mode", choices=["acceptance", "resume-acceptance", "full", "phase04-smoke", "phase04-full"], default="acceptance")
    parser.add_argument("--condition", choices=["single", "multi"], help="Phase 4 fresh training condition")
    parser.add_argument("--split-hash", help="Expected frozen manifest self-hash (required for Phase 4)")
    parser.add_argument("--resume-kernel", help="Attach a prior notebook's saved session-stop export")
    parser.add_argument("--resume-archive", help="Output archive filename; defaults to training-export.zip for full runs")
    parser.add_argument("--session-seconds", type=int, default=39600,
                        help="Available trainer wall time after setup; verify current account limit")
    args = parser.parse_args(argv)
    if args.command == "launch" and args.mode.startswith("phase04"):
        if not args.condition or not args.split_hash or not re.fullmatch(r"[0-9a-f]{64}", args.split_hash):
            parser.error("Phase 4 requires --condition and --split-hash (64 lowercase hex characters)")
    directory = Path(args.directory).resolve()
    if args.command == "package":
        stage = package_source(directory)
        write_json(stage / "dataset-metadata.json", dict(title="RL Racing Frozen Training Source", id=f"{args.username}/{args.slug}-source",
                   licenses=[{"name": "other"}]))
        print(stage)
        return 0
    cli = kaggle_command()
    notebook_id = f"{args.username}/{args.slug}"
    if args.command == "launch":
        if not args.dataset:
            parser.error("launch requires --dataset")
        directory.mkdir(parents=True, exist_ok=False)
        notebook = json.loads((ROOT / "notebooks/kaggle_train.ipynb").read_text(encoding="utf-8"))
        parameters = notebook["cells"][1]
        archive = args.resume_archive or ("training-export.zip" if args.mode in ("full", "phase04-full", "phase04-smoke") else "session-stop.zip")
        parameters["source"] = [f"MODE = {args.mode!r}\n", f"MAX_SESSION_SECONDS = {args.session_seconds}\n", f"RESUME_ARCHIVE = {archive!r}\n" if args.resume_kernel else "RESUME_ARCHIVE = None\n",
                                f"CONDITION = {args.condition!r}\n", f"SPLIT_MANIFEST_SHA256 = {args.split_hash!r}\n"]
        (directory / "kaggle_train.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        write_json(directory / "kernel-metadata.json", dict(id=notebook_id, title=args.slug.replace("-", " ").title(),
            code_file="kaggle_train.ipynb", language="python", kernel_type="notebook", is_private=True,
            enable_gpu=False, enable_internet=True, dataset_sources=[args.dataset], competition_sources=[],
            kernel_sources=[args.resume_kernel] if args.resume_kernel else []))
        result = subprocess.run(cli + ["kernels", "push", "-p", str(directory)], check=True,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONUTF8": "1"})
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if "could not be added" in result.stdout + result.stderr:
            raise ValueError("Kaggle rejected an input source; wait until the parent notebook output is saved before launching resume")
    elif args.command == "status":
        subprocess.run(cli + ["kernels", "status", notebook_id], check=True)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(cli + ["kernels", "output", notebook_id, "-p", str(directory),
                             "--file-pattern", r"(^|/)(training-export|session-stop)\.(zip|inventory\.json)$",
                             "--page-size", "200"], check=True, env={**os.environ, "PYTHONUTF8": "1"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
