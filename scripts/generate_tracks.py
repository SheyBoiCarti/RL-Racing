"""Freeze the complete dataset; no generation occurs during training startup."""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing.core.config import PROJECT_ROOT, project_path
from racing.tracks.generator import (
    canonical_hash,
    generate_track,
    load_generator_config,
)
from racing.tracks.schema import read_track
from racing.tracks.splits import manifest_hash, safe_track_path


def _generate(args):
    return generate_track(*args)


def freeze(config_path, output, manifest_path):
    config, pools = load_generator_config(config_path)
    output, manifest_path = project_path(output).resolve(), project_path(manifest_path).resolve()
    if not output.is_relative_to(PROJECT_ROOT) or not manifest_path.is_relative_to(PROJECT_ROOT):
        raise ValueError('output/manifest: must stay within project root')
    jobs, seen = [], set()
    if set(pools) != {'train', 'validation', 'test'}:
        raise ValueError('seed_pools: all three splits required')
    for split, (start, count) in pools.items():
        if type(start) is not int or type(count) is not int or count < 1:
            raise ValueError('seed_pools: positive integer counts required')
        for index, seed in enumerate(range(start, start+count)):
            if seed in seen:
                raise ValueError('seed_pools: overlapping seeds')
            seen.add(seed)
            jobs.append((seed, config, f'{split}_{index:03}', split))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.tracks-stage-', dir=output.parent))
    started = time.perf_counter()
    try:
        entries, reports, hashes = [], [], set()
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            for data, report in executor.map(_generate, jobs):
                # Geometry payload normalization exactly matches Track without
                # repeating its expensive topology audit after generation.
                payload = {k: data[k] for k in ('schema_version', 'centerline', 'width_m', 'checkpoints', 'spawns')}
                geometry_hash = canonical_hash(payload)
                if geometry_hash in hashes:
                    raise ValueError('geometry_hash: duplicate generated geometry')
                hashes.add(geometry_hash)
                name = data['track_id']+'.json'
                (stage/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
                path = (output/name).relative_to(PROJECT_ROOT).as_posix()
                safe_track_path(path)
                entries.append(dict(track_id=data['track_id'], split=data['split'], seed=report['seed'],
                                    attempt=report['attempt'], path=path, geometry_hash=geometry_hash,
                                    length_m=report['length_m'], max_curvature=report['max_curvature']))
                reports.append(dict(track_id=data['track_id'], **report))
                print(f"validated {data['track_id']} attempt={report['attempt']}", flush=True)
        manifest = dict(schema_version='split-v1', fixture_only=False, generator_version=config.version,
                        generator_config=asdict(config), generator_config_hash=canonical_hash(asdict(config)),
                        counts={split: pool[1] for split, pool in pools.items()}, entries=entries)
        manifest['manifest_sha256'] = manifest_hash(manifest)
        schema = json.loads((PROJECT_ROOT/'docs/specs/schemas/split.schema.json').read_text())
        errors = list(Draft202012Validator(schema).iter_errors(manifest))
        if errors:
            raise ValueError(f'staged manifest schema: {errors[0].message}')
        for entry in entries:
            saved = read_track(stage/(entry['track_id']+'.json'))
            payload = {key: saved[key] for key in ('schema_version', 'centerline', 'width_m', 'checkpoints', 'spawns')}
            if canonical_hash(payload) != entry['geometry_hash']:
                raise ValueError(f"staged geometry hash: {entry['track_id']}")
        (stage/'generation_report.json').write_text(json.dumps(dict(runtime_s=time.perf_counter()-started,
            validator_version='geometry-v1', tracks=reports), indent=2)+'\n', encoding='utf-8')
        # Roll back both active artifacts on any publication failure. Readers
        # fail closed during the brief two-path replacement window.
        backup = output.with_name(output.name+'.previous')
        if backup.exists():
            raise ValueError(f'publication backup already exists: {backup}')
        previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
        temporary_manifest = stage/'manifest.json'
        temporary_manifest.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
            os.replace(output/'manifest.json', manifest_path)
        except BaseException:
            if output.exists():
                shutil.rmtree(output)
            if backup.exists():
                backup.rename(output)
            if previous_manifest is not None:
                manifest_path.write_bytes(previous_manifest)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/generator.yaml')
    parser.add_argument('--output', default='assets/tracks/generated/')
    parser.add_argument('--manifest', default='experiments/splits/v1.json')
    args = parser.parse_args()
    freeze(args.config, args.output, args.manifest)
