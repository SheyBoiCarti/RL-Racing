"""Fail-closed frozen split catalogs, with a separate structural audit."""
import copy
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from racing.core.config import PROJECT_ROOT, project_path
from racing.tracks.generator import GeneratorConfig, canonical_hash
from racing.tracks.validation import validate_generated


def manifest_hash(data):
    return canonical_hash({k: v for k, v in data.items() if k != 'manifest_sha256'})


def _validate_track_id(entry):
    if not re.fullmatch(re.escape(entry['split']) + r'_[0-9]{3,}', entry['track_id']):
        raise ValueError(f"track_id: safe split-specific identifier required: {entry['track_id']}")


def safe_track_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError(f'path: project-root-relative path required: {value}')
    resolved = (PROJECT_ROOT/path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f'path: escapes project root: {value}')
    return resolved


def _read_manifest(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'manifest: duplicate field {key}')
            result[key] = value
        return result
    data = json.loads(project_path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite: {value}')))
    schema = json.loads((PROJECT_ROOT/'docs/specs/schemas/split.schema.json').read_text())
    errors = list(Draft202012Validator(schema).iter_errors(data))
    if errors:
        raise ValueError(f'manifest schema: {errors[0].message}')
    if data['fixture_only'] or data['generator_version'] != 'generator-v1':
        raise ValueError('manifest: production generator-v1 dataset required')
    if manifest_hash(data) != data['manifest_sha256']:
        raise ValueError('manifest_sha256: mismatch')
    config = GeneratorConfig(**data['generator_config'])
    if canonical_hash(asdict(config)) != data['generator_config_hash']:
        raise ValueError('generator_config_hash: mismatch')
    entries = data['entries']
    if dict(Counter(e['split'] for e in entries)) != data['counts']:
        raise ValueError('counts: mismatch')
    for key in ('track_id', 'seed', 'geometry_hash', 'path'):
        if len({e[key] for e in entries}) != len(entries):
            raise ValueError(f'{key}: duplicate across splits')
    indices = Counter()
    for entry in entries:
        _validate_track_id(entry)
        expected = f"{entry['split']}_{indices[entry['split']]:03d}"
        if entry['track_id'] != expected:
            raise ValueError(f"track_id: expected seed-order identifier {expected}")
        indices[entry['split']] += 1
        safe_track_path(entry['path'])
    return data, config


@lru_cache(maxsize=256)
def _validated_content(content, config_json, entry_json, config_hash):
    config = GeneratorConfig(**json.loads(config_json))
    entry = json.loads(entry_json)
    _validate_track_id(entry)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'track: duplicate field {key}')
            result[key] = value
        return result
    data = json.loads(content, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite: {value}')))
    schema = json.loads(Path(__file__).with_name('track.schema.json').read_text())
    errors = list(Draft202012Validator(schema).iter_errors(data))
    if errors:
        raise ValueError(f'track schema: {errors[0].message}')
    provenance = dict(version=config.version, seed=entry['seed'], attempt=entry['attempt'], config_hash=config_hash)
    if data['generator'] != provenance or data['split'] != entry['split'] or data['track_id'] != entry['track_id']:
        raise ValueError(f"{entry['path']}: identity/provenance mismatch")
    track, metrics = validate_generated(data, config)
    if canonical_hash(track.geometry_payload) != entry['geometry_hash']:
        raise ValueError(f"{entry['path']}: geometry_hash mismatch")
    for key in ('length_m', 'max_curvature'):
        if not math.isclose(metrics[key], entry[key], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{entry['path']}: {key} mismatch")
    return track


def _load_entry(entry, config, config_hash):
    # Read actual bytes on every call. Cache keys include every identity,
    # configuration and file byte; callers receive independent mutable objects.
    content = safe_track_path(entry['path']).read_bytes()
    return copy.deepcopy(_validated_content(content, json.dumps(asdict(config), sort_keys=True),
                                            json.dumps(entry, sort_keys=True), config_hash))


def load_manifest(path):
    """Audit schema, integrity and actual geometry of every split (no inference)."""
    data, config = _read_manifest(path)
    for entry in data['entries']:
        _load_entry(entry, config, data['generator_config_hash'])
    return data


def load_split_catalog(manifest_path, split):
    """Return only explicitly allowlisted train/validation tracks."""
    if split not in ('train', 'validation'):
        raise ValueError('split: only train/validation catalogs permitted; test is reserved for Phase 6')
    data, config = _read_manifest(manifest_path)
    # Every identity and path is checked globally; geometry is loaded solely
    # from the requested allowlist, never by scanning a directory.
    return {e['track_id']: _load_entry(e, config, data['generator_config_hash'])
            for e in data['entries'] if e['split'] == split}
