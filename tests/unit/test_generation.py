import importlib
import json

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('racing.tracks.generator'), 'generator is required'
    return importlib.import_module('racing.tracks.generator')


def test_cyclic_dense_samples_and_resampling():
    g = module()
    points = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    dense = g.catmull_rom(points)
    assert dense.shape == (256, 2)
    np.testing.assert_allclose(dense[::64], points)
    np.testing.assert_allclose(dense[32], [.625, .625])
    sampled = g.resample(dense, .1)
    perimeter = np.linalg.norm(np.roll(dense, -1, axis=0)-dense, axis=1).sum()
    assert len(sampled) == int(np.ceil(perimeter/.1))
    assert np.max(np.abs(sampled.mean(axis=0))) < 1e-6
    assert np.array_equal(sampled, np.round(sampled, 6))
    assert not np.array_equal(sampled[0], sampled[-1])


def test_seed_determinism_and_bounded_failure():
    g = module()
    config = g.GeneratorConfig()
    first, report = g.generate_track(10000, config)
    second, _ = g.generate_track(10000, config)
    assert first == second
    assert report['validator_version'] == 'geometry-v1'
    with pytest.raises(ValueError, match='seed'):
        g.generate_track(-1, config)
    with pytest.raises(g.GenerationError) as error:
        g.generate_track(1, g.GeneratorConfig(length_min_m=790, max_attempts=2))
    assert sum(error.value.rejection_counts.values()) == 2


def test_config_allows_zero_variation_but_requires_integer_control_count():
    g = module()
    assert g.GeneratorConfig(radial_variation=0, angular_jitter_fraction=0).radial_variation == 0.
    with pytest.raises(ValueError, match='control'):
        g.GeneratorConfig(control_points=16.)


def test_frozen_manifest_and_embargo():
    module()
    from racing.tracks.splits import load_manifest, load_split_catalog
    manifest = load_manifest('experiments/splits/v1.json')
    assert manifest['counts'] == {'train': 60, 'validation': 15, 'test': 25}
    assert len({e['geometry_hash'] for e in manifest['entries']}) == 100
    for split, start, count in [('train', 10000, 60), ('validation', 20000, 15), ('test', 30000, 25)]:
        entries = [e for e in manifest['entries'] if e['split'] == split]
        assert [e['seed'] for e in entries] == list(range(start, start+count))
        assert [e['track_id'] for e in entries] == [f'{split}_{i:03}' for i in range(count)]
    assert len(load_split_catalog('experiments/splits/v1.json', 'validation')) == 15
    with pytest.raises(ValueError, match='test'):
        load_split_catalog('experiments/splits/v1.json', 'test')


@pytest.mark.parametrize('mutation,reason', [
    ('hash', 'manifest_sha256'), ('escape', 'path'), ('count', 'counts'),
    ('seed', 'seed'), ('config', 'generator_config_hash'), ('geometry', 'geometry_hash'),
])
def test_manifest_corruption_is_rejected(tmp_path, mutation, reason):
    from racing.core.config import PROJECT_ROOT
    from racing.tracks.splits import load_split_catalog, manifest_hash
    data = json.loads((PROJECT_ROOT/'experiments/splits/v1.json').read_text())
    if mutation == 'hash':
        data['manifest_sha256'] = '0'*64
    elif mutation == 'escape':
        data['entries'][0]['path'] = '../escape.json'
    elif mutation == 'count':
        data['counts']['train'] -= 1
    elif mutation == 'seed':
        data['entries'][1]['seed'] = data['entries'][0]['seed']
    elif mutation == 'config':
        data['generator_config_hash'] = '0'*64
    elif mutation == 'geometry':
        data['entries'][0]['geometry_hash'] = '0'*64
    if mutation != 'hash':
        data['manifest_sha256'] = manifest_hash(data)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=reason):
        load_split_catalog(path, 'train')


@pytest.mark.parametrize('mutation,reason', [('width', 'width'), ('spawn', 'spawn'),
    ('gate', 'checkpoint'), ('crossing', 'heading|intersection'), ('curvature', 'curvature'),
    ('miter', 'miter'), ('clearance', 'clearance'), ('quantization', 'quantization')])
def test_invalid_geometry_fixtures(mutation, reason):
    from racing.core.config import PROJECT_ROOT
    from racing.tracks.generator import GeneratorConfig
    from racing.tracks.validation import validate_generated
    data = json.loads((PROJECT_ROOT/'assets/tracks/generated/train_000.json').read_text())
    config = GeneratorConfig()
    if mutation == 'width':
        data['width_m'] = 10
    elif mutation == 'spawn':
        data['spawns'][0]['lateral_m'] = 6
    elif mutation == 'gate':
        data['checkpoints'][1]['s_m'] = 1
    elif mutation == 'crossing':
        data['centerline'][20], data['centerline'][80] = data['centerline'][80], data['centerline'][20]
        config = GeneratorConfig(length_max_m=2000)
    elif mutation == 'curvature':
        config = GeneratorConfig(max_abs_curvature_per_m=.001)
    elif mutation == 'miter':
        from racing.tracks.loader import Track
        data['centerline'] = [[0., 0.], [100., 0.], [200., 0.], [300., 0.],
                              [400., 0.], [300., 5.], [200., 10.], [100., 15.]]
        with pytest.raises(ValueError, match='miter'):
            Track(data)
        return
    elif mutation == 'clearance':
        config = GeneratorConfig(min_road_clearance_m=100)
    elif mutation == 'quantization':
        data['centerline'][0][0] += .0000001
    with pytest.raises(ValueError, match=reason):
        validate_generated(data, config)


def test_catalog_cache_checks_actual_bytes_and_isolates_mutation(tmp_path, monkeypatch):
    import racing.tracks.splits as splits
    from racing.core.config import PROJECT_ROOT
    manifest, config = splits._read_manifest('experiments/splits/v1.json')
    entry = dict(manifest['entries'][0], path='track.json')
    source = (PROJECT_ROOT/manifest['entries'][0]['path']).read_bytes()
    (tmp_path/'track.json').write_bytes(source)
    monkeypatch.setattr(splits, 'PROJECT_ROOT', tmp_path)
    first = splits._load_entry(entry, config, manifest['generator_config_hash'])
    first.name = 'mutated'
    first.points[:] = 0
    second = splits._load_entry(entry, config, manifest['generator_config_hash'])
    assert second.name != 'mutated'
    assert np.any(second.points != 0)
    damaged = json.loads(source)
    damaged['centerline'][0][0] += 1
    (tmp_path/'track.json').write_text(json.dumps(damaged))
    with pytest.raises(ValueError):
        splits._load_entry(entry, config, manifest['generator_config_hash'])


def test_generation_failure_leaves_active_dataset_untouched(tmp_path, monkeypatch):
    import scripts.generate_tracks as command
    from racing.tracks.generator import GenerationError
    monkeypatch.setattr(command, 'PROJECT_ROOT', tmp_path)
    output = tmp_path/'generated'
    output.mkdir()
    (output/'existing').write_text('keep')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('keep')
    config = tmp_path/'generator.yaml'
    config.write_text('length_min_m: 790\nmax_attempts: 1\nseed_pools:\n  train: [10000, 1]\n  validation: [20000, 1]\n  test: [30000, 1]\n')
    with pytest.raises(GenerationError):
        command.freeze(config, output, manifest)
    assert manifest.read_text() == 'keep'
    assert (output/'existing').read_text() == 'keep'
    assert not list(tmp_path.glob('.tracks-stage-*'))


def test_geometry_acceleration_preserves_scalar_intersections_and_distances():
    from racing.core.geometry import (
        intersection_candidates,
        intersects,
        segment_distances,
        segment_project,
    )
    rng = np.random.default_rng(19)
    a, b, c, d = rng.normal(size=(4, 20, 2))
    candidates = intersection_candidates(a, b, c, d)
    for i in range(20):
        for j in range(20):
            if intersects(a[i], b[i], c[j], d[j]):
                assert candidates[i, j]
    p = np.array([.25, -.1])
    b[0] = a[0]
    np.testing.assert_allclose(segment_distances(p, a, b),
                               [segment_project(p, aa, bb)[2] for aa, bb in zip(a, b)],
                               rtol=0, atol=1e-15)
