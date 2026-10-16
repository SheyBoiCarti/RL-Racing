"""The versioned, bounded NumPy Catmull–Rom track recipe."""
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields

import numpy as np
import yaml

from racing.core.config import project_path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class GeneratorConfig:
    version: str = 'generator-v1'
    control_points: int = 16
    base_radius_m: float = 90.0
    radial_variation: float = .18
    angular_jitter_fraction: float = .20
    width_m: float = 12.0
    resample_spacing_m: float = 4.0
    length_min_m: float = 400.0
    length_max_m: float = 800.0
    max_abs_curvature_per_m: float = .10
    max_heading_change_rad: float = .35
    min_road_clearance_m: float = 2.0
    checkpoint_spacing_target_m: float = 60.0
    max_attempts: int = 100

    def __post_init__(self):
        if self.version != 'generator-v1' or type(self.control_points) is not int or self.control_points != 16:
            raise ValueError('generator-v1 requires 16 control points')
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == 'version':
                continue
            zero_allowed = f.name in ('radial_variation', 'angular_jitter_fraction', 'min_road_clearance_m')
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
                raise ValueError(f'{f.name}: finite number in allowed positive/nonnegative range required')
            if f.name not in ('control_points', 'max_attempts'):
                object.__setattr__(self, f.name, float(value))
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100:
            raise ValueError('max_attempts: integer in [1,100] required')
        if self.angular_jitter_fraction >= .45 or self.radial_variation > .35:
            raise ValueError('jitter/variation outside generator-v1 limits')
        if self.length_min_m > self.length_max_m:
            raise ValueError('length bounds reversed')


def load_generator_config(path='configs/generator.yaml'):
    data = yaml.safe_load(project_path(path).read_text(encoding='utf-8'))
    pools = data.pop('seed_pools', {'train': [10000, 60], 'validation': [20000, 15], 'test': [30000, 25]})
    return GeneratorConfig(**data), pools


def catmull_rom(points):
    p0, p1, p2, p3 = (np.roll(points, shift, axis=0)[:, None, :]
                       for shift in (1, 0, -1, -2))
    t = np.arange(64)[None, :, None] / 64
    return (.5 * (2*p1 + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t*t
                  + (-p0+3*p1-3*p2+p3)*t*t*t)).reshape(-1, 2)


def resample(dense, spacing):
    edges = np.roll(dense, -1, axis=0)-dense
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    stations = np.arange(math.ceil(cumulative[-1]/spacing))*cumulative[-1]/math.ceil(cumulative[-1]/spacing)
    indices = np.searchsorted(cumulative, stations, side='right')-1
    result = dense[indices]+edges[indices]*((stations-cumulative[indices])/lengths[indices])[:, None]
    result -= result.mean(axis=0)
    return np.array([[float(round(float(v), 6)) or 0.0 for v in p] for p in result])


class GenerationError(ValueError):
    def __init__(self, seed, rejection_counts):
        self.seed, self.rejection_counts = seed, dict(rejection_counts)
        super().__init__(f'seed {seed}: bounded generation exhausted: {self.rejection_counts}')

    def __reduce__(self):
        return type(self), (self.seed, self.rejection_counts)


def generate_track(seed, config=None, track_id='generated', split='train'):
    from racing.tracks.validation import validate_generated
    config = config or GeneratorConfig()
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError('seed: nonnegative uint32 integer required')
    started, rejected = time.perf_counter(), Counter()
    for attempt in range(config.max_attempts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        k = config.control_points
        jitter = config.angular_jitter_fraction*2*np.pi/k
        angles = 2*np.pi*np.arange(k)/k+rng.uniform(-jitter, jitter, k)
        radii = config.base_radius_m*(1+rng.uniform(-config.radial_variation, config.radial_variation, k))
        order = np.argsort(angles)
        controls = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))[order]
        points = resample(catmull_rom(controls), config.resample_spacing_m)
        length = float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).sum())
        count = max(4, math.ceil(length/config.checkpoint_spacing_target_m))
        data = dict(schema_version='track-v1', track_id=track_id, name=track_id.replace('_', ' ').title(),
                    split=split, width_m=config.width_m, centerline=points.tolist(),
                    checkpoints=[dict(index=i, s_m=float(i*length/count)) for i in range(count)],
                    spawns=[dict(spawn_id=sid, s_m=length-5, lateral_m=lateral)
                            for sid, lateral in [('solo', 0.), ('grid_left', -1.5), ('grid_right', 1.5)]],
                    generator=dict(version=config.version, seed=seed, attempt=attempt, config_hash=canonical_hash(asdict(config))))
        try:
            _, metrics = validate_generated(data, config)
        except ValueError as error:
            rejected[str(error).split(':')[0]] += 1
            continue
        return data, dict(metrics, seed=seed, attempt=attempt, rejection_counts=dict(rejected),
                          runtime_s=time.perf_counter()-started, numpy_version=np.__version__)
    raise GenerationError(seed, rejected)
