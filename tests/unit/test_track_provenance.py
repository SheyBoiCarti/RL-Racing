import hashlib
import json

from racing.core.config import project_path


def test_geometry_provenance_matches_source_and_excludes_labels(demo_track):
    raw = json.loads(project_path("assets/tracks/demo.json").read_text())
    payload = demo_track.geometry_payload
    assert set(payload) == {
        "schema_version",
        "centerline",
        "width_m",
        "checkpoints",
        "spawns",
    }
    assert payload == {key: raw[key] for key in payload}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "960352b04ce6f6d45661e0e60b7bc68e1fd90133fd9bd52637dec718affb63c4"
    )
    assert demo_track.split == "demo"
