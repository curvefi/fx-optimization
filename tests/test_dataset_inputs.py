import hashlib
import json

import pytest

from fxopt.config import ConfigError, RunConfig
from fxopt.run import run_metadata


def test_run_records_published_input_identity_and_rejects_changed_bytes(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    market = data / "candles.json"
    contents = b"[[1,100,100,100,100,0]]\n"
    market.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()
    manifest = data / "dataset.json"
    manifest.write_text(json.dumps({
        "format": "fxopt-dataset-v1", "dataset_id": "test-market", "revision": "r1",
        "files": {market.name: {"sha256": expected, "bytes": len(contents)}},
    }))
    config_path = tmp_path / "run.toml"
    config_path.write_text('''[run]
id = "published-input"
evaluator = "evaluator"
template = "template.json"
batch_size = 1
workers = 1
metric_fields = ["score"]
[scenario]
id = "case"
market = "data/candles.json"
[candidate.defaults]
policy_params = []
pool = {}
''')
    config = RunConfig.from_toml(config_path)
    identity = run_metadata(config, effective_batch=1)["datasets"]["market"]
    assert identity == {
        "dataset_id": "test-market", "revision": "r1", "file": "candles.json",
        "sha256": expected, "bytes": len(contents),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    market.write_bytes(contents.replace(b"100", b"101"))
    with pytest.raises(ConfigError):
        run_metadata(config, effective_batch=1)
