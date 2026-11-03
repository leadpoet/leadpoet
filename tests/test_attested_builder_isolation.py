import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attested_builder_isolation", ROOT / ".github/scripts/verify_attested_builder_isolation.py"
)
ISOLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ISOLATION)


def test_fresh_runner_allows_only_normal_github_artifact_endpoints():
    ISOLATION.verify_isolation({})
    ISOLATION.verify_isolation({
        "ACTIONS_CACHE_URL": "https://cache.actions.githubusercontent.com/cache/",
        "ACTIONS_RESULTS_URL": "https://results-receiver.actions.githubusercontent.com/",
    })


@pytest.mark.parametrize("name", [
    "RUNS_ON_POOL_NAME", "RUNS_ON_POOL_STANDBY_TYPE",
    "RUNS_ON_STICKYDISK_DIR", "RUNS_ON_STICKYDISK_NAME",
    "RUNS_ON_STICKYDISK_READY_FILE", "RUNS_ON_STICKYDISK_UNAVAILABLE_FILE", "RUNS_ON_EFS_ID",
])
def test_persistent_or_warm_runner_features_fail_closed(name):
    with pytest.raises(ValueError, match="feature is forbidden"):
        ISOLATION.verify_isolation({name: "enabled"})


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:5000/cache", "https://cache.example.invalid/",
    "https://actions.githubusercontent.com.attacker.invalid/",
    "http://cache.actions.githubusercontent.com/",
])
def test_shared_cache_broker_is_rejected_without_printing_endpoint(url):
    with pytest.raises(ValueError, match="cache broker is forbidden") as error:
        ISOLATION.verify_isolation({"ACTIONS_CACHE_URL": url})
    assert url not in str(error.value)
