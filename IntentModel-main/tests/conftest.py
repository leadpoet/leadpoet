import pytest
from prometheus_client import REGISTRY

@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    # Remove all collectors before each test
    collectors = list(REGISTRY._names_to_collectors.values())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield 