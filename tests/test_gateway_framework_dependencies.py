"""Keep the gateway's installed HTTP framework on the patched compatible pair."""

from importlib.metadata import requires, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPOSITORY = Path(__file__).resolve().parents[1]
PATCHED = {"fastapi": "0.141.1", "starlette": "1.6.0"}


def _requirement(distribution: str, dependency: str) -> Requirement:
    matches = [
        Requirement(value)
        for value in requires(distribution) or ()
        if Requirement(value).name.lower() == dependency
    ]
    assert len(matches) == 1
    return matches[0]


def test_host_requirements_pin_the_tested_patched_pair():
    # The canonical gateway restart installs the candidate requirements.txt.
    declared = [
        Requirement(line.split(" #", 1)[0])
        for line in (REPOSITORY / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for package, release in PATCHED.items():
        matches = [item for item in declared if item.name.lower() == package]
        assert len(matches) == 1
        assert matches[0].specifier == SpecifierSet(f"=={release}")


def test_installed_pair_is_patched_and_satisfies_bittensor_and_fastapi():
    for package, release in PATCHED.items():
        assert Version(version(package)) == Version(release)
    assert Version(version("starlette")) in _requirement("fastapi", "starlette").specifier
    assert Version(version("fastapi")) in _requirement("bittensor", "fastapi").specifier


def test_installed_pair_serves_real_gateway_health_and_unknown_route():
    # Do not enter the production lifespan: it requires a measured enclave and
    # chain. The actual gateway middleware, router, and response stack still run.
    from fastapi.testclient import TestClient
    from gateway.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    assert client.get("/not-a-gateway-route").status_code == 404
