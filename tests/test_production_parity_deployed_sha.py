import json
from pathlib import Path
import subprocess

import pytest

from scripts.resolve_production_parity_deployed_sha import (
    DeployedShaResolutionError,
    resolve_deployed_sha,
)


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, value: object, *, status: int = 200) -> None:
        self.status = status
        self._payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self._payload


def _opener(value: object, *, status: int = 200):
    def open_request(request, *, timeout: int):
        assert request.full_url == "https://gateway.example/build-info"
        assert timeout == 10
        return _Response(value, status=status)

    return open_request


def test_deployed_sha_uses_exact_public_build_identity() -> None:
    candidate = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    resolved = resolve_deployed_sha(
        root=ROOT,
        gateway_url="https://gateway.example/",
        candidate_sha=candidate,
        opener=_opener(
            {
                "git_commit": candidate,
                "is_commit_known": True,
            }
        ),
    )
    assert resolved == candidate


@pytest.mark.parametrize(
    "gateway_url",
    (
        "http://gateway.example",
        "https://user:secret@gateway.example",
        "https://gateway.example/path",
        "https://gateway.example/?candidate=bad",
    ),
)
def test_deployed_sha_rejects_noncanonical_gateway_boundary(
    gateway_url: str,
) -> None:
    candidate = "a" * 40
    with pytest.raises(
        DeployedShaResolutionError,
        match="HTTPS build-info boundary",
    ):
        resolve_deployed_sha(
            root=ROOT,
            gateway_url=gateway_url,
            candidate_sha=candidate,
            opener=_opener({}),
        )


def test_deployed_sha_requires_known_exact_commit() -> None:
    with pytest.raises(
        DeployedShaResolutionError,
        match="exact build identity",
    ):
        resolve_deployed_sha(
            root=ROOT,
            gateway_url="https://gateway.example",
            candidate_sha="a" * 40,
            opener=_opener(
                {
                    "git_commit": "unknown",
                    "is_commit_known": False,
                }
            ),
        )
