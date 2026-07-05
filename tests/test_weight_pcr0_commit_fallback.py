from __future__ import annotations

import json

from gateway.api import weights


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


def test_extract_commit_hash_from_explicit_allowlist_field():
    entry = {
        "pcr0": "0" * 96,
        "commit_hash": "9e2f22846d8a0fb4a3ff1c73f821fed8f127b40e",
        "notes": "validator enclave: current production build on commit badcafe",
    }

    assert (
        weights._extract_commit_hash_from_allowlist_entry(entry)
        == "9e2f22846d8a0fb4a3ff1c73f821fed8f127b40e"
    )


def test_extract_commit_hash_from_notes_fallback():
    entry = {
        "pcr0": "0" * 96,
        "notes": "validator enclave: current production build on commit 9e2f228",
    }

    assert weights._extract_commit_hash_from_allowlist_entry(entry) == "9e2f228"


def test_lookup_pcr0_commit_hash_from_remote_allowlist(monkeypatch):
    pcr0 = "01877a6d4fd0f4b6f335e7816af36072c63e141ed40e9b100703f467a1feca740cdbca2fd29196e0808d80d50fee5104"
    payload = {
        "validator_pcr0": [
            {
                "pcr0": pcr0,
                "commit_hash": "9e2f22846d8a0fb4a3ff1c73f821fed8f127b40e",
            }
        ]
    }

    def fake_urlopen(request, timeout):
        assert timeout == 10
        return _FakeResponse(payload)

    monkeypatch.setattr(weights.urllib.request, "urlopen", fake_urlopen)

    assert (
        weights._lookup_pcr0_commit_hash_from_allowlist(pcr0)
        == "9e2f22846d8a0fb4a3ff1c73f821fed8f127b40e"
    )


def test_lookup_pcr0_commit_prefers_structured_local_commit_over_remote_notes(
    monkeypatch, tmp_path
):
    pcr0 = "01877a6d4fd0f4b6f335e7816af36072c63e141ed40e9b100703f467a1feca740cdbca2fd29196e0808d80d50fee5104"
    remote_payload = {
        "validator_pcr0": [
            {
                "pcr0": pcr0,
                "notes": "validator enclave: current production build on commit 9e2f228",
            }
        ]
    }
    local_payload = {
        "validator_pcr0": [
            {
                "pcr0": pcr0,
                "commit_hash": "9e2f22846d8a0fb4a3ff1c73f821fed8f127b40e",
            }
        ]
    }
    (tmp_path / "pcr0_allowlist.json").write_text(json.dumps(local_payload), encoding="utf-8")

    def fake_urlopen(request, timeout):
        return _FakeResponse(remote_payload)

    monkeypatch.setattr(weights.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.chdir(tmp_path)

    assert (
        weights._lookup_pcr0_commit_hash_from_allowlist(pcr0)
        == "9e2f22846d8a0fb4a3ff1c73f821fed8f127b40e"
    )
