"""Auditors mirror the live epoch from staged (publication-first) authority.

The primary's chain finalization completes shortly after the epoch boundary,
so a finalized-only view is typically one epoch behind the auditor's
submission window. The staged view serves the enclave-signed bundle plus the
durable gateway publication as soon as they exist, attaches the
finalized-chain proof once it lands, and leaves the finalized-only
/v2/latest contract untouched.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import leadpoet_canonical.auditor_v2 as auditor_v2


# ---------------------------------------------------------------------------
# Canonical staged verifier: stage gating and routing
# ---------------------------------------------------------------------------


def _staged_authority(stage, finalization):
    return {
        "schema_version": "leadpoet.published_weight_authority_stage.v2",
        "authority_stage": stage,
        "bundle": {"kind": "bundle"},
        "publication": {"kind": "publication"},
        "finalization": finalization,
    }


@pytest.fixture()
def routed(monkeypatch):
    calls = {}

    def fake_bundle_verify(bundle, *, identity_cache, boot_verifier=None):
        calls["bundle"] = bundle
        return {"bundle_hash": "sha256:" + "a" * 64}

    def fake_tail(**kwargs):
        calls["finalization_arg"] = kwargs["finalization"]
        return {"epoch_id": 24084, "netuid": 71}

    monkeypatch.setattr(
        auditor_v2, "verify_attested_weight_bundle_v2", fake_bundle_verify
    )
    monkeypatch.setattr(
        auditor_v2, "_verify_publication_and_finalization", fake_tail
    )
    return calls


def test_published_stage_verifies_without_finalization(routed):
    verified = auditor_v2.verify_published_weight_authority_stage_v2(
        _staged_authority("published", None),
        identity_cache={},
        chain_signing_profile={},
    )
    assert routed["finalization_arg"] is None
    assert verified["authority_stage"] == "published"


def test_finalized_stage_passes_proof_through(routed):
    proof = {"kind": "finalization"}
    verified = auditor_v2.verify_published_weight_authority_stage_v2(
        _staged_authority("finalized", proof),
        identity_cache={},
        chain_signing_profile={},
    )
    assert routed["finalization_arg"] == proof
    assert verified["authority_stage"] == "finalized"


def test_published_stage_must_not_carry_finalization(routed):
    with pytest.raises(auditor_v2.AuditorV2Error):
        auditor_v2.verify_published_weight_authority_stage_v2(
            _staged_authority("published", {"kind": "finalization"}),
            identity_cache={},
            chain_signing_profile={},
        )


def test_finalized_stage_requires_its_proof(routed):
    with pytest.raises(auditor_v2.AuditorV2Error):
        auditor_v2.verify_published_weight_authority_stage_v2(
            _staged_authority("finalized", None),
            identity_cache={},
            chain_signing_profile={},
        )


def test_unknown_stage_fails_closed(routed):
    with pytest.raises(auditor_v2.AuditorV2Error):
        auditor_v2.verify_published_weight_authority_stage_v2(
            _staged_authority("committed", None),
            identity_cache={},
            chain_signing_profile={},
        )


# ---------------------------------------------------------------------------
# Store loader: staged payload shapes, legacy shape untouched
# ---------------------------------------------------------------------------


_DEFAULT_PUBLICATION = object()


def _store(
    monkeypatch,
    *,
    finalization_row,
    publication_row=_DEFAULT_PUBLICATION,
):
    from gateway.research_lab import attested_v2_store as store

    async def fake_bundle(**_kwargs):
        return {"kind": "bundle"}

    def fake_validate(bundle):
        return {"bundle_hash": "sha256:" + "b" * 64}

    async def fake_select_one(table, filters):
        if table == store.PUBLICATION_TABLE:
            if publication_row is _DEFAULT_PUBLICATION:
                return {
                    "weight_submission_event_hash": "sha256:" + "c" * 64,
                    "publication_receipt_hash": "sha256:" + "d" * 64,
                    "publication_doc": {"kind": "publication-doc"},
                }
            return publication_row
        if table == store.FINALIZATION_TABLE:
            return finalization_row
        raise AssertionError(table)

    async def fake_graph(_receipt_hash):
        return {"kind": "graph"}

    monkeypatch.setattr(store, "load_weight_bundle_v2", fake_bundle)
    monkeypatch.setattr(
        store, "validate_published_weight_bundle_v2", fake_validate
    )
    monkeypatch.setattr(store, "select_one", fake_select_one)
    monkeypatch.setattr(store, "load_receipt_graph_v2", fake_graph)
    return store


def test_store_returns_published_stage_before_finalization(monkeypatch):
    store = _store(monkeypatch, finalization_row=None)
    authority = asyncio.run(
        store.load_weight_authority_v2(
            netuid=71,
            epoch_id=24084,
            validator_hotkey="hk",
            require_finalization=False,
        )
    )
    assert authority["authority_stage"] == "published"
    assert authority["finalization"] is None
    assert authority["publication"]["receipt_graph"] == {"kind": "graph"}


def test_store_keeps_finalized_only_contract_by_default(monkeypatch):
    store = _store(monkeypatch, finalization_row=None)
    authority = asyncio.run(
        store.load_weight_authority_v2(
            netuid=71,
            epoch_id=24084,
            validator_hotkey="hk",
        )
    )
    assert authority is None


def test_store_treats_missing_staged_publication_as_not_ready(monkeypatch):
    store = _store(
        monkeypatch,
        publication_row=None,
        finalization_row=None,
    )
    authority = asyncio.run(
        store.load_weight_authority_v2(
            netuid=71,
            epoch_id=24084,
            validator_hotkey="hk",
            require_finalization=False,
        )
    )
    assert authority is None


def test_store_keeps_missing_publication_fail_closed_for_finalized_contract(
    monkeypatch,
):
    store = _store(
        monkeypatch,
        publication_row=None,
        finalization_row=None,
    )
    with pytest.raises(
        store.AttestedV2StoreError,
        match="V2 bundle publication is missing",
    ):
        asyncio.run(
            store.load_weight_authority_v2(
                netuid=71,
                epoch_id=24084,
                validator_hotkey="hk",
            )
        )


def test_store_upgrades_staged_payload_once_finalized(monkeypatch):
    finalization_row = {
        "weight_finalization_event_hash": "sha256:" + "e" * 64,
        "finalization_receipt_hash": "sha256:" + "f" * 64,
        "finalization_doc": {"kind": "finalization-doc"},
    }
    store = _store(monkeypatch, finalization_row=finalization_row)
    staged = asyncio.run(
        store.load_weight_authority_v2(
            netuid=71,
            epoch_id=24084,
            validator_hotkey="hk",
            require_finalization=False,
        )
    )
    assert staged["authority_stage"] == "finalized"
    assert staged["finalization"]["submission"]["finalization"] == {
        "kind": "finalization-doc"
    }
    legacy = asyncio.run(
        store.load_weight_authority_v2(
            netuid=71,
            epoch_id=24084,
            validator_hotkey="hk",
        )
    )
    assert set(legacy) == {
        "schema_version",
        "bundle",
        "publication",
        "finalization",
    }
    assert legacy["schema_version"] == "leadpoet.published_weight_authority.v2"


# ---------------------------------------------------------------------------
# Auditor fetch: staged route preferred, legacy fallback, dispatch by stage
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _fetch_auditor(monkeypatch, responses):
    import neurons.auditor_validator as auditor_module

    session = _FakeSession(responses)
    monkeypatch.setattr(
        auditor_module.aiohttp,
        "ClientSession",
        lambda trust_env=False: session,
    )
    auditor = SimpleNamespace(
        gateway_url="http://gw",
        config=SimpleNamespace(netuid=71),
    )
    fetch = auditor_module.AuditorValidator.fetch_attested_weights_v2
    return auditor_module, auditor, session, fetch


def test_fetch_prefers_staged_route(monkeypatch):
    _, auditor, session, fetch = _fetch_auditor(
        monkeypatch,
        [_FakeResponse(200, {"authority_stage": "published"})],
    )
    value = asyncio.run(fetch(auditor, 24084))
    assert value == {"authority_stage": "published"}
    assert session.urls == ["http://gw/weights/v2/published/71/24084"]


def test_fetch_falls_back_when_gateway_predates_staged_route(monkeypatch):
    _, auditor, session, fetch = _fetch_auditor(
        monkeypatch,
        [
            _FakeResponse(404, {"detail": "Not Found"}),
            _FakeResponse(200, {"schema_version": "legacy"}),
        ],
    )
    value = asyncio.run(fetch(auditor, 24084))
    assert value == {"schema_version": "legacy"}
    assert session.urls == [
        "http://gw/weights/v2/published/71/24084",
        "http://gw/weights/v2/latest/71/24084",
    ]
    assert auditor._last_v2_authority_was_absent is False


def test_fetch_marks_absent_authority(monkeypatch):
    _, auditor, session, fetch = _fetch_auditor(
        monkeypatch,
        [
            _FakeResponse(
                404, {"detail": "published v2 weight authority not found"}
            )
        ],
    )
    value = asyncio.run(fetch(auditor, 24084))
    assert value is None
    assert auditor._last_v2_authority_was_absent is True
    assert session.urls == ["http://gw/weights/v2/published/71/24084"]


def test_verify_dispatches_staged_authority(monkeypatch, tmp_path):
    import neurons.auditor_validator as auditor_module

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"version_key": 1}), encoding="utf-8")
    monkeypatch.setenv("AUDITOR_CHAIN_SIGNING_PROFILE_FILE", str(profile_path))
    calls = {}

    def staged_verify(authority, *, identity_cache, chain_signing_profile):
        calls["staged"] = authority
        return {"epoch_id": 24084, "netuid": 71}

    def finalized_verify(authority, *, identity_cache, chain_signing_profile):
        calls["finalized"] = authority
        return {"epoch_id": 24084, "netuid": 71}

    monkeypatch.setattr(
        auditor_module,
        "verify_published_weight_authority_stage_v2",
        staged_verify,
    )
    monkeypatch.setattr(
        auditor_module,
        "verify_attested_weight_authority_v2",
        finalized_verify,
    )
    auditor = SimpleNamespace(_verify_stateful_bundle_epoch=lambda _v: None)
    verify = auditor_module.AuditorValidator.verify_attested_weights_v2

    staged = {"authority_stage": "published"}
    assert verify(auditor, staged, identity_cache={}) is not None
    assert calls["staged"] is staged

    legacy = {"schema_version": "leadpoet.published_weight_authority.v2"}
    assert verify(auditor, legacy, identity_cache={}) is not None
    assert calls["finalized"] is legacy
