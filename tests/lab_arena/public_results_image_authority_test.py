"""Published reads retain internal grading identities for November 3 authority."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStore
from tests.lab_arena import public_result_company_evidence_reads_test as evidence_tests
from tests.lab_arena import scoring_validator_attribution_test as attribution_tests
from tests.lab_arena.public_results_projection_test import RoundTransport


ROUND_ID = "arena-2026-11-03"
IMAGE_FIELDS = ("scorer_image_digest", "scorer_image_reference")


def _published_service(monkeypatch, *, preserved):
    # Build the existing real immutable-evidence fixture for this round.
    monkeypatch.setattr(evidence_tests, "ROUND_ID", ROUND_ID)
    monkeypatch.setattr(attribution_tests, "ROUND_ID", ROUND_ID)
    service, ref, _, _, _, _, _ = evidence_tests._public_company_result()
    row = service._round(ROUND_ID)
    configuration = row["configuration_doc"]
    configuration.update(mode="live", network_name="finney", netuid=71)
    configuration.update({key: ref["scope_doc"][key] for key in IMAGE_FIELDS})
    original = deepcopy(configuration)
    if preserved:
        configuration.update(
            scorer_image_digest="sha256:" + "d" * 64,
            scorer_image_reference="registry/scorer@sha256:" + "d" * 64,
        )
    participant = row["publication_doc"]["participants"][0]
    participant.update(is_baseline=preserved, is_king=preserved)
    row["participants"] = [deepcopy(participant)]
    service._store.get_submission = lambda _submission_id: {"is_king": preserved}
    judge = service._store.rows["score-current"]
    judge["terminal_cause"] = "accepted"
    authority = {
        "operation": "nov03_forward_only_verifier_v1",
        "scoring_generation_floor": 0,
        **{key: original[key] for key in (*IMAGE_FIELDS, "scorer_policy")},
        **{"current_" + key: configuration[key] for key in IMAGE_FIELDS},
    }
    service._store.preserved_judgment_policy = Mock(
        return_value=authority if preserved else None,
    )
    service._config = SimpleNamespace(
        mode="live", pinned_round_id=None, network_name="finney", netuid=71,
    )
    service._chain_scope = lambda: ("finney", 71)
    transport = RoundTransport(row)
    return service, row, authority, transport


def _use_projection(service, transport):
    service._store.get_published_results_round = ArenaStore(transport).get_published_results_round
    service._round = lambda _round_id: pytest.fail("published results fetched full round")


@pytest.mark.parametrize("preserved", [False, True], ids=["current-miner", "preserved-baseline"])
def test_published_results_keep_full_row_authority_and_public_output(monkeypatch, preserved):
    service, row, _, transport = _published_service(monkeypatch, preserved=preserved)
    before = deepcopy((row, service._store.rows))
    expected = service.public_results(ROUND_ID, attribution_tests.SUBMISSION_ID)
    _use_projection(service, transport)
    with TestClient(create_app(service)) as client:
        response = client.get(
            "/arena/v1/rounds/%s/results/%s" % (ROUND_ID, attribution_tests.SUBMISSION_ID),
        )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    actual = response.json()
    assert actual == expected
    assert actual["company_diagnostics"]
    assert actual["scoring_attribution"]["unattributed_icp_count"] == 0
    assert (row, service._store.rows) == before
    encoded = json.dumps(actual)
    for key in IMAGE_FIELDS:
        assert key not in encoded
        assert row["configuration_doc"][key] not in encoded
        assert "cfg_%s:configuration_doc->%s::text" % (key, key) in transport.calls[0]["columns"]


@pytest.mark.parametrize("field,value", [
    ("operation", "unapproved-regrade"),
    ("scoring_generation_floor", 1),
    ("current_scorer_image_digest", "sha256:" + "f" * 64),
    ("current_scorer_image_reference", "registry/scorer@sha256:" + "f" * 64),
])
def test_published_preserved_authority_still_rejects_changed_binding(monkeypatch, field, value):
    service, _, authority, transport = _published_service(monkeypatch, preserved=True)
    authority[field] = value
    _use_projection(service, transport)
    with pytest.raises(ServiceError, match="preserved_judgment_policy_invalid"):
        service.public_results(ROUND_ID, attribution_tests.SUBMISSION_ID)


@pytest.mark.parametrize("field", IMAGE_FIELDS)
def test_published_current_judgment_still_rejects_foreign_image(monkeypatch, field):
    service, row, _, transport = _published_service(monkeypatch, preserved=False)
    row["configuration_doc"][field] = "sha256:" + "f" * 64
    _use_projection(service, transport)
    with pytest.raises(ServiceError, match="public_company_verification_unavailable"):
        service.public_results(ROUND_ID, attribution_tests.SUBMISSION_ID)
