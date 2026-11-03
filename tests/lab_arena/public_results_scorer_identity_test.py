"""The lean published read retains the frozen judgment image authority."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lab_arena.service import ServiceError
from lab_arena.store import ArenaStore
from tests.lab_arena import public_result_company_evidence_reads_test as company_reads
from tests.lab_arena import scoring_validator_attribution_test as attribution
from tests.lab_arena.public_results_projection_test import RoundTransport


ROUND = "arena-2026-11-03"
IDENTITY_FIELDS = ("scorer_image_digest", "scorer_image_reference")


def published_company_service(monkeypatch, *, preserved):
    monkeypatch.setattr(attribution, "ROUND_ID", ROUND)
    monkeypatch.setattr(company_reads, "ROUND_ID", ROUND)
    service, ref, _, cached, _, objects, _ = company_reads._public_company_result()
    company_reads._make_current_score_the_authority(service, ref, cached, objects)
    row = service._round(ROUND)
    row["scoring_generation_floor"] = 0
    configuration = row["configuration_doc"]
    configuration.update(
        mode="live", network_name="finney", netuid=71,
        scorer_image_digest=ref["scope_doc"]["scorer_image_digest"],
        scorer_image_reference=ref["scope_doc"]["scorer_image_reference"],
        deepline_catalog={"private-catalog": "must-stay-private"},
        private_setting="private-configuration-marker",
    )
    original = deepcopy(configuration)
    for run in service._store.rows.values():
        if run["status"] == "accepted":
            run["terminal_cause"] = "accepted"
    if preserved:
        configuration.update(scorer_image_digest="sha256:" + "d" * 64,
                             scorer_image_reference="registry/scorer@sha256:" + "d" * 64)
    proof = {
        "operation": "nov03_forward_only_verifier_v1", "scoring_generation_floor": 0,
        **{key: original[key] for key in IDENTITY_FIELDS + ("scorer_policy",)},
        **{"current_" + key: configuration[key] for key in IDENTITY_FIELDS},
    }
    service._store.preserved_judgment_policy = Mock(return_value=proof if preserved else None)
    service._config = SimpleNamespace(mode="live", pinned_round_id=None,
                                      network_name="finney", netuid=71)
    service._chain_scope = lambda: ("finney", 71)
    return service, row, proof


def install_projection(service, row):
    transport = RoundTransport(row)
    service._store.get_published_results_round = ArenaStore(transport).get_published_results_round
    service._round = lambda _round_id: pytest.fail("published results fetched a full round")
    return transport


@pytest.mark.parametrize("preserved", [False, True])
def test_real_published_results_keep_current_and_preserved_judgment_authority(monkeypatch, preserved):
    service, row, _ = published_company_service(monkeypatch, preserved=preserved)
    expected = service.public_results(ROUND, attribution.SUBMISSION_ID)
    service._store.preserved_judgment_policy.reset_mock()
    transport = install_projection(service, row)
    actual = service.public_results(ROUND, attribution.SUBMISSION_ID)
    assert actual == expected
    assert actual["scoring_attribution"]["unattributed_icp_count"] == 0
    for key in IDENTITY_FIELDS:
        assert json.loads(transport.returned[0]["cfg_" + key]) == row["configuration_doc"][key]
    assert service._store.preserved_judgment_policy.called is preserved
    assert "configuration_doc" not in transport.returned[0]
    for marker in ("private-catalog", "private-configuration-marker"):
        assert marker not in json.dumps(transport.returned)
        assert marker not in json.dumps(actual)
    for key in IDENTITY_FIELDS:
        assert key not in json.dumps(actual)
    assert row["configuration_doc"]["private_setting"] == "private-configuration-marker"


@pytest.mark.parametrize("mutation", ["missing_authority", "wrong_current_digest", "wrong_current_reference"])
def test_published_projection_keeps_preserved_authority_fail_closed(monkeypatch, mutation):
    service, row, proof = published_company_service(monkeypatch, preserved=True)
    if mutation == "missing_authority":
        service._store.preserved_judgment_policy.return_value = None
        error = "public_company_verification_unavailable"
    else:
        key = "scorer_image_digest" if mutation == "wrong_current_digest" else "scorer_image_reference"
        proof["current_" + key] = "wrong-binding"
        error = "preserved_judgment_policy_invalid"
    install_projection(service, row)
    with pytest.raises(ServiceError, match=error):
        service.public_results(ROUND, attribution.SUBMISSION_ID)


@pytest.mark.parametrize("key", IDENTITY_FIELDS)
def test_published_projection_does_not_accept_a_wrong_current_image_binding(monkeypatch, key):
    service, row, _ = published_company_service(monkeypatch, preserved=False)
    row["configuration_doc"][key] = "wrong-image-binding"
    install_projection(service, row)
    with pytest.raises(ServiceError, match="public_company_verification_unavailable"):
        service.public_results(ROUND, attribution.SUBMISSION_ID)
