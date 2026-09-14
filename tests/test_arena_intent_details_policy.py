"""Frozen activation and contract checks for simplified Arena output v5."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import contracts, intent_details_policy, scoring
from lab_arena.service import ArenaService, RoundDefaults, ServiceConfig, ServiceError
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


def _config(defaults: RoundDefaults, store: object | None = None) -> ServiceConfig:
    return ServiceConfig(
        mode="shadow",
        store=store or object(),
        object_store=object(),
        signer=None,
        chain=object(),
        verify_signature=lambda *_: True,
        daily_icp_source=lambda **_: {},
        banned_hotkeys_source=lambda: (),
        broker_factory=lambda *_: None,
        defaults=defaults,
    )


@pytest.mark.parametrize(
    "integrity_from,intent_details_from",
    [
        (None, "2026-12-01T00:00:00Z"),
        ("2026-12-03T00:00:00Z", "2026-12-01T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "2026-12-01T00:00:00"),
    ],
)
def test_intent_details_activation_requires_announced_integrity_and_timezone(
    integrity_from: str | None, intent_details_from: str
) -> None:
    with pytest.raises(ServiceError, match="intent_details_activation_invalid"):
        _config(
            RoundDefaults(
                integrity_from=integrity_from,
                intent_details_from=intent_details_from,
            )
        )


def test_intent_details_activation_is_frozen_at_submission_open() -> None:
    class Store:
        def create_round(self, _round_id, _configuration):
            return {"status": "created"}

    runner = fixtures.keypair("intent-details-runner").ss58_address
    defaults = RoundDefaults(
        runner_hotkeys=(runner,),
        baseline_hotkey=fixtures.keypair("intent-details-baseline").ss58_address,
        scorer_image_digest=fixtures.SCORER_IMAGE_DIGEST,
        scorer_image_reference=fixtures.SCORER_IMAGE_REFERENCE,
        integrity_from="2026-01-01T00:00:00Z",
        intent_details_from="2026-12-01T00:00:00Z",
    )
    service = ArenaService(_config(defaults, Store()))
    service.runner_settings = lambda: ([runner], [])
    service._require_integrity_schema = lambda: None
    activation = datetime(2026, 12, 1, tzinfo=timezone.utc)

    old = service.create_round(
        activation + timedelta(days=1, seconds=-1),
        round_id="arena-2026-12-01-intentbefore",
    )
    new = service.create_round(
        activation + timedelta(days=1),
        round_id="arena-2026-12-02-intentafter",
    )

    assert "intent_details_policy" not in old
    assert "intent_details_policy" not in old["scorer_policy"]
    assert new["intent_details_policy"] == intent_details_policy.POLICY
    assert new["scorer_policy"]["intent_details_policy"] == (
        intent_details_policy.POLICY
    )
    assert "contact_policy" not in new
    assert "company_quality_policy" not in new


def test_round_contract_rejects_unpaired_intent_details_markers() -> None:
    policy = contracts.validate_scorer_policy(
        scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2"
        )
    )
    config = base_round_configuration()
    config["integrity_policy"] = "arena_integrity_v1"
    config["scorer_policy"] = policy
    config["intent_details_policy"] = intent_details_policy.POLICY
    with pytest.raises(contracts.ArenaContractError, match="matching scorer"):
        contracts.validate_round_configuration(config)
