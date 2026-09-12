"""Contact activation is valid up front and applies only to newly opened rounds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import contact_policy, integrity
from lab_arena.service import ArenaService, RoundDefaults, ServiceConfig, ServiceError
from tests.lab_arena import test_lab_arena_service_round as fixtures


def _config(*, integrity_from: str, contacts_from: str) -> ServiceConfig:
    """Build only enough configuration to exercise activation validation."""

    return ServiceConfig(
        mode="shadow",
        store=object(),
        object_store=object(),
        signer=None,
        chain=object(),
        verify_signature=lambda *_args: True,
        daily_icp_source=lambda **_kwargs: {},
        banned_hotkeys_source=lambda: (),
        broker_factory=lambda *_args: None,
        defaults=RoundDefaults(
            integrity_from=integrity_from,
            contacts_from=contacts_from,
        ),
    )


def test_contact_activation_rejects_a_gap_before_integrity() -> None:
    with pytest.raises(ServiceError, match="contact_activation_invalid") as exc:
        _config(
            integrity_from="2026-09-13T00:00:01Z",
            contacts_from="2026-09-12T00:00:00Z",
        )

    assert exc.value.code == "contact_activation_invalid"


def test_contact_activation_requires_a_timezone() -> None:
    with pytest.raises(ServiceError, match="contact_activation_invalid") as exc:
        _config(
            integrity_from="2026-09-12T00:00:00Z",
            contacts_from="2026-09-12T00:00:00",
        )

    assert exc.value.code == "contact_activation_invalid"


@pytest.mark.parametrize("mode", ("shadow", "live"))
@pytest.mark.parametrize(
    "contacts_from",
    (
        "2026-12-01T00:00:00Z",
        "2026-12-01T00:00:00+00:00",
        "2026-11-30T19:00:00-05:00",
    ),
)
def test_contact_activation_changes_only_rounds_opened_at_or_after_boundary(
    mode: str, contacts_from: str
) -> None:
    runner = fixtures.keypair("activation-runner").ss58_address
    baseline = fixtures.keypair("activation-baseline").ss58_address

    class Store:
        def __init__(self) -> None:
            self.rounds = {}

        def create_round(self, round_id, configuration):
            self.rounds[round_id] = configuration
            return {"status": "created"}

    store = Store()
    service = ArenaService(
        ServiceConfig(
            mode=mode,
            store=store,
            object_store=object(),
            signer=None,
            chain=object(),
            verify_signature=lambda *_args: True,
            daily_icp_source=lambda **_kwargs: {},
            banned_hotkeys_source=lambda: (),
            broker_factory=lambda *_args: None,
            defaults=RoundDefaults(
                runner_hotkeys=(runner,),
                baseline_hotkey=baseline,
                scorer_image_digest=fixtures.SCORER_IMAGE_DIGEST,
                scorer_image_reference=fixtures.SCORER_IMAGE_REFERENCE,
                integrity_from="2026-01-01T00:00:00Z",
                contacts_from=contacts_from,
            ),
        )
    )
    service.runner_settings = lambda: ([runner], [])
    service._require_integrity_schema = lambda: None
    service._require_contact_schema = lambda: None
    activation = datetime(2026, 12, 1, tzinfo=timezone.utc)

    prior = service.create_round(
        activation + timedelta(days=1, seconds=-1),
        round_id="arena-2026-12-01-precontact",
    )
    first = service.create_round(
        activation + timedelta(days=1),
        round_id="arena-2026-12-02-contacts",
    )

    assert prior["integrity_policy"] == integrity.POLICY
    assert prior["scorer_policy"]["scoring_adapter_version"] == integrity.SCORING_ADAPTER
    assert "contact_policy" not in prior
    assert first["integrity_policy"] == integrity.POLICY
    assert first["contact_policy"] == contact_policy.POLICY
    assert first["scorer_policy"]["scoring_adapter_version"] == contact_policy.SCORING_ADAPTER
