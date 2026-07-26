"""Opt-in worker-fleet deferral must unblock sealing without weakening it.

A weight-submission outage was held hostage by the preflight refusing to seal
when the (already paused) worker fleets' egress proxies could not satisfy the
V2 HTTPS:443 transport. Deferral seals a role with zero proxy commitments so
its workers stay fail-closed, while non-deferred roles keep byte-identical
validation.
"""
import types

import pytest

from gateway.tee import prepare_gateway_envelopes_v2 as mod


def _plan(hosted_proxies, scoring_proxies):
    def fleet(values):
        return types.SimpleNamespace(
            enabled=True, proxy_values=list(values), worker_count=len(values)
        )
    return types.SimpleNamespace(
        hosted=fleet(hosted_proxies), scoring=fleet(scoring_proxies)
    )


_BAD = ["http://plain-proxy.example:8080"]          # plaintext non-443: invalid
_GOOD = ["https://user:pw@relay.example:443"]        # compliant


@pytest.fixture()
def _stub_plan(monkeypatch):
    def install(hosted, scoring):
        monkeypatch.setattr(
            mod,
            "build_research_lab_worker_autostart_plan",
            lambda _env: _plan(hosted, scoring),
        )
    return install


def test_default_still_fails_closed_on_plaintext_proxy(_stub_plan):
    _stub_plan(_BAD, _GOOD)
    with pytest.raises(mod.GatewayEnvelopePreparationV2Error):
        mod._validated_worker_proxy_configuration({})


def test_defer_all_seals_with_no_commitments(_stub_plan):
    _stub_plan(_BAD, _BAD)
    _plan_out, commitments = mod._validated_worker_proxy_configuration(
        {"GATEWAY_V2_DEFER_WORKER_FLEETS": "all"}
    )
    assert commitments == {
        "gateway_autoresearch": [],
        "gateway_scoring": [],
    }


def test_defer_one_role_keeps_other_validated(_stub_plan):
    # Deferring autoresearch must not relax scoring validation.
    _stub_plan(_BAD, _BAD)
    with pytest.raises(mod.GatewayEnvelopePreparationV2Error):
        mod._validated_worker_proxy_configuration(
            {"GATEWAY_V2_DEFER_WORKER_FLEETS": "gateway_autoresearch"}
        )
    # With scoring compliant, the same deferral seals: empty commitments for
    # the deferred role, real hashes for the validated one.
    _stub_plan(_BAD, _GOOD)
    _plan_out, commitments = mod._validated_worker_proxy_configuration(
        {"GATEWAY_V2_DEFER_WORKER_FLEETS": "gateway_autoresearch"}
    )
    assert commitments["gateway_autoresearch"] == []
    assert len(commitments["gateway_scoring"]) == 1


def test_unknown_role_rejected():
    with pytest.raises(mod.GatewayEnvelopePreparationV2Error):
        mod._deferred_worker_fleet_roles(
            {"GATEWAY_V2_DEFER_WORKER_FLEETS": "bogus_role"}
        )
