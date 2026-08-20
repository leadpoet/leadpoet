from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json

import pytest

from gateway.research_lab.routing_predictleads_workflows import (
    ACTION_COMPANY,
    ACTION_CONNECTIONS,
    ACTION_DETECTIONS,
    ACTION_EXA,
    ACTION_JOB,
    ACTION_NEWS,
    ACTION_TECHNOLOGY,
    ROUTE_CONNECTIONS,
    ROUTE_ACTION_ORDER,
    ROUTE_NEWS,
    ROUTE_TECHNOLOGY,
    REVIEWED_ACTIONS,
    export_workflow_manifests,
    workflow_manifest,
    run_predictleads_connections,
    run_predictleads_news,
    run_predictleads_technology,
    validate_workflow_input,
)


DOMAIN = "acme.example"
MINIMUM = "2026-07-01"
MAXIMUM = "2026-08-18"


def company(company_id: str = "co-acme", domain: str = DOMAIN, name: str = "Acme"):
    return {
        "type": "company",
        "id": company_id,
        "attributes": {"domain": domain, "name": name},
    }


def response(rows, included=()):
    return {"result": {"data": {"data": rows, "included": list(included)}}}


def reserve_log():
    calls = []

    def reserve(route, max_calls, credits, timeout):
        calls.append((route, max_calls, credits, timeout))
        return True

    return calls, reserve


def connection_row(
    *,
    category="partner",
    company_id="co-acme",
    counterparty_id="co-beta",
    company1_id=None,
    company2_id=None,
    source=True,
    last_seen="2026-08-01",
):
    company1_id = company_id if company1_id is None else company1_id
    company2_id = counterparty_id if company2_id is None else company2_id
    attrs = {
        "category": category,
        "first_seen_at": "2026-07-15",
        "last_seen_at": last_seen,
    }
    if source:
        attrs["source_url"] = "https://news.example/partnership/acme-beta"
    return {
        "type": "connection",
        "id": "connection-1",
        "attributes": attrs,
        "relationships": {
            "company1": {"data": {"type": "company", "id": company1_id}},
            "company2": {"data": {"type": "company", "id": company2_id}},
        },
    }


def news_row(*, category="partners_with", source=True, relation_id="co-acme", event_id="event-1"):
    attrs = {
        "category": category,
        "planning": False,
        "found_at": "2026-08-01",
        "effective_date": "2026-08-01",
        "partner": "Beta",
    }
    if source:
        attrs["source_url"] = "https://news.example/acme-beta"
    return {
        "type": "news_event",
        "id": event_id,
        "attributes": attrs,
        "relationships": {
            "company1": {"data": {"type": "company", "id": relation_id}},
            "company2": {"data": {"type": "company", "id": "co-beta"}},
        },
    }


def detection_row(*, domain=DOMAIN, technology_id="tech-1", job_id="job-1", source_type="job_opening", source_count=2):
    return {
        "type": "technology_detection",
        "id": "detection-1",
        "attributes": {
            "last_seen_at": "2026-08-01",
            "source_type": source_type,
            "source_count": source_count,
        },
        "relationships": {
            "company": {"data": {"type": "company", "id": "co-acme"}},
            "technology": {"data": {"type": "technology", "id": technology_id}},
            "seen_on_job_openings": {"data": [{"type": "job_opening", "id": job_id}]},
        },
    }


def technology_row(technology_id="tech-1", name="Snowflake"):
    return {"type": "technology", "id": technology_id, "attributes": {"name": name}}


def job_row(*, job_id="job-1", company_id="co-acme", title="Data Engineer Snowflake"):
    return {
        "type": "job_opening",
        "id": job_id,
        "attributes": {
            "status": None,
            "title": title,
            "posted_at": "2026-08-02",
            "url": "https://jobs.example/acme/1",
        },
        "relationships": {"company": {"data": {"type": "company", "id": company_id}}},
    }


def test_registry_is_closed_and_routes_have_deterministic_action_order():
    assert set(REVIEWED_ACTIONS) == {
        ACTION_COMPANY,
        ACTION_CONNECTIONS,
        ACTION_DETECTIONS,
        ACTION_EXA,
        ACTION_JOB,
        ACTION_NEWS,
        ACTION_TECHNOLOGY,
    }
    assert set(REVIEWED_ACTIONS[ACTION_EXA].allowed_fields) == {
        "query", "numResults", "startPublishedDate", "endPublishedDate", "category",
    }


def test_workflow_manifests_are_deterministic_hash_bound_and_json_ready():
    first = export_workflow_manifests()
    second = export_workflow_manifests()
    assert first == second
    assert list(first) == sorted(first)
    assert set(first) == {ROUTE_CONNECTIONS, ROUTE_NEWS, ROUTE_TECHNOLOGY}

    for route, exported in first.items():
        manifest = workflow_manifest(route)
        assert exported == manifest.to_dict()
        assert exported["ordered_actions"]
        assert exported["max_calls"] == len(exported["ordered_actions"])
        assert exported["branch_optional_actions"] == sorted(exported["branch_optional_actions"])
        assert exported["manifest_hash"] == hashlib.sha256(
            json.dumps(
                manifest.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
        json.dumps(exported, sort_keys=True, separators=(",", ":"))
        assert tuple(item["action_id"] for item in exported["ordered_actions"]) == ROUTE_ACTION_ORDER[route]

    first[ROUTE_NEWS]["ordered_actions"].clear()
    assert export_workflow_manifests() == second


def test_unknown_workflow_manifest_is_rejected():
    with pytest.raises(ValueError, match="workflow route is not reviewed"):
        workflow_manifest("intent.source_add.unknown")


@pytest.mark.parametrize(
    "route",
    (ROUTE_CONNECTIONS, ROUTE_NEWS, ROUTE_TECHNOLOGY),
)
def test_workflow_input_validation_is_provider_independent_and_exact(route):
    values = {
        "company_domain": DOMAIN,
        "minimum_date": MINIMUM,
        "maximum_date": MAXIMUM,
    }
    if route == ROUTE_NEWS:
        values["intent_category"] = "partnership"
    if route == ROUTE_TECHNOLOGY:
        values["technology"] = "Snowflake"
    assert validate_workflow_input(route, values)["company_domain"] == DOMAIN
    with pytest.raises(ValueError, match="workflow input fields"):
        validate_workflow_input(route, {**values, "endpoint": "https://example.invalid"})


def test_connections_uses_company_connections_counterparty_order_and_reserves_full_route():
    reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append((action, dict(payload), timeout))
        if len(seen) == 1:
            return response([company()])
        if len(seen) == 2:
            assert seen[-1][1] == {
                "company_id_or_domain": "co-acme",
                "first_seen_at_from": MINIMUM,
                "first_seen_at_until": MAXIMUM,
                "categories": ["partner"],
                "page": 1,
                "limit": 25,
            }
            return response(
                [connection_row()],
                [company("co-beta", "beta.example", "Beta")],
            )
        return response([company("co-beta", "beta.example", "Beta")])

    result = run_predictleads_connections(
        company_domain=DOMAIN,
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.qualified
    assert result.calls == (ACTION_COMPANY, ACTION_CONNECTIONS, ACTION_COMPANY)
    assert reservations == [(ROUTE_CONNECTIONS, 3, 1_680_000, 30_000)]
    assert result.evidence["counterparty_domain"] == "beta.example"
    assert result.evidence["source_url"].startswith("https://")
    assert all(0 < item[2] <= 30_000 for item in seen)


@pytest.mark.parametrize(
    ("company1_id", "company2_id"),
    [("co-acme", "co-beta"), ("co-beta", "co-acme")],
)
def test_connections_accepts_requested_company_on_either_ordered_side(company1_id, company2_id):
    _reservations, reserve = reserve_log()

    def call(action, payload, timeout):
        if action == ACTION_COMPANY and payload["id_or_domain"] == DOMAIN:
            return response([company()])
        if action == ACTION_CONNECTIONS:
            return response([connection_row(company1_id=company1_id, company2_id=company2_id)])
        return response([company("co-beta", "beta.example", "Beta")])

    result = run_predictleads_connections(
        company_domain=DOMAIN,
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.qualified
    assert result.evidence["counterparty_company_id"] == "co-beta"


@pytest.mark.parametrize(
    ("company1_id", "company2_id"),
    [("co-one", "co-two"), ("co-acme", "co-acme")],
)
def test_connections_rejects_rows_where_requested_company_is_on_neither_or_both_sides(
    company1_id, company2_id
):
    _reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append(action)
        if action == ACTION_COMPANY:
            return response([company()])
        assert action == ACTION_CONNECTIONS
        return response([
            connection_row(company1_id=company1_id, company2_id=company2_id)
        ])

    result = run_predictleads_connections(
        company_domain=DOMAIN,
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.status == "miss"
    assert result.reason_code == "no_verified_current_partner"
    assert seen == [ACTION_COMPANY, ACTION_CONNECTIONS]


def test_connections_accepts_relationship_id_without_included_counterparty():
    _reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append(action)
        if action == ACTION_COMPANY and len(seen) == 1:
            return response([company()])
        if action == ACTION_CONNECTIONS:
            return response([connection_row()], [])
        assert action == ACTION_COMPANY
        return response([company("co-beta", "beta.example", "Beta")])

    result = run_predictleads_connections(
        company_domain=DOMAIN,
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.qualified
    assert result.calls == (ACTION_COMPANY, ACTION_CONNECTIONS, ACTION_COMPANY)
    assert result.evidence["counterparty_domain"] == "beta.example"


def test_connections_rejects_counterparty_detail_cross_id_without_included():
    _reservations, reserve = reserve_log()

    def call(action, payload, timeout):
        if action == ACTION_COMPANY and payload["id_or_domain"] == DOMAIN:
            return response([company()])
        if action == ACTION_CONNECTIONS:
            return response([connection_row()], [])
        return response([company("co-other", "beta.example", "Beta")])

    result = run_predictleads_connections(
        company_domain=DOMAIN,
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.status == "miss"
    assert result.reason_code == "counterparty_identity_unverified"
    assert result.evidence is None


@pytest.mark.parametrize(
    "change",
    [
        {"category": "vendor"},
        {"category": "partner", "source": False},
        {"category": "partner", "last_seen": "2023-12-31"},
    ],
)
def test_connections_rejects_weak_rows_without_calling_counterparty(change):
    _reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append(action)
        if action == ACTION_COMPANY:
            return response([company()]) if len(seen) == 1 else response([])
        return response([connection_row(**change)], [company("co-beta", "beta.example")])

    result = run_predictleads_connections(
        company_domain=DOMAIN,
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert not result.qualified
    assert result.reason_code == "no_verified_current_partner"
    assert seen == [ACTION_COMPANY, ACTION_CONNECTIONS]


def test_connections_fails_closed_on_cross_entity_company_and_malformed_response():
    _reservations, reserve = reserve_log()

    def cross_entity(action, payload, timeout):
        if action == ACTION_COMPANY:
            return response([company(domain="other.example")])
        raise AssertionError("connection call must not follow an unverified company")

    result = run_predictleads_connections(
        company_domain=DOMAIN, minimum_date=MINIMUM, maximum_date=MAXIMUM,
        reserve=reserve, call=cross_entity,
    )
    assert result.reason_code == "company_identity_unverified"

    result = run_predictleads_connections(
        company_domain=DOMAIN, minimum_date=MINIMUM, maximum_date=MAXIMUM,
        reserve=reserve, call=lambda *_args: {"result": "malformed"},
    )
    assert result.reason_code == "company_identity_unverified"


def test_news_with_included_source_stops_after_company_and_never_calls_exa():
    reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append((action, dict(payload)))
        if action == ACTION_NEWS:
            return response([news_row(source=True)], [company()])
        assert action == ACTION_COMPANY
        return response([company()])

    result = run_predictleads_news(
        company_domain=DOMAIN, intent_category="PARTNERSHIP",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=call,
    )
    assert result.qualified
    assert result.calls == (ACTION_NEWS, ACTION_COMPANY)
    assert ACTION_EXA not in result.calls
    assert reservations == [(ROUTE_NEWS, 3, 1_680_000, 30_000)]


def test_news_without_source_uses_conditional_exa_with_generated_bounded_query():
    reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append((action, dict(payload)))
        if action == ACTION_NEWS:
            return response([news_row(source=False)], [company()])
        if action == ACTION_COMPANY:
            return response([company()])
        assert action == ACTION_EXA
        assert payload["numResults"] == 5
        assert payload["startPublishedDate"] == MINIMUM
        assert payload["endPublishedDate"] == MAXIMUM
        assert "http" not in payload["query"]
        return {"results": [{
            "url": "https://press.example/acme-beta",
            "title": "Acme partnered with Beta",
            "text": "Acme partnered with Beta in August 2026.",
            "publishedDate": "2026-08-03",
        }]}

    result = run_predictleads_news(
        company_domain=DOMAIN, intent_category="PARTNERSHIP",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=call,
    )
    assert result.qualified
    assert result.calls == (ACTION_NEWS, ACTION_COMPANY, ACTION_EXA)
    assert result.evidence["source_resolution"] == "exa_fallback"
    assert reservations == [(ROUTE_NEWS, 3, 1_680_000, 30_000)]


@pytest.mark.parametrize(
    "row_kwargs",
    [
        {"category": "ends_partnership_with"},
        {"category": "partners_with", "relation_id": "co-other"},
    ],
)
def test_news_rejects_category_or_company_relation_mismatch(row_kwargs):
    _reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append(action)
        return response([news_row(**row_kwargs)], [company()])

    result = run_predictleads_news(
        company_domain=DOMAIN, intent_category="PARTNERSHIP",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=call,
    )
    assert result.reason_code in {"no_verified_news_event", "company_identity_unverified"}
    assert seen in ([ACTION_NEWS], [ACTION_NEWS, ACTION_COMPANY])


def test_news_rejects_unrelated_exa_result_and_does_not_emit_partial_evidence():
    _reservations, reserve = reserve_log()

    def call(action, payload, timeout):
        if action == ACTION_NEWS:
            return response([news_row(source=False)], [company()])
        if action == ACTION_COMPANY:
            return response([company()])
        return {"results": [{
            "url": "https://press.example/other",
            "title": "Other partnered with Gamma",
            "text": "Other partnered with Gamma.",
            "publishedDate": "2026-08-03",
        }]}

    result = run_predictleads_news(
        company_domain=DOMAIN, intent_category="PARTNERSHIP",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=call,
    )
    assert result.status == "miss"
    assert result.reason_code == "original_source_unresolved"
    assert result.evidence is None


def test_technology_uses_exact_four_call_order_and_job_backed_evidence():
    reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append((action, dict(payload)))
        if action == ACTION_DETECTIONS:
            return response(
                [detection_row()],
                [company(), technology_row()],
            )
        if action == ACTION_COMPANY:
            return response([company()])
        if action == ACTION_TECHNOLOGY:
            return response([technology_row()])
        assert action == ACTION_JOB
        return response([job_row()])

    result = run_predictleads_technology(
        company_domain=DOMAIN, technology="Snowflake",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=call,
    )
    assert result.qualified
    assert result.calls == (ACTION_DETECTIONS, ACTION_COMPANY, ACTION_TECHNOLOGY, ACTION_JOB)
    assert reservations == [(ROUTE_TECHNOLOGY, 4, 2_240_000, 30_000)]
    assert result.evidence["source_url"] == "https://jobs.example/acme/1"


def test_technology_accepts_relationship_ids_without_included_detection_resources():
    _reservations, reserve = reserve_log()
    seen = []

    def call(action, payload, timeout):
        seen.append(action)
        if action == ACTION_DETECTIONS:
            return response([detection_row()], [])
        if action == ACTION_COMPANY:
            return response([company()])
        if action == ACTION_TECHNOLOGY:
            return response([technology_row()])
        return response([job_row()])

    result = run_predictleads_technology(
        company_domain=DOMAIN,
        technology="Snowflake",
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.qualified
    assert seen == [ACTION_DETECTIONS, ACTION_COMPANY, ACTION_TECHNOLOGY, ACTION_JOB]


@pytest.mark.parametrize("crossed_action", [ACTION_COMPANY, ACTION_TECHNOLOGY])
def test_technology_rejects_cross_id_detail_resources_without_included(crossed_action):
    _reservations, reserve = reserve_log()

    def call(action, payload, timeout):
        if action == ACTION_DETECTIONS:
            return response([detection_row()], [])
        if action == ACTION_COMPANY:
            if crossed_action == ACTION_COMPANY:
                return response([company("co-other", "other.example", "Other")])
            return response([company()])
        if action == ACTION_TECHNOLOGY:
            if crossed_action == ACTION_TECHNOLOGY:
                return response([technology_row("tech-other", "HubSpot")])
            return response([technology_row()])
        return response([job_row()])

    result = run_predictleads_technology(
        company_domain=DOMAIN,
        technology="Snowflake",
        minimum_date=MINIMUM,
        maximum_date=MAXIMUM,
        reserve=reserve,
        call=call,
    )
    assert result.status == "miss"
    assert result.evidence is None
    assert result.reason_code in {
        "company_identity_unverified",
        "technology_identity_unverified",
    }


@pytest.mark.parametrize(
    "row_kwargs,reason",
    [
        ({"source_type": "dns_only"}, "no_verified_technology_detection"),
        ({"source_count": 0}, "no_verified_technology_detection"),
        ({"technology_id": "tech-other"}, "no_verified_technology_detection"),
    ],
)
def test_technology_rejects_non_job_or_cross_technology_detection(row_kwargs, reason):
    _reservations, reserve = reserve_log()

    def call(action, payload, timeout):
        return response(
            [detection_row(**row_kwargs)],
            [company(), technology_row(), technology_row("tech-other", "HubSpot")],
        )

    result = run_predictleads_technology(
        company_domain=DOMAIN, technology="Snowflake",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=call,
    )
    assert result.reason_code == reason
    assert result.evidence is None


def test_technology_rejects_job_cross_entity_or_missing_source():
    _reservations, reserve = reserve_log()

    def cross_job(action, payload, timeout):
        if action == ACTION_DETECTIONS:
            return response([detection_row()], [company(), technology_row()])
        if action == ACTION_COMPANY:
            return response([company()])
        if action == ACTION_TECHNOLOGY:
            return response([technology_row()])
        return response([job_row(company_id="co-other")])

    result = run_predictleads_technology(
        company_domain=DOMAIN, technology="Snowflake",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=cross_job,
    )
    assert result.reason_code == "job_company_relationship_mismatch"

    def no_source(action, payload, timeout):
        if action == ACTION_DETECTIONS:
            return response([detection_row()], [company(), technology_row()])
        if action == ACTION_COMPANY:
            return response([company()])
        if action == ACTION_TECHNOLOGY:
            return response([technology_row()])
        row = job_row()
        row["attributes"].pop("url")
        return response([row])

    result = run_predictleads_technology(
        company_domain=DOMAIN, technology="Snowflake",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=no_source,
    )
    assert result.reason_code == "job_source_missing"


def test_reservation_rejection_and_provider_exception_make_zero_calls_and_no_retry():
    reservation_calls = []
    provider_calls = []

    def reject(*args):
        reservation_calls.append(args)
        return False

    result = run_predictleads_news(
        company_domain=DOMAIN, intent_category="PARTNERSHIP",
        minimum_date=MINIMUM, maximum_date=MAXIMUM,
        reserve=reject, call=lambda *_args: provider_calls.append(1),
    )
    assert result.status == "blocked"
    assert provider_calls == []
    assert reservation_calls and reservation_calls[0][1:] == (3, 1_680_000, 30_000)

    calls = []
    reservations, reserve = reserve_log()

    def raises(action, payload, timeout):
        calls.append(action)
        raise TimeoutError("provider timeout")

    result = run_predictleads_news(
        company_domain=DOMAIN, intent_category="PARTNERSHIP",
        minimum_date=MINIMUM, maximum_date=MAXIMUM, reserve=reserve, call=raises,
    )
    assert result.reason_code == "provider_call_failed"
    assert calls == [ACTION_NEWS]
    assert len(reservations) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"company_domain": "not-a-domain"},
        {"company_domain": DOMAIN, "minimum_date": "2026-08-20"},
    ],
)
def test_invalid_request_fails_before_reservation(kwargs):
    reservations, reserve = reserve_log()
    args = {
        "company_domain": DOMAIN,
        "minimum_date": MINIMUM,
        "maximum_date": MAXIMUM,
        "reserve": reserve,
        "call": lambda *_args: {},
    }
    args.update(kwargs)
    result = run_predictleads_connections(**args)
    assert result.reason_code == "invalid_request"
    assert reservations == []
