"""The submitted research source must emit the current frozen company shape."""

from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path

from lab_arena import output


SOURCE = Path(__file__).parent / "fixtures" / "full_thinking_sourcing_e2e" / "harness.py"


def _source(monkeypatch):
    monkeypatch.setitem(sys.modules, "lab_arena_checkpoint", types.ModuleType("lab_arena_checkpoint"))
    return runpy.run_path(str(SOURCE))


def _public_company():
    return {
        "company_name": "Example Systems",
        "company_website": "https://example.com",
        "company_linkedin": "",
        "industry": "Software",
        "employee_count": "1,001-5,000",
        "company_stage": "Public",
        "country": "United States",
        "state": "California",
        "intent_details": (
            "Example Systems sells workflow software to business teams. "
            "Its company news page describes a product launch dated September 1, 2026."
        ),
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "The company announced a workflow software release.",
            "date": "2026-09-01",
            "url": "https://example.com/news",
        }],
    }


def test_fixture_company_matches_current_company_only_intent_details_contract(monkeypatch):
    source = _source(monkeypatch)
    company = _public_company()
    model_schema = source["RESPONSE_SCHEMA"]["properties"]["companies"]["items"]
    signal_schema = model_schema["properties"]["intent_signals"]["items"]
    assert set(model_schema["properties"]) == set(model_schema["required"]) == set(company)
    assert set(signal_schema["properties"]) == set(signal_schema["required"]) == set(company["intent_signals"][0])
    assert source["_company"](company) == company
    document = json.dumps({"companies": [company]}).encode()
    accepted = output.output_document_from_bytes(
        document, expected_schema_version="leadpoet.lab_arena.output.v6",
        require_intent_dates=True,
    )
    assert len(accepted["companies"]) == 1


def test_fixture_rejects_prior_field_set_before_worker_submission(monkeypatch):
    source = _source(monkeypatch)
    company = _public_company()
    company["fit_summary"] = "Old company claim"
    assert source["_company"](company) is None
    company.pop("fit_summary")
    company["intent_signals"][0]["why_now"] = "Old signal claim"
    assert source["_company"](company) is None
