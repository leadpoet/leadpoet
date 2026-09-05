"""Miner-facing SOURCE_ADD admission and process-exit coverage."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
MINER_HOTKEY = "5" + "A" * 47
SUBMISSION_ID = "source_add_submission:" + "1" * 16
ADAPTER_ID = "source_add_adapter:test"


def _load_miner_function(name: str, namespace: dict):
    source_path = ROOT / "neurons" / "miner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


def _source_add_flow(result, monkeypatch):
    required_values = iter(
        (
            "Example API",
            "https://api.example.test",
            "https://docs.example.test",
            "100 requests per minute",
        )
    )
    inputs = iter(("", "y"))
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    namespace = {
        "Optional": Optional,
        "GATEWAY_URL": "https://gateway.example.test",
        "_get_research_lab_status": lambda _url: {"source_add": {"intake_enabled": True}},
        "source_add_submission_ready": lambda _status: True,
        "SOURCE_ADD_SOURCE_KINDS": ("web",),
        "SOURCE_ADD_SOURCE_KIND_DESCRIPTIONS": {"web": "public web API"},
        "_research_lab_prompt_required_text": (
            lambda _prompt, max_length: next(required_values)
        ),
        "_research_lab_prompt_source_add_auth_type": lambda: "none",
        "_research_lab_prompt_source_add_endpoint_examples": lambda: [
            {"method": "GET", "path": "/v1/search", "description": "Search"}
        ],
        "_research_lab_prompt_optional_text": lambda _prompt, max_length: "",
        "_research_lab_prompt_source_add_third_party_refs": lambda: [],
        "build_source_add_submission_docs": lambda **_kwargs: (
            {
                "adapter_id": ADAPTER_ID,
                "source_kind": "web",
                "declared_base_domains": ["api.example.test"],
            },
            "source brief",
            "source-add-idempotency-key",
            {
                "api_base_url": "https://api.example.test",
                "documentation_url": "https://docs.example.test",
                "auth_type": "none",
                "endpoint_examples": [
                    {"method": "GET", "path": "/v1/search"}
                ],
            },
        ),
        "_research_lab_source_add_signed_payload": lambda _wallet, payload: payload,
        "_post_research_lab_json": lambda *_args, **_kwargs: result,
        "re": re,
    }
    _load_miner_function("_source_add_submission_error_message", namespace)
    return _load_miner_function("run_research_lab_source_add_flow", namespace)


def _wallet():
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=MINER_HOTKEY))


def test_source_add_cli_prints_specific_cap_and_returns_failure(
    monkeypatch,
    capsys,
):
    flow = _source_add_flow(
        {
            "status_code": 429,
            "error": {
                "detail": {
                    "code": "research_lab_rate_limited",
                    "message": (
                        "You already have 3 submissions in review. "
                        "Retry once one of them finishes."
                    ),
                }
            },
        },
        monkeypatch,
    )

    assert flow(_wallet(), SimpleNamespace(), 71) is False

    output = capsys.readouterr().out
    assert "SOURCE_ADD submission failed: HTTP 429" in output
    assert "You already have 3 submissions in review" in output
    assert "SOURCE_ADD submission received" not in output


def test_source_add_cli_keeps_duplicate_failure_generic(monkeypatch, capsys):
    flow = _source_add_flow(
        {"status_code": 409, "error": {"detail": "Submission failed"}},
        monkeypatch,
    )

    assert flow(_wallet(), SimpleNamespace(), 71) is False

    output = capsys.readouterr().out
    assert "Submission failed" in output
    assert "duplicate" not in output.lower()
    assert "SOURCE_ADD submission received" not in output


@pytest.mark.parametrize(
    "result",
    (
        [],
        {},
        {
            "submission_id": "invalid",
            "adapter_id": ADAPTER_ID,
            "stage": "provenance_queued",
        },
        {
            "submission_id": SUBMISSION_ID,
            "adapter_id": "source_add_adapter:other",
            "stage": "provenance_queued",
        },
        {
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "stage": "unexpected",
        },
    ),
)
def test_source_add_cli_rejects_malformed_success_receipt(
    result,
    monkeypatch,
    capsys,
):
    flow = _source_add_flow(result, monkeypatch)

    assert flow(_wallet(), SimpleNamespace(), 71) is False

    output = capsys.readouterr().out
    assert "SOURCE_ADD submission failed" in output
    assert "SOURCE_ADD submission received" not in output


def test_source_add_cli_accepts_only_complete_admission_receipt(
    monkeypatch,
    capsys,
):
    flow = _source_add_flow(
        {
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "stage": "provenance_queued",
            "precheck_reasons": [],
        },
        monkeypatch,
    )

    assert flow(_wallet(), SimpleNamespace(), 71) is True

    output = capsys.readouterr().out
    assert "SOURCE_ADD submission received" in output
    assert f"Submission ID: {SUBMISSION_ID}" in output
    assert "select Submit SOURCE_ADD, then check your submissions" in output


@pytest.mark.parametrize(
    ("outcome", "raises", "expected_code", "expected_text", "absent_text"),
    (
        (True, False, 0, "Done", "NOT SAVED"),
        (False, False, 1, "SOURCE_ADD submission NOT SAVED", "Done"),
        (None, False, 0, "", "NOT SAVED"),
        (None, True, 1, "SOURCE_ADD submission NOT SAVED", "Done"),
    ),
)
def test_source_add_mode_exit_matches_saved_result(
    outcome,
    raises,
    expected_code,
    expected_text,
    absent_text,
    capsys,
):
    class Logging:
        @staticmethod
        def error(_message):
            return None

    def run_flow(*_args):
        if raises:
            raise RuntimeError("network unavailable")
        return outcome

    namespace = {
        "bt": SimpleNamespace(
            Wallet=lambda config: _wallet(),
            logging=Logging(),
        ),
        "run_research_lab_source_add_flow": run_flow,
        "traceback": SimpleNamespace(print_exc=lambda: None),
    }
    runner = _load_miner_function(
        "_run_research_lab_source_add_mode",
        namespace,
    )

    code = runner(SimpleNamespace(netuid=71))

    assert code == expected_code
    output = capsys.readouterr().out
    if expected_text:
        assert expected_text in output
    assert absent_text not in output


def test_source_add_main_propagates_mode_exit_code():
    source = (ROOT / "neurons" / "miner.py").read_text(encoding="utf-8")

    assert (
        "raise SystemExit(_run_research_lab_source_add_mode(config))" in source
    )
