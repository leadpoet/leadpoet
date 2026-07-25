"""Wrapper-contract conformance guard for the sourcing model.

The production sourcing flow is a harness built around frozen model symbols
(run_icp/adapter_metadata/qualify + the discovery/validation/client seams);
these tests lock the lab-side guard that protects that surface: the pure AST
verifier, the flag-gated candidate-build gate, and tolerance of the new
harness output shape in the lab's company normalizer.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from research_lab.sourcing_model_contract_check import (
    load_wrapper_contract,
    verify_source_tree_contract,
)


def _write(root: Path, relative: str, body: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def _conforming_tree(root: Path) -> None:
    _write(root, "requirements.txt", "httpx\n")
    _write(root, "research_lab_adapter.py", """
        def adapter_metadata():
            return {}

        def run_icp(icp, context=None):
            return []
    """)
    _write(root, "sourcing_model/__init__.py", "from .core import qualify\n")
    _write(root, "sourcing_model/clients.py", """
        import urllib.request

        def _exa_call():
            pass

        def agent_get():
            pass

        def agent_post():
            pass

        def exa_search():
            pass

        def sd_company():
            pass

        def sd_scrape():
            pass
    """)
    _write(root, "sourcing_model/core.py", """
        _GOAL_MAX_COMPANIES = 50
        _GOAL_MAX_ROUNDS = 8

        def _emitted_intent_evidence_url(c, icp, selected_url):
            pass

        def _fallback_sources(primary):
            pass

        def qualify(icp):
            return []
    """)
    _write(root, "sourcing_model/discovery.py", """
        def agent_results(icp, effort, timeout_s, avoid_companies):
            pass

        def apply_keep_gates(cand, source, today, max_age):
            pass

        def discover_goal_round(icp, source):
            pass

        def resolve_linkedin(name, anchor, agent_linkedin):
            pass
    """)
    _write(root, "sourcing_model/scoring.py", "SCORING = True\n")
    _write(root, "sourcing_model/validation.py", """
        def bonus_requirements(icp):
            pass

        def make_deps():
            pass

        async def validate_candidate():
            pass
    """)


def test_contract_loads_and_declares_frozen_surface() -> None:
    contract = load_wrapper_contract()
    assert contract["contract_id"] == "leadpoet-sourcing-wrapper-contract-v1"
    assert "research_lab_adapter.py" in contract["functions"]
    assert contract["functions"]["research_lab_adapter.py"]["run_icp"] == [
        "icp",
        "context",
    ]
    assert contract["required_imports"]["sourcing_model/clients.py"] == [
        "urllib.request"
    ]


def test_conforming_tree_has_no_violations(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    assert verify_source_tree_contract(tmp_path) == []


def test_missing_file_and_function_reported(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    (tmp_path / "sourcing_model/validation.py").unlink()
    violations = verify_source_tree_contract(tmp_path)
    assert "missing required file: sourcing_model/validation.py" in violations
    assert not any("run_icp" in v for v in violations)


def test_parameter_drift_reported(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "research_lab_adapter.py", """
        def adapter_metadata():
            return {}

        def run_icp(request, context=None):
            return []
    """)
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        v.startswith("parameter drift research_lab_adapter.py:run_icp") for v in violations
    )


def test_integer_floor_breach_reported(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "sourcing_model/core.py", """
        _GOAL_MAX_COMPANIES = 10
        _GOAL_MAX_ROUNDS = 8

        def _emitted_intent_evidence_url(c, icp, selected_url):
            pass

        def _fallback_sources(primary):
            pass

        def qualify(icp):
            return []
    """)
    violations = verify_source_tree_contract(tmp_path)
    assert any("integer floor breach" in v and "_GOAL_MAX_COMPANIES" in v for v in violations)


def test_unparseable_module_reported_not_raised(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "sourcing_model/core.py", "def qualify(icp:\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any(v.startswith("unparseable module sourcing_model/core.py") for v in violations)


def test_annotated_constant_assignment_conforms(tmp_path: Path) -> None:
    """``X: int = 50`` is behaviorally identical to ``X = 50`` and must not
    be reported as a missing constant (it would wrongly fail a legitimate
    candidate under enforce)."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(
        core.read_text().replace(
            "_GOAL_MAX_COMPANIES = ", "_GOAL_MAX_COMPANIES: int = ", 1
        )
    )
    assert verify_source_tree_contract(tmp_path) == []


def test_non_utf8_coding_header_is_violation_not_crash(tmp_path: Path) -> None:
    """A legal PEP 263 latin-1 module must parse like the interpreter parses
    it; a truly undecodable file must surface as a violation — never as an
    exception that would let the build gate fail open."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    legal = ("# -*- coding: latin-1 -*-\n# café\n" + core.read_text()).encode(
        "latin-1"
    )
    compile(legal, "core.py", "exec")  # sanity: importable Python
    core.write_bytes(legal)
    assert verify_source_tree_contract(tmp_path) == []

    # Null byte: rejected by the parser with ValueError, must be a violation.
    core.write_bytes(b"_GOAL_MAX_COMPANIES = 50\x00\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        v.startswith(("unreadable module sourcing_model/core.py",
                      "unparseable module sourcing_model/core.py"))
        for v in violations
    )


def test_dynamic_constant_rebinding_is_violation(tmp_path: Path) -> None:
    """A conforming literal followed by a top-level non-literal rebinding
    means the runtime value is no longer statically verifiable — the earlier
    literal must not satisfy the floor check."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(
        core.read_text()
        + "\n_GOAL_MAX_COMPANIES = min(2, _GOAL_MAX_ROUNDS)\n"
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any("_GOAL_MAX_COMPANIES" in v and "missing integer constant" in v
               for v in violations)

    # Augmented assignment likewise poisons the constant.
    _conforming_tree(tmp_path)
    core.write_text(core.read_text() + "\n_GOAL_MAX_ROUNDS -= 5\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any("_GOAL_MAX_ROUNDS" in v and "missing integer constant" in v
               for v in violations)

    # And a later conforming literal restores verifiability (last wins).
    _conforming_tree(tmp_path)
    core.write_text(core.read_text() + "\n_GOAL_MAX_COMPANIES = 60\n")
    assert verify_source_tree_contract(tmp_path) == []


def test_asyncness_drift_is_violation(tmp_path: Path) -> None:
    """Sync-vs-async is part of the callable surface: flipping it hands
    callers a coroutine (or breaks an await) despite identical parameters."""
    # Frozen-async function made sync → violation.
    _conforming_tree(tmp_path)
    val = tmp_path / "sourcing_model" / "validation.py"
    val.write_text(val.read_text().replace("async def validate_candidate", "def validate_candidate"))
    violations = verify_source_tree_contract(tmp_path)
    assert any("asyncness drift" in v and "validate_candidate" in v for v in violations)

    # Frozen-sync function made async → violation.
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(core.read_text().replace("def qualify(icp):", "async def qualify(icp):"))
    violations = verify_source_tree_contract(tmp_path)
    assert any("asyncness drift" in v and "qualify" in v for v in violations)


@pytest.mark.parametrize(
    "replacement",
    (
        "from urllib.request import urlopen",
        "import urllib.request as request",
    ),
)
def test_wrapper_reachable_import_must_remain_bound(
    tmp_path: Path,
    replacement: str,
) -> None:
    _conforming_tree(tmp_path)
    clients = tmp_path / "sourcing_model" / "clients.py"
    clients.write_text(
        clients.read_text().replace("import urllib.request", replacement)
    )

    violations = verify_source_tree_contract(tmp_path)

    assert (
        "missing bound import sourcing_model/clients.py:urllib.request"
        in violations
    )


def test_simple_alias_rebinding_conforms(tmp_path: Path) -> None:
    """``qualify = _impl`` is a runtime-valid rebinding — the alias carries
    the implementation's surface instead of reporting a missing function."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(
        core.read_text().replace("def qualify(icp):", "def _qualify_impl(icp):")
        + "\nqualify = _qualify_impl\n"
    )
    assert verify_source_tree_contract(tmp_path) == []

    # Aliasing to a wrong-signature implementation still drifts.
    _conforming_tree(tmp_path)
    core.write_text(
        core.read_text().replace("def qualify(icp):", "def _qualify_impl(wrong_name):")
        + "\nqualify = _qualify_impl\n"
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any("parameter drift" in v and "qualify" in v for v in violations)


def test_deleted_function_is_missing(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(core.read_text() + "\ndel qualify\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any("missing function" in v and "qualify" in v for v in violations)


# ---------------------------------------------------------------------------
# The flag-gated candidate-build gate
# ---------------------------------------------------------------------------


def test_build_gate_shadow_logs_and_proceeds(tmp_path: Path, monkeypatch, caplog) -> None:
    from gateway.research_lab.code_build import _sourcing_contract_gate

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "shadow")
    _conforming_tree(tmp_path)
    (tmp_path / "research_lab_adapter.py").unlink()
    with caplog.at_level("WARNING"):
        _sourcing_contract_gate(tmp_path)  # must NOT raise in shadow
    assert any(
        "sourcing_contract_gate_shadow_violation" in rec.message for rec in caplog.records
    )


def test_build_gate_enforce_fails_broken_tree(tmp_path: Path, monkeypatch) -> None:
    from gateway.research_lab.code_build import (
        CodeEditPrivateTestError,
        _sourcing_contract_gate,
    )

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "enforce")
    _conforming_tree(tmp_path)
    (tmp_path / "research_lab_adapter.py").unlink()
    with pytest.raises(CodeEditPrivateTestError, match="wrapper contract violation"):
        _sourcing_contract_gate(tmp_path)
    # a conforming tree passes enforce silently
    _conforming_tree(tmp_path)
    _sourcing_contract_gate(tmp_path)


def test_build_gate_disabled_and_fail_open(tmp_path: Path, monkeypatch) -> None:
    import gateway.research_lab.code_build as cb

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "disabled")
    cb._sourcing_contract_gate(tmp_path)  # empty tree, disabled -> no-op

    # Internal failure fails OPEN even in enforce (availability contract).
    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "enforce")
    import research_lab.sourcing_model_contract_check as check_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("contract file unreadable")

    monkeypatch.setattr(check_mod, "verify_source_tree_contract", _boom)
    cb._sourcing_contract_gate(tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# Harness output tolerance: the new flow's company shape through the lab
# normalizer (subindustry/hq_* mapping, extra fields ignored, intents mapped)
# ---------------------------------------------------------------------------


def test_lab_normalizer_tolerates_harness_company_shape() -> None:
    from research_lab.eval.evaluator import _normalize_company_output

    harness_company = {
        "company_name": "Acme Robotics",
        "domain": "acmerobotics.io",
        "company_website": "https://acmerobotics.io",
        "company_linkedin": "https://www.linkedin.com/company/acme-robotics",
        "industry": "Manufacturing",
        "subindustry": "Robotics",
        "hq_city": "Austin",
        "hq_state": "Texas",
        "hq_country": "United States",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "description": "Industrial robotics automation",
        "intent": {
            "source": "news",
            "url": "https://technews.io/acme-expansion",
            "date": "2026-07-01",
            "signal": "Acme announced a facility expansion",
            "why_valid": "Direct first-party announcement",
        },
        "required_attribute": {
            "text": "Manufactures its own hardware",
            "passed": True,
            "evidence_url": "https://acmerobotics.io/products",
            "evidence_quote": "We design and manufacture our robots in-house",
        },
        "additional_intents": [
            {
                "category": "HIRING",
                "signal": "Hiring robotics engineers",
                "source": "job_listing",
                "url": "https://jobs.acmerobotics.io/1",
                "date": "2026-07-10",
                "why_valid": "Live posting",
                "points": 40,
            }
        ],
        "score": 87,
        "discovery_audit": {"lane": "exa_agent", "round": 2},
    }
    normalized = _normalize_company_output(harness_company)
    assert normalized["company_name"] == "Acme Robotics"
    assert normalized["sub_industry"] == "Robotics"      # subindustry mapped
    assert normalized["country"] == "United States"       # hq_country mapped
    assert normalized["employee_count"] == "51-200"
    # intents arrive as a scoreable list with the primary mapped to index 0
    signals = normalized.get("intent_signals") or []
    assert signals, "primary intent must map into intent_signals"
