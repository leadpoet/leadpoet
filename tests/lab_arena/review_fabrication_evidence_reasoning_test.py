"""Prompt reasoning controls; response validation and semantic authority stay unchanged."""

import json

import pytest

from lab_arena import code_review
from tests.lab_arena.test_lab_arena_code_review import _archive, _openrouter_response


SOURCE = {
    "harness.py": b"from output import company_row\ndef run_icp(icp, tools):\n    return [company_row(icp)]\n",
    "output.py": b'def company_row(icp):\n    return {"country": icp["country"]}\n',
    "review.py": b'def review(company, sources):\n    return model.check("Use company evidence", company, sources)\n',
}


def _finding():
    return {
        "category": "fabricated_evidence", "file": "output.py",
        "evidence": '"country": icp["country"]',
        "explanation": (
            "The emitted country field asserts the requested ICP country as a company fact. "
            "run_icp directly returns company_row, which copies icp country without evidence. "
            "Semantic review exists, but this entrypoint bypasses it; the finding concerns "
            "positive substitution and output rather than a missing machine check."
        ),
    }


@pytest.mark.parametrize("model", [
    code_review.DEFAULT_REVIEW_MODEL, code_review.FALLBACK_REVIEW_MODEL,
    code_review.PROSPECTIVE_REVIEW_MODEL, code_review.PROSPECTIVE_ALTERNATE_MODEL,
])
@pytest.mark.parametrize("research", [False, True])
def test_existing_explanation_requests_claim_origin_output_and_semantic_controls(model, research):
    prepared = code_review.prepare_request(_archive(SOURCE), model=model, research_control=research)
    prompt = prepared.parameters["messages"][0]["content"]
    assert "use the existing explanation to trace the factual" in prompt
    assert "claim's origin and its reachable final or recovery output" in prompt
    assert "defaults and saved state" in prompt
    assert "required semantic review are relevant controls" in prompt
    assert "A missing machine check alone is not a violation" in prompt
    assert "Instructions to use evidence are also not proof" in prompt
    assert "including any reachable bypass" in prompt
    assert "same clear-evidence threshold and all five misconduct categories" in prompt
    assert "source_trace" not in prompt
    assert prepared.parameters["max_tokens"] == code_review.DEFAULT_MAX_OUTPUT_TOKENS
    assert not {"temperature", "top_p", "top_k"} & prepared.parameters.keys()
    submitted = json.loads(prepared.parameters["messages"][1]["content"])
    assert {x["path"]: x["content"].encode() for x in submitted["submission_files"]} == SOURCE
    schema = prepared.parameters["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["findings"]["items"]["required"] == [
        "category", "file", "evidence", "explanation",
    ]


def test_sourced_direct_invention_remains_a_valid_rejection_with_existing_fields():
    prepared = code_review.prepare_request(_archive(SOURCE))
    result = code_review.parse_response(
        _openrouter_response(prepared, verdict="reject", findings=[_finding()]), prepared,
    )
    assert result.verdict == "reject"
    assert dict(result.findings[0]) == _finding()


def test_parser_retains_exact_quote_and_complete_coverage_checks():
    prepared = code_review.prepare_request(_archive(SOURCE))
    finding = _finding()
    finding["evidence"] = '"country": captured["country"]'
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(_openrouter_response(prepared, verdict="reject", findings=[finding]), prepared)
    assert failure.value.response_reason == "finding_evidence_mismatch"
    response = _openrouter_response(prepared, verdict="reject", findings=[_finding()])
    document = json.loads(response["choices"][0]["message"]["content"])
    document["reviewed_files"].pop()
    response["choices"][0]["message"]["content"] = json.dumps(document)
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == "coverage"


def test_semantic_control_presence_does_not_override_a_complete_verdict():
    # Source-only host validation cannot decide semantic sufficiency. Both a
    # complete pass and a complete reject retain their established authority.
    prepared = code_review.prepare_request(_archive(SOURCE))
    assert code_review.parse_response(_openrouter_response(prepared), prepared).passed
    finding = _finding()
    finding["explanation"] = "No dedicated field check exists."
    result = code_review.parse_response(_openrouter_response(prepared, verdict="reject", findings=[finding]), prepared)
    assert result.verdict == "reject"  # Prompt reasoning is not a new parser gate.


@pytest.mark.parametrize("category", [
    "hardcoded_prepared_answers", "malicious_behavior", "reviewer_manipulation",
])
def test_other_security_categories_keep_the_existing_finding_contract(category):
    prepared = code_review.prepare_request(_archive(SOURCE))
    finding = {**_finding(), "category": category}
    assert code_review.parse_response(_openrouter_response(prepared, verdict="reject", findings=[finding]), prepared).verdict == "reject"


def test_duplicate_comparisons_keep_the_existing_contract():
    contexts = ({"comparison_id": "trusted-reference"},)
    prepared = code_review.prepare_request(_archive(SOURCE), similarity_contexts=contexts)
    finding = {**_finding(), "category": "duplicate_submission", "comparison_id": "trusted-reference", "confidence": 0.99}
    response = _openrouter_response(prepared, verdict="reject", findings=[finding])
    document = json.loads(response["choices"][0]["message"]["content"])
    document["reviewed_comparisons"] = ["trusted-reference"]
    response["choices"][0]["message"]["content"] = json.dumps(document)
    assert code_review.parse_response(response, prepared).verdict == "reject"
    assert json.loads(prepared.parameters["messages"][1]["content"])["similarity_contexts"] == list(contexts)
