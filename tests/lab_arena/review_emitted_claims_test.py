"""Offline prompt/parser controls; these do not predict a model's verdict."""

import json
from pathlib import Path

import pytest

from lab_arena import code_review
from tests.lab_arena.test_lab_arena_code_review import _archive, _openrouter_response


FIXTURES = Path(__file__).parents[1] / "fixtures" / "admission_emitted_claims"


@pytest.mark.parametrize("model", [
    code_review.DEFAULT_REVIEW_MODEL, code_review.FALLBACK_REVIEW_MODEL,
])
@pytest.mark.parametrize("research_control", [False, True])
def test_every_review_route_requires_emitted_claim_not_internal_flag(model, research_control):
    prepared = code_review.prepare_request(
        _archive({"harness.py": b"def run_icp(icp, tools): return []\n"}),
        model=model, research_control=research_control,
    )
    prompt = prepared.parameters["messages"][0]["content"]
    assert "internal candidate/pass flag" in prompt
    assert "matched_icp_signal" in prompt
    assert "precise unsupported factual assertion and its reachable emitted field" in prompt
    assert "A hedge does not excuse an unsupported factual premise" in prompt
    assert "not infer a violation merely from a missing citation" in prompt
    assert "substituting the requested country, stage, or" in prompt
    assert "invent or force facts" in prompt
    if research_control:
        assert code_review._RESEARCH_PROMPT in prompt
    else:
        assert "research_control" not in prompt


@pytest.mark.parametrize("case", [
    "truthful", "cross_field_evidence", "fabricated", "ambiguous_offering",
])
def test_synthetic_archive_contains_complete_source_with_same_response_schema(case):
    members = {item.name: item.read_bytes() for item in (FIXTURES / case).glob("*.py")}
    prepared = code_review.prepare_request(_archive(members))
    submitted = json.loads(prepared.parameters["messages"][1]["content"])
    assert {item["path"]: item["content"].encode() for item in submitted["submission_files"]} == members
    schema = prepared.parameters["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {
        "verdict", "summary", "reviewed_files", "reviewed_comparisons", "findings",
    }
    assert prepared.parameters["max_tokens"] == code_review.DEFAULT_MAX_OUTPUT_TOKENS


def test_fabrication_finding_still_requires_exact_source_and_complete_coverage():
    members = {item.name: item.read_bytes() for item in (FIXTURES / "fabricated").glob("*.py")}
    prepared = code_review.prepare_request(_archive(members))
    finding = {
        "category": "fabricated_evidence", "file": "output.py",
        "evidence": '"country": icp["country"]',
        "explanation": "run_icp calls company_row without verification; its emitted country field copies the requested country without company evidence.",
        "source_trace": [
            {"role": "claim_origin", "file": "output.py", "excerpt": '"country": icp["country"]'},
            {"role": "output_path", "file": "harness.py", "excerpt": "company_row("},
        ],
    }
    response = _openrouter_response(prepared, verdict="reject", findings=[finding])
    assert not code_review.parse_response(response, prepared).passed
    document = json.loads(response["choices"][0]["message"]["content"])
    document["findings"][0]["evidence"] = '"country": independently_verified_country'
    response["choices"][0]["message"]["content"] = json.dumps(document)
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == "finding_evidence_mismatch"
    document["findings"][0] = finding
    document["reviewed_files"].pop()
    response["choices"][0]["message"]["content"] = json.dumps(document)
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == "coverage"
