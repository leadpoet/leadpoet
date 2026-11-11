"""Source-bound finding controls; these do not predict a model's verdict."""

import copy
import json

import pytest

from lab_arena import code_review
from tests.lab_arena.test_lab_arena_code_review import _archive, _openrouter_response


SOURCE = {
    "harness.py": b"from output import company_row\ndef run_icp(icp, tools):\n    return [company_row(icp)]\n",
    "output.py": b'def company_row(icp):\n    return {"country": icp["country"]}\n',
    "review.py": b'def review(company, sources):\n    return model.check("Use company evidence", company, sources)\n',
    "prompts.txt": b"Use only company evidence. Omit unsupported claims.\n",
}


def _finding():
    return {
        "category": "fabricated_evidence", "file": "output.py",
        "evidence": '"country": icp["country"]',
        "explanation": (
            "The emitted country field asserts the requested ICP country as a company fact. "
            "run_icp directly returns company_row, which copies icp country without evidence. "
            "The evidence instruction and semantic review exist, but this entrypoint bypasses "
            "review entirely; the finding is the positive substitution and output path, not "
            "the absence of a dedicated machine check."
        ),
        "source_trace": [
            {"role": "claim_origin", "file": "output.py", "excerpt": '"country": icp["country"]'},
            {"role": "output_path", "file": "harness.py", "excerpt": "return [company_row(icp)]"},
            {"role": "evidence_control", "file": "review.py", "excerpt": 'return model.check("Use company evidence", company, sources)'},
            {"role": "evidence_control", "file": "prompts.txt", "excerpt": "Use only company evidence. Omit unsupported claims."},
        ],
    }


def _parse(finding):
    prepared = code_review.prepare_request(_archive(SOURCE))
    return code_review.parse_response(
        _openrouter_response(prepared, verdict="reject", findings=[finding]), prepared,
    )


def test_speculative_single_quote_rejection_is_incomplete_not_a_pass():
    finding = _finding()
    finding.pop("source_trace")
    finding["explanation"] = "There is no dedicated country machine check, so this is fabricated."
    with pytest.raises(code_review.CodeReviewError) as failure:
        _parse(finding)
    assert failure.value.code == "review_response_invalid"
    assert failure.value.response_reason == "finding_keys"


def test_positive_icp_invention_and_reachable_output_are_valid_rejection():
    finding = _finding()
    result = _parse(finding)
    assert result.verdict == "reject"
    assert result.findings[0]["source_trace"] == finding["source_trace"]
    # The parser copies the validated trace rather than retaining mutable input.
    finding["source_trace"][0]["excerpt"] = "changed"
    assert result.findings[0]["source_trace"][0]["excerpt"] != "changed"


@pytest.mark.parametrize("mutation, reason", [
    (lambda x: x.update(source_trace=[]), "finding_evidence"),
    (lambda x: x.update(source_trace=x["source_trace"][:1]), "finding_evidence"),
    (lambda x: x.update(source_trace=x["source_trace"] * 3), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(role="evidence_control"), "finding_evidence"),
    (lambda x: x["source_trace"][1].update(role="evidence_control"), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(role="unknown"), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(role=[]), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(file="unsubmitted.py"), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(excerpt=" \n"), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(excerpt="x" * 2001), "finding_evidence"),
    (lambda x: x["source_trace"][0].update(excerpt='"country": captured["country"]'), "finding_evidence_mismatch"),
    (lambda x: x["source_trace"][0].update(file="harness.py"), "finding_evidence_mismatch"),
    (lambda x: x["source_trace"][0].update(explanation="extra"), "finding_evidence"),
    (lambda x: x.update(explanation=" \n"), "finding_explanation"),
])
def test_trace_fails_closed_for_missing_roles_bounds_shape_and_exact_source(mutation, reason):
    finding = _finding()
    mutation(finding)
    with pytest.raises(code_review.CodeReviewError) as failure:
        _parse(finding)
    assert failure.value.response_reason == reason


def test_control_presence_does_not_override_semantic_review_or_force_a_pass():
    # The source-aware parser proves quotes, not whether the stated semantic
    # control is sufficient. It cannot turn an evidence-backed reject into pass.
    assert _parse(_finding()).verdict == "reject"
    prepared = code_review.prepare_request(_archive(SOURCE))
    assert code_review.parse_response(_openrouter_response(prepared), prepared).passed
    prompt = prepared.parameters["messages"][0]["content"]
    assert "A missing machine check alone is not a violation" in prompt
    assert "Evidence-directed model classification and required semantic review" in prompt
    assert "whenever relevant evidence-producing instructions" in prompt
    assert "do not replace the independent review of all five misconduct categories" in prompt


@pytest.mark.parametrize("category", [
    "hardcoded_prepared_answers", "malicious_behavior", "reviewer_manipulation",
])
def test_other_unsafe_categories_retain_their_existing_contract(category):
    finding = _finding()
    finding.pop("source_trace")
    finding["category"] = category
    assert _parse(finding).verdict == "reject"


def test_duplicate_contract_and_comparison_contexts_are_preserved():
    contexts = ({"comparison_id": "trusted-reference"},)
    prepared = code_review.prepare_request(_archive(SOURCE), similarity_contexts=contexts)
    finding = _finding()
    finding.pop("source_trace")
    finding.update(category="duplicate_submission", comparison_id="trusted-reference", confidence=0.99)
    response = _openrouter_response(prepared, verdict="reject", findings=[finding])
    document = json.loads(response["choices"][0]["message"]["content"])
    document["reviewed_comparisons"] = ["trusted-reference"]
    response["choices"][0]["message"]["content"] = json.dumps(document)
    assert code_review.parse_response(response, prepared).verdict == "reject"
    submitted = json.loads(prepared.parameters["messages"][1]["content"])
    assert submitted["similarity_contexts"] == list(contexts)
    assert {x["path"]: x["content"].encode() for x in submitted["submission_files"]} == SOURCE


def test_schema_enforces_category_specific_trace_and_parser_checks_source():
    from jsonschema import Draft202012Validator, ValidationError

    prepared = code_review.prepare_request(_archive(SOURCE))
    schema = prepared.parameters["response_format"]["json_schema"]["schema"]
    validator = Draft202012Validator(schema)
    Draft202012Validator.check_schema(schema)
    finding = _finding()
    document = {"verdict": "reject", "summary": "reviewed", "reviewed_files": list(prepared.reviewed_files),
                "reviewed_comparisons": [], "findings": [finding]}
    validator.validate(document)
    incomplete = copy.deepcopy(document)
    incomplete["findings"][0].pop("source_trace")
    with pytest.raises(ValidationError):
        validator.validate(incomplete)
    # A syntactically valid invented quote still cannot establish a rejection.
    finding["source_trace"][0]["excerpt"] = "invented evidence"
    validator.validate(document)
    with pytest.raises(code_review.CodeReviewError) as failure:
        _parse(finding)
    assert failure.value.response_reason == "finding_evidence_mismatch"


def test_direct_invention_without_any_control_can_still_reject():
    finding = _finding()
    finding["source_trace"] = finding["source_trace"][:2]
    prepared = code_review.prepare_request(_archive({k: v for k, v in SOURCE.items() if k in {"harness.py", "output.py"}}))
    result = code_review.parse_response(_openrouter_response(prepared, verdict="reject", findings=[finding]), prepared)
    assert result.verdict == "reject"
