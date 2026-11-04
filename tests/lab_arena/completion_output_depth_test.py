"""Signed completion adds body/output levels around an already bounded score."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts as c, runner as rn, scoring
from lab_arena.service import ArenaService
from tests.lab_arena import completion_durability_test as durable
from tests.lab_arena.test_lab_arena_contracts import Keypair, signed, wallet_verifier


FIXTURE = Path(__file__).parents[1] / "fixtures/p7_vytex_native_completion_output.json"


def retained_output():
    fixture = json.loads(FIXTURE.read_text())
    output = fixture["native_output"]
    assert c.document_hash(output) == fixture["provenance"]["raw_output_sha256"]
    return output


def wrapped(output):
    return signed(Keypair.create_from_uri("//Alice"), scope=c.SCOPE_COMPLETE,
        body={"run_id": "completion-depth-control", "output": output})


def validate(envelope, scope=c.SCOPE_COMPLETE):
    return c.validate_signed_request(envelope, expected_scope=scope,
        now=envelope["timestamp"], verify_signature=wallet_verifier)


def test_exact_retained_vytex_output_survives_normal_signed_completion_wrappers():
    output = retained_output()
    assert scoring.validate_scoring_output_document(output) == output
    envelope = wrapped(output)
    assert validate(envelope) == envelope
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(verify_signature=wallet_verifier)
    service.now = lambda: datetime.fromtimestamp(envelope["timestamp"], timezone.utc)
    assert service.validate_request(envelope, scope=c.SCOPE_COMPLETE, round_id=None) == envelope
    assert c.document_hash(envelope["body"]["output"]) == c.document_hash(output)


def depth_output(extra_depth):
    # output0 -> breakdowns1 -> row2 -> extra3. A13-level chain reaches16.
    value = "bounded"
    for _ in range(extra_depth):
        value = {"node": value}
    return {"schema_version": scoring.SCORING_OUTPUT_SCHEMA_VERSION,
        "scored_run_id": "depth-boundary", "breakdowns": [{"final_score": 0., "extra": value}]}


def test_maximum_valid_publication_depth_has_exactly_two_envelope_levels():
    output = depth_output(13)
    assert c.PUBLICATION_LIMITS.max_depth == 16
    scoring.validate_scoring_output_document(output)
    assert validate(wrapped(output))["body"]["output"] == output


def test_too_deep_score_stays_rejected_by_publication_and_completion():
    output = depth_output(14)
    with pytest.raises(scoring.ScoringError, match="structural"):
        scoring.validate_scoring_output_document(output)
    with pytest.raises(c.ArenaContractError, match="nesting"):
        validate(wrapped(output))


def test_only_completion_depth_allowance_changes_other_bounds_stay_fixed():
    observed = asdict(c.COMPLETION_REQUEST_LIMITS)
    assert observed.pop("max_depth") == c.PUBLICATION_LIMITS.max_depth + 2
    assert observed == dict(max_list_items=20_000, max_object_keys=512,
        max_string_bytes=65_536, max_total_bytes=2*1_048_576+65_536,
        max_abs_int=9_007_199_254_740_992, allow_control_characters_in_values=False)
    assert c.REQUEST_LIMITS.max_depth == 10
    request = signed(Keypair.create_from_uri("//Alice"), scope=c.SCOPE_CLAIM,
        body={"output": depth_output(13)})
    with pytest.raises(c.ArenaContractError, match="nesting"):
        validate(request, c.SCOPE_CLAIM)


@pytest.mark.parametrize("damage", ["signature", "string", "nan"])
def test_completion_signature_string_and_finite_number_guards_remain(damage):
    output = {"value": "x"}
    if damage == "string": output["value"] = "x" * 65_537
    elif damage == "nan": output["value"] = float("nan")
    if damage == "nan":
        with pytest.raises(ValueError): wrapped(output)
        return
    envelope = wrapped(output)
    if damage == "signature": envelope["signature"] = "0x" + "0" * 128
    with pytest.raises(c.ArenaContractError): validate(envelope)


def test_runner_retains_and_replays_exact_deep_signed_completion(tmp_path):
    api = durable.Api()
    cfg = durable.config(tmp_path, api)
    cfg.work_dir.mkdir()
    worker = rn.Runner(cfg)
    original = durable.envelope(cfg)
    envelope = c.build_signed_request(scope=c.SCOPE_COMPLETE, round_id=original["round_id"],
        hotkey=original["hotkey"], timestamp=original["timestamp"],
        body={**original["body"], "output": retained_output()}, sign_message=durable.sign)
    before = c.request_bytes_hash(envelope)
    try:
        with worker._retain_completion(envelope) as path:
            descriptor = os.open(path, os.O_RDONLY)
            try: assert worker._read_pending_completion(descriptor, path) == envelope
            finally: os.close(descriptor)
        with worker._retain_completion(envelope) as repeated:
            assert repeated == path
        worker._replay_pending_completions()
        assert len(api.attempts) == 1
        assert c.request_bytes_hash(api.attempts[0]) == before
        assert api.attempts[0]["body"]["output"] == envelope["body"]["output"]
        assert not path.exists()
        assert cfg.sandbox_runtime.specs == []
    finally: worker.close()


def test_supported_conditional_brand_proof_stays_within_original_publication_limit(monkeypatch):
    from tests.test_industry_conditional_brand_sources import native_loop
    result, _, _, _ = native_loop(monkeypatch)
    assert result["claims"]["industry"]["_industry_review"]["industry_source_bindings"]["brands"]
    output = retained_output()
    for row in [output["breakdowns"][0], output["company_judgments"][0]["raw_judgment"]]:
        row["verifier_gate_receipts"][0]["supporting_receipts"][0]["claims"]["industry"] = deepcopy(result["claims"]["industry"])
    # Structural boundary only; this synthetic mixed-company document is never
    # submitted as a real company judgment or treated as semantic evidence.
    c.check_strict_document(output, c.PUBLICATION_LIMITS)
    validate(wrapped(output))
