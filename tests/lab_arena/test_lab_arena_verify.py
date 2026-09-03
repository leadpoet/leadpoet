"""Public verifier tests (labarena.md sections 12.2-12.4 and 18.7)."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List

import pytest

from leadpoet_verifier.research_evaluation import compute_evaluation_aggregates
from research_lab.eval import evaluator

from lab_arena import contracts
from lab_arena import rewards, verify
from lab_arena.contracts import (
    ArenaContractError,
    BENCHMARK_COMMITMENT_SCHEMA_VERSION,
    GENERATION_JOURNAL_SCHEMA_VERSION,
    KING_OUTCOMES,
    OUTPUT_DOCUMENT_SCHEMA_VERSION,
    ROUND_CONFIGURATION_SCHEMA_VERSION,
    SCORE_BUNDLE_SCHEMA_VERSION,
    SCORER_POLICY_SCHEMA_VERSION,
    SCORING_PLAN_SCHEMA_VERSION,
    STAGE_SCHEDULE_FIELDS,
    benchmark_roots,
    document_hash,
    finalize_benchmark_commitment,
    finalize_round_configuration,
    finalize_scorer_policy,
    finalize_scoring_plan,
    participant_set_hash,
    work_item_id,
)
from lab_arena.signing import LocalSigner, sign_document, signing_key_document

BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ROUND_ID = "arena-2026-09-02"
PAYLOAD_KEYS = {
    "verification_trace",
    "quote",
    "snippet",
    "raw_response",
    "judge_prompt",
    "evidence_text",
    "page_content",
    "url",
    "web_evidence",
    "reason",
    "rejection_reason",
    "supporting_receipts",
    "identity_receipt",
    "provider_observations",
    "observed_decision",
    "submitted_decision",
    "stage1_status",
    "client_ready",
    "freshness_explanation",
}


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _hotkey(label: str) -> str:
    digest = hashlib.sha512(label.encode()).digest()
    return "5" + "".join(BASE58[byte % 58] for byte in digest[:47])


POLICY = finalize_scorer_policy(
    {
        "schema_version": SCORER_POLICY_SCHEMA_VERSION,
        "scoring_adapter_version": "qualification-v2",
        "fp_penalty_points": 10.0,
        "fp_unverified_primary_penalty_points": 10.0,
        "fp_penalty_icp_floor": 0.0,
        "company_cap_rule": "icp_max_companies",
        "max_scored_companies": 0,
        "judge_models": {"intent": "perplexity/sonar"},
        "cache_version": "v1",
        "provider_profile": "arena",
        "pre_slice_rule": "first_n_model_order",
        "employee_bucket_rule": "lab_relaxed_buckets",
        "env_bindings": {"RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES": "0"},
    }
)


# ---------------------------------------------------------------------------
# Breakdown fixtures (shaped like qualification/scoring/lead_scorer.py output)
# ---------------------------------------------------------------------------


def _receipt(decision: str, failure_class: str = "", gate: str = "company_fit") -> dict:
    dims = {"identity": decision, "employee_size": decision, "industry": decision, "geography": decision, "stage": "match"}
    receipt = {
        "gate": gate,
        "contract_id": "company-fit-decision:v1",
        "contract_version": "company-fit-decision:v1",
        "decision": decision,
        "reason": "company fit verified from independent identity and web evidence: 'quoted page text'",
        "company_fit_decision": decision,
        "company_fit_dimensions": dims,
        "company_fit_stage_required": False,
        "required_attribute_decision": "match",
        "dimension_evidence": {
            name: {
                "decision": value,
                "submitted_decision": value,
                "observed_decision": value,
                "web_evidence": {"url": "https://example.com/about", "quote": "We employ 120 people"},
            }
            for name, value in dims.items()
        },
        "identity_receipt": {"observed_company_name": "Acme", "raw_response": "{...}"},
        "provider_observations": {"observed_company_website": "https://acme.example"},
        "supporting_receipts": [{"gate": "web", "raw_response": "<html>", "page_content": "text"}],
    }
    if failure_class:
        receipt["failure_class"] = failure_class
    return receipt


def _signal(matched: int, after_decay: float, decision: str = "verified", **verdict_extra: Any) -> dict:
    verdict = {
        "decision": decision,
        "pipeline_decision": "verified" if decision == "verified" else "rejected",
        "stage1_status": "ok",
        "client_ready": decision == "verified",
        "rejection_reason": None if decision == "verified" else "claim_not_supported",
        "verification_trace": {
            "evidence_url": "https://example.com/jobs",
            "intent_verdict": {"signal_evaluations": [{"explanation": "quoted page text"}]},
        },
    }
    verdict.update(verdict_extra)
    return {
        "raw": after_decay,
        "after_decay": after_decay,
        "decay": 1.0,
        "confidence": 90,
        "date_status": "verified",
        "matched_icp_signal": matched,
        "evidence_type": "HIRING",
        "quote": "We are hiring backend engineers",
        "snippet": "excerpt",
        "judge_verdict": verdict,
    }


def _breakdown(score: float, failure_reason: Any = None, details: Any = "default", receipts: Any = "default") -> dict:
    row = {
        "icp_fit": 0.0,
        "decision_maker": 0.0,
        "intent_signal_raw": score,
        "time_decay_multiplier": 1.0,
        "intent_signal_final": score,
        "cost_penalty": 0.0,
        "time_penalty": 0.0,
        "final_score": score,
        "failure_reason": failure_reason,
        "judge_prompt": "system prompt text",
        "evidence_text": "page text",
        "page_content": "<html>",
    }
    row["intent_signals_detail"] = [_signal(0, score)] if details == "default" else details
    row["verifier_gate_receipts"] = [_receipt("match")] if receipts == "default" else receipts
    return row


def _junk(reason: str = "Company is on the ICP exclusion list: acme") -> dict:
    return _breakdown(0.0, failure_reason=reason, details=None, receipts=[_receipt("match")])


def _fp_fixtures() -> List[dict]:
    """One breakdown per branch of ``count_penalizable_false_positives``."""

    return [
        _breakdown(70.0),  # clean, primary verified
        _junk(),  # gate FP via failure_reason marker
        _breakdown(0.0, failure_reason="opaque wrapper text", details=None, receipts=[_receipt("mismatch")]),  # structured mismatch
        _breakdown(
            0.0,
            failure_reason="Company fit not proven",
            details=None,
            receipts=[_receipt("unavailable", failure_class="model_contract_incompatible")],
        ),  # model-contract incompatible -> gate FP
        _breakdown(30.0, details=[_signal(1, 30.0)]),  # only bonus verified -> unverified primary
        _breakdown(0.0, details=[_signal(-1, 0.0, decision="rejected_verifier_error", error_class="ReadTimeout")]),  # verifier error -> not counted
        _breakdown(0.0, details=[_signal(-1, 0.0, decision="rejected_three_stage", pipeline_decision="unavailable")]),  # pipeline unavailable -> retryable
        _breakdown(
            0.0,
            failure_reason="Company fit pre-check unavailable: provider outage",
            details=None,
            receipts=[_receipt("unavailable", gate="taxonomy_industry")],
        ),  # retryable infrastructure failure -> not counted
        _breakdown(0.0, failure_reason="LLM scoring error: timeout talking to provider", details=None),  # never penalized
        _breakdown(0.0, details=[_signal(-1, 0.0, decision="rejected_three_stage")]),  # content rejection -> unverified primary
        _breakdown(0.0, details=[]),  # no signals at all -> nothing to count
        _breakdown(0.0, details=["garbage"]),  # non-object detail keeps the list non-empty
    ]


# ---------------------------------------------------------------------------
# Round fixtures
# ---------------------------------------------------------------------------


def _icps() -> List[dict]:
    return [
        {
            "id": "icp-%02d" % position,
            "industry": "Software",
            "employee_count": "51-200",
            "employee_count_buckets": ["11-50", "51-200", "201-500"],
            "max_companies": 3 + position % 3,
            "intent_signals": ["Hiring backend engineers"],
            "country": "United States",
        }
        for position in range(50)
    ]


def _round_configuration(signing_key_hash: str, runner_hotkeys: List[str], *, all_stage_2: bool) -> dict:
    schedule = {spec.name: "2026-09-02T%02d:00:00Z" % index for index, spec in enumerate(STAGE_SCHEDULE_FIELDS)}
    return {
        "schema_version": ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": ROUND_ID,
        "mode": "shadow",
        "schedule": schedule,
        "generator": {
            "prompt_hash": _sha("prompt"),
            "exclusion_prompt_hash": _sha("exclusion"),
            "model": "generator-model",
            "settings": {"temperature": 0.7},
            "journal_schema_version": GENERATION_JOURNAL_SCHEMA_VERSION,
            "batch_sizes": [20, 20, 10],
            "max_generation_attempts": 12,
        },
        "tie_break_rule": "finalized_block_after_cutoff.v1",
        "stage_1_icp_count": 20,
        "stage_2_icp_count": 30,
        "finalist_count": 10,
        "max_challengers": 15,
        "runner_slot_ceiling": 8,
        "max_attempts_per_assignment": 2,
        "lease_ttl_seconds": 420,
        "companies_per_icp": 5,
        "release": {
            "repository_commit": "477222ac" * 5,
            "runsc_lock_hash": _sha("runsc"),
            "worker_release_hash": _sha("worker"),
            "shim_hash": _sha("shim"),
            "base_image_digest": "sha256:" + "ab" * 32,
        },
        "operation_table_hash": _sha("operations"),
        "openrouter_price_table_hash": _sha("openrouter-prices"),
        "openrouter_allowed_models": ["openai/gpt-4o-mini"],
        "miner_key_providers": list(contracts.MINER_KEY_PROVIDERS),
        "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
        "call_quota_hash": contracts.document_hash(contracts.call_quota_document()),
        "icp_wall_clock_seconds": 300,
        "scorer_policy_hash": POLICY["policy_hash"],
        "scoring_cap_microusd": 5_000_000,
        "runner_allowlist": runner_hotkeys,
        "floor_runner_hotkeys": runner_hotkeys[:1],
        "banned_hotkeys_snapshot_hash": _sha("banned"),
        "signing_public_key_hash": signing_key_hash,
        "artifact_rules": {
            "max_package_bytes": 1_000_000,
            "max_files": 100,
            "max_file_bytes": 100_000,
            "approved_dependency_set_hash": _sha("dependencies"),
        },
        "publication_terms_hash": _sha("terms"),
        "reward_constants": {
            "pool_percent": 25,
            "king_pool_share_percent_by_week": [100, 80, 60, 40, 20],
            "epochs_per_reward_week": 140,
            "eligibility_max_epochs": 45,
        },
        "all_participants_run_stage_2": all_stage_2,
    }


def _output_document(companies: List[dict]) -> dict:
    return {"schema_version": OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": companies}


def _companies(label: str, position: int, goal: int) -> List[dict]:
    companies = []
    for index in range(goal):
        bucket = "2-10" if (position % 4 == 1 and index == 1) else "51-200"
        companies.append(
            {
                "company_name": "%s Co %d-%d" % (label, position, index),
                "company_website": "https://%s-%d-%d.example" % (label.lower(), position, index),
                "employee_count": bucket,
            }
        )
    return companies


def _build_round(
    *,
    challenger_count: int = 15,
    king_mode: Any = "valid_lower",
    challenger_mode: str = "normal",
    all_stage_2: bool = False,
    with_reward_basis: bool = True,
) -> Dict[str, Any]:
    """A synthetic published round. ``king_mode``: valid_lower, valid_top, tie, preflight, or None."""

    signer = LocalSigner.generate()
    key_document = signing_key_document(signer.public_key_der)
    icps = _icps()
    icp_hashes = [document_hash(icp) for icp in icps]

    participants: List[dict] = []
    if king_mode is not None:
        participants.append(
            {
                "submission_id": "king",
                "miner_hotkey": _hotkey("king"),
                "image_digest": "sha256:" + "aa" * 32,
                "source_tree_hash": _sha("tree-king"),
                "is_king": True,
            }
        )
    for index in range(challenger_count):
        participants.append(
            {
                "submission_id": "sub-%02d" % index,
                "miner_hotkey": _hotkey("challenger-%d" % index),
                "image_digest": "sha256:" + ("%02x" % index) * 32,
                "source_tree_hash": _sha("tree-%d" % index),
                "is_king": False,
            }
        )
    by_id = {item["submission_id"]: item for item in participants}
    runner_hotkeys = [_hotkey("runner-%d" % index) for index in range(3)]

    config = sign_document(
        signer,
        finalize_round_configuration(_round_configuration(signer.public_key_hash, runner_hotkeys, all_stage_2=all_stage_2)),
        hash_field="configuration_hash",
    )
    roots = benchmark_roots(icp_hashes)
    commitment = sign_document(
        signer,
        finalize_benchmark_commitment(
            {
                "schema_version": BENCHMARK_COMMITMENT_SCHEMA_VERSION,
                "round_id": ROUND_ID,
                "configuration_hash": config["configuration_hash"],
                "participant_set_hash": participant_set_hash(participants),
                "tie_break_block_number": 1_234_567,
                "tie_break_block_hash": "0x" + hashlib.sha256(b"tie-break").hexdigest(),
                "journal_head_hash": _sha("journal"),
                "journal_length": 3,
                "evaluation_date": "2026-09-02",
                "benchmark_root": roots["benchmark_root"],
                "stage_1_root": roots["stage_1_root"],
                "stage_2_root": roots["stage_2_root"],
                "icp_leaf_hashes": roots["icp_leaf_hashes"],
                "generation_started_at": "2026-09-02T00:00:00Z",
                "generation_finished_at": "2026-09-02T00:20:00Z",
            }
        ),
        hash_field="commitment_hash",
    )
    salt = commitment["tie_break_block_hash"]
    top_challenger = "sub-%02d" % (challenger_count - 1)

    def challenger_profile(index: int, position: int, label: str) -> Any:
        if index == 3 and position in (2, 7, 25):
            return ("zero", "model_timeout")
        if index == 5 and position == 0:
            return ("zero", "invalid_output")
        icp = icps[position]
        companies = _companies(label, position, icp["max_companies"])
        scored, _ = verify.bucket_skip(icp, companies)
        base = 20 + 5 * index
        breakdowns = []
        for order, _company_index in enumerate(scored):
            if challenger_mode == "junk" or ((index + position) % 9 == 0 and order == 0):
                breakdowns.append(_junk())
            else:
                breakdowns.append(_breakdown(float(min(100, base + (position * 7 + order * 3) % 11))))
        return ("scored", companies, breakdowns)

    def profile(submission_id: str, position: int) -> Any:
        if submission_id == "king":
            if king_mode == "preflight":
                return ("zero", "preflight_failed")
            if king_mode == "tie":
                return profile(top_challenger, position)
            if king_mode == "valid_top":
                icp = icps[position]
                companies = _companies("King", position, icp["max_companies"])
                scored, _ = verify.bucket_skip(icp, companies)
                return ("scored", companies, [_breakdown(100.0) for _ in scored])
            return ("valid_lower_king", None)
        return challenger_profile(int(submission_id.split("-")[1]), position, submission_id)

    outputs: Dict[str, dict] = {}

    def rows_for(stage: int, ids: List[str]) -> List[dict]:
        rows = []
        for submission_id in ids:
            for position in verify._stage_positions(stage):
                spec = profile(submission_id, position)
                if spec[0] == "valid_lower_king":
                    icp = icps[position]
                    companies = _companies("King", position, icp["max_companies"])
                    scored, _ = verify.bucket_skip(icp, companies)
                    spec = ("scored", companies, [_breakdown(10.0) for _ in scored])
                if spec[0] == "zero":
                    rows.append(verify.zero_row(submission_id, position, icp_hashes[position], spec[1]))
                    continue
                companies, breakdowns = spec[1], spec[2]
                document = _output_document(companies)
                output_hash = document_hash(document)
                outputs[output_hash] = document
                rows.append(
                    verify.scored_row(
                        submission_id, position, icp_hashes[position], output_hash, icps[position], companies, breakdowns, POLICY
                    )
                )
        return rows

    def plan_for(stage: int, rows: List[dict]) -> dict:
        work: Dict[Any, dict] = {}
        zero_rows = []
        for row in rows:
            if row["cause"] == "accepted":
                key = (row["icp_hash"], row["output_hash"])
                item = work.setdefault(
                    key,
                    {
                        "work_item_id": work_item_id(*key),
                        "icp_position": row["icp_position"],
                        "icp_hash": key[0],
                        "output_hash": key[1],
                        "submission_ids": [],
                    },
                )
                item["submission_ids"].append(row["submission_id"])
            else:
                zero_rows.append({"submission_id": row["submission_id"], "icp_position": row["icp_position"], "cause": row["cause"]})
        plan = finalize_scoring_plan(
            {
                "schema_version": SCORING_PLAN_SCHEMA_VERSION,
                "round_id": ROUND_ID,
                "stage": stage,
                "configuration_hash": config["configuration_hash"],
                "commitment_hash": commitment["commitment_hash"],
                "scorer_policy_hash": POLICY["policy_hash"],
                "work_items": sorted(work.values(), key=lambda item: (item["icp_position"], item["work_item_id"])),
                "zero_rows": zero_rows,
            }
        )
        return sign_document(signer, plan, hash_field="plan_hash")

    def bundle_for(stage: int, rows: List[dict], scores: Dict[str, float], plan: dict, stage_1_hash: Any) -> dict:
        document = {
            "schema_version": SCORE_BUNDLE_SCHEMA_VERSION,
            "round_id": ROUND_ID,
            "stage": stage,
            "scorer_policy": POLICY,
            "scoring_plan_hash": plan["plan_hash"],
            "rows": rows,
            "submission_scores": scores,
        }
        if stage_1_hash is not None:
            document["stage_1_bundle_hash"] = stage_1_hash
        return sign_document(signer, verify.finalize_score_bundle(document), hash_field="bundle_hash")

    all_ids = [item["submission_id"] for item in participants]
    stage1_rows = rows_for(1, all_ids)
    stage1_scores = {}
    for submission_id in all_ids:
        values = [row["per_icp_score"] for row in stage1_rows if row["submission_id"] == submission_id]
        stage1_scores[submission_id] = verify.stage_score(values, 20)
    plan1 = plan_for(1, stage1_rows)
    bundle1 = bundle_for(1, stage1_rows, stage1_scores, plan1, None)

    ranking = verify.stage1_ranking(
        [
            {
                "submission_id": submission_id,
                "artifact_hash": by_id[submission_id]["source_tree_hash"],
                "stage1_score": stage1_scores[submission_id],
                "is_king": by_id[submission_id]["is_king"],
            }
            for submission_id in all_ids
        ],
        salt,
    )
    finalists = verify.select_finalists(ranking)
    stage2_ids = list(all_ids) if all_stage_2 else finalists + (["king"] if king_mode is not None else [])
    stage2_rows = rows_for(2, stage2_ids)
    final_scores = {}
    validity = {}
    for submission_id in stage2_ids:
        rows = {row["icp_position"]: row for row in stage1_rows + stage2_rows if row["submission_id"] == submission_id}
        final_scores[submission_id] = verify.stage_score([rows[position]["per_icp_score"] for position in range(50)], 50)
        validity[submission_id] = verify.result_is_valid(rows, tuple(range(50)))
    plan2 = plan_for(2, stage2_rows)
    bundle2 = bundle_for(2, stage2_rows, final_scores, plan2, bundle1["bundle_hash"])

    final_entries = [
        {
            "submission_id": submission_id,
            "hotkey": by_id[submission_id]["miner_hotkey"],
            "artifact_hash": by_id[submission_id]["source_tree_hash"],
            "final_score": final_scores[submission_id] if validity[submission_id] else None,
            "is_king": by_id[submission_id]["is_king"],
        }
        for submission_id in stage2_ids
    ]
    final_rank = verify.final_ranking(final_entries, salt)
    king_entry = next((entry for entry in final_entries if entry["is_king"]), None)
    decision = verify.king_decision([entry for entry in final_entries if not entry["is_king"]], king_entry, salt)

    public_bundle: Dict[str, Any] = {
        "round_configuration": config,
        "benchmark_commitment": commitment,
        "benchmark": icps,
        "participants": participants,
        "scorer_policy": POLICY,
        "stage_plans": {"1": plan1, "2": plan2},
        "score_bundles": {"1": bundle1, "2": bundle2},
        "outputs": outputs,
        "stage1_ranking": ranking,
        "finalists": finalists,
        "final_ranking": final_rank,
        "king_decision": decision,
    }
    if with_reward_basis:
        basis = rewards.reward_basis_document(
            round_id=ROUND_ID,
            configuration_hash=config["configuration_hash"],
            commitment_hash=commitment["commitment_hash"],
            result_bundle_hash=_sha("result-bundle"),
            published_at="2026-09-02T10:00:00Z",
            finalized_epoch=24999,
            king_outcome=decision["outcome"],
            king_hotkey=decision["king_hotkey"],
            previous_king_start_epoch=None if decision["outcome"] in ("crowned", "no_king") else 24900,
        )
        public_bundle["reward_basis"] = sign_document(signer, basis, hash_field="reward_basis_hash")

    return {
        "bundle": public_bundle,
        "key_document": key_document,
        "signer": signer,
        "salt": salt,
        "stage1_scores": stage1_scores,
        "final_scores": final_scores,
        "finalists": finalists,
        "decision": decision,
        "participants": participants,
    }


def _resign_stage(round_data: Dict[str, Any], stage: int) -> None:
    """Re-finalize and re-sign a mutated score bundle, cascading the stage 1 binding."""

    signer = round_data["signer"]
    bundles = round_data["bundle"]["score_bundles"]
    bundles[str(stage)] = sign_document(signer, verify.finalize_score_bundle(bundles[str(stage)]), hash_field="bundle_hash")
    if stage == 1:
        bundles["2"]["stage_1_bundle_hash"] = bundles["1"]["bundle_hash"]
        bundles["2"] = sign_document(signer, verify.finalize_score_bundle(bundles["2"]), hash_field="bundle_hash")


def _failed_details(report: Dict[str, Any]) -> str:
    return " | ".join(item["detail"] for item in report["checks"] if item["status"] == "failed")


@pytest.fixture(scope="module")
def full_round() -> Dict[str, Any]:
    return _build_round()


# ---------------------------------------------------------------------------
# Per-ICP arithmetic, slice, skip, stage denominators
# ---------------------------------------------------------------------------


def test_per_icp_score_equals_compute_evaluation_aggregates():
    icp = _icps()[0]
    assert icp["max_companies"] == 3
    icp = dict(icp, max_companies=5)
    breakdowns = [_breakdown(80.0), _breakdown(60.0), _breakdown(40.0), _junk(), _breakdown(30.0, details=[_signal(1, 30.0)])]
    gate, primary = evaluator.count_penalizable_false_positives(breakdowns, icp_has_intent_signals=True)
    assert (gate, primary) == (1, 1)
    expected = compute_evaluation_aggregates(
        [
            {
                "icp_ref": "x",
                "icp_hash": "x",
                "icp_company_goal": 5,
                "base_company_scores": [],
                "candidate_company_scores": [80.0, 60.0, 40.0, 0.0, 30.0],
                "candidate_fp_gate_count": gate,
                "candidate_fp_unverified_primary_count": primary,
            }
        ],
        leads_per_icp_normalizer=5,
        fp_penalty_points=10.0,
        fp_unverified_primary_penalty_points=10.0,
        fp_penalty_icp_floor=0.0,
    )["per_icp_results"][0]["candidate_per_icp_score"]
    result = verify.per_icp_score(icp, breakdowns, POLICY, icp_hash=_sha("icp"))
    assert result["per_icp_score"] == expected == 38.0  # 210/5 - 20/5
    assert (result["fp_gate_count"], result["fp_unverified_primary_count"]) == (1, 1)
    # The floor and the >100 clamp come from the published-bundle arithmetic.
    floored = verify.per_icp_score(icp, [_junk(), _junk(), _junk()], POLICY)
    assert floored["per_icp_score"] == 0.0 and floored["fp_gate_count"] == 3
    clamped = verify.per_icp_score(dict(icp, intent_signals=[]), [_breakdown(150.0)], POLICY)
    assert clamped["per_icp_score"] == 20.0
    # Redaction does not move the number.
    redacted = [verify.redact_breakdown(item) for item in breakdowns]
    assert verify.per_icp_score(icp, redacted, POLICY)["per_icp_score"] == expected
    with pytest.raises(ArenaContractError):
        verify.per_icp_score(icp, ["not-a-breakdown"], POLICY)


def test_first_n_slice_and_bucket_skip_are_recomputed():
    icp = {"industry": "Software", "employee_count": "51-200", "max_companies": 3, "intent_signals": ["x"]}
    companies = [
        {"employee_count": "51-200"},
        {"employee_count": "2-10"},
        {"employee_count": 120},
        {},
        {"employee_count": "51-200"},
        {"employee_count": "51-200"},
        {"employee_count": "51-200"},
    ]
    assert verify.slice_first_n(companies, 3) == companies[:3]
    assert verify.slice_first_n(companies, 50) == companies
    # Skips consume no slot; scoring stops once N companies are scored and
    # later companies are neither scored nor skipped.
    assert verify.bucket_skip(icp, companies) == ([0, 2, 4], [1, 3])
    assert verify.bucket_skip(icp, verify.slice_first_n(companies, 3)) == ([0, 2], [1])
    assert verify.bucket_skip(icp, companies, max_scored_companies=2) == ([0, 2], [1])
    assert verify.bucket_skip(icp, []) == ([], [])
    relaxed = dict(icp, employee_count_buckets=["2-10", "11-50", "51-200"])
    assert verify.bucket_skip(relaxed, companies) == ([0, 1, 2], [])
    assert verify.icp_company_goal({"max_companies": 500}) == 50
    assert verify.icp_company_goal({"max_companies": 0}) == 1
    with pytest.raises(ArenaContractError):
        verify.icp_company_goal({"industry": "Software"})
    with pytest.raises(ArenaContractError):
        verify.bucket_skip(icp, ["not-a-company"])
    with pytest.raises(ArenaContractError):
        verify.slice_first_n(companies, 0)


def test_stage_score_divides_by_exactly_20_and_50():
    assert verify.stage_score([0.1] * 20, 20) == 0.1
    assert verify.stage_score([50.0] * 19 + [0.0], 20) == 47.5
    assert verify.stage_score([1.0] * 50, 50) == 1.0
    assert verify.stage_score([1.0] * 20 + [0.0] * 30, 50) == 0.4
    assert verify.stage_score(list(range(50)), 50) == verify.stage_score(list(reversed(range(50))), 50)
    for scores, denominator in (([1.0] * 19, 20), ([1.0] * 21, 20), ([1.0] * 30, 30), ([1.0] * 49, 50)):
        with pytest.raises(ArenaContractError):
            verify.stage_score(scores, denominator)
    with pytest.raises(ArenaContractError):
        verify.stage_score([float("nan")] * 20, 20)


def test_zero_rows_only_for_terminal_causes():
    for cause in ("model_timeout", "invalid_output", "budget_exhausted", "model_error", "preflight_failed", "receipt_rejected"):
        row = verify.zero_row("sub-01", 3, _sha("icp"), cause)
        assert row["per_icp_score"] == 0.0 and row["breakdowns"] == [] and row["output_hash"] is None
    for cause in ("accepted", "refused", "", None):
        with pytest.raises(ArenaContractError):
            verify.zero_row("sub-01", 3, _sha("icp"), cause)
    with pytest.raises(ArenaContractError):
        verify.zero_row("sub-01", 50, _sha("icp"), "model_timeout")


# ---------------------------------------------------------------------------
# Ranking and decision
# ---------------------------------------------------------------------------


def test_stage1_ranking_finalists_and_tie_break():
    salt = "0x" + "ab" * 32
    entries = [
        {"submission_id": "sub-%02d" % index, "artifact_hash": _sha("artifact-%d" % index), "stage1_score": float(index % 5), "is_king": False}
        for index in range(15)
    ]
    entries.append({"submission_id": "king", "artifact_hash": _sha("king"), "stage1_score": 99.0, "is_king": True})
    ranking = verify.stage1_ranking(entries, salt)
    assert [row["submission_id"] for row in ranking if row["submission_id"] == "king"] == []
    assert [row["rank"] for row in ranking] == list(range(1, 16))
    scores = [row["stage1_score"] for row in ranking]
    assert scores == sorted(scores, reverse=True)
    for left, right in zip(ranking, ranking[1:]):
        if left["stage1_score"] == right["stage1_score"]:
            assert left["tie_break_hash"] < right["tie_break_hash"]
            assert left["tie_break_hash"] == verify.tie_break_hash(salt, _sha("artifact-%d" % int(left["submission_id"][4:])))
    finalists = verify.select_finalists(ranking)
    assert len(finalists) == 10 and finalists == [row["submission_id"] for row in ranking[:10]]
    assert verify.select_finalists(ranking[:6]) == [row["submission_id"] for row in ranking[:6]]
    # A different salt can reorder exact ties but never scores.
    other = verify.stage1_ranking(entries, "0x" + "cd" * 32)
    assert [row["stage1_score"] for row in other] == scores
    with pytest.raises(ArenaContractError):
        verify.stage1_ranking(entries + [entries[0]], salt)


def test_king_decision_all_outcomes_and_ties():
    salt = "0x" + "ab" * 32

    def entry(name: str, score: Any, king: bool = False) -> dict:
        return {"submission_id": name, "hotkey": _hotkey(name), "artifact_hash": _sha(name), "final_score": score, "is_king": king}

    king = entry("king", 50.0, True)
    # An exact tie keeps the king.
    tie = verify.king_decision([entry("a", 50.0), entry("b", 10.0)], king, salt)
    assert tie == {"outcome": "defended", "king_submission_id": "king", "king_hotkey": _hotkey("king"), "winner_submission_id": None}
    # Strictly higher crowns.
    crowned = verify.king_decision([entry("a", 50.000001), entry("b", 10.0)], king, salt)
    assert crowned == {"outcome": "crowned", "king_submission_id": "a", "king_hotkey": _hotkey("a"), "winner_submission_id": "a"}
    # Challenger ties break by the salted artifact hash, lower first.
    a_hash, b_hash = verify.tie_break_hash(salt, _sha("a")), verify.tie_break_hash(salt, _sha("b"))
    winner = verify.king_decision([entry("a", 60.0), entry("b", 60.0)], king, salt)["winner_submission_id"]
    assert winner == ("a" if a_hash < b_hash else "b")
    # A king without a valid result is replaced by any contender...
    assert verify.king_decision([entry("a", 0.5)], entry("king", None, True), salt)["outcome"] == "crowned"
    # ...and persists ineligible when there is none (zero or invalid challengers).
    retained = verify.king_decision([entry("a", 0.0), entry("b", None)], entry("king", None, True), salt)
    assert retained == {"outcome": "retained_ineligible", "king_submission_id": "king", "king_hotkey": _hotkey("king"), "winner_submission_id": None}
    # No contender against a valid king: defended, even at a valid zero.
    assert verify.king_decision([entry("a", 0.0)], entry("king", 0.0, True), salt)["outcome"] == "defended"
    # First round: highest valid challenger above zero, else no king.
    first = verify.king_decision([entry("a", 0.0), entry("b", 12.0), entry("c", 12.0)], None, salt)
    assert first["outcome"] == "crowned" and first["winner_submission_id"] in ("b", "c") and first["king_hotkey"] == _hotkey(first["winner_submission_id"])
    assert verify.king_decision([entry("a", 0.0), entry("b", None)], None, salt) == {
        "outcome": "no_king",
        "king_submission_id": None,
        "king_hotkey": "",
        "winner_submission_id": None,
    }
    assert verify.king_decision([], None, salt)["outcome"] == "no_king"
    for decision in (tie, crowned, retained, first):
        assert decision["outcome"] in KING_OUTCOMES
    with pytest.raises(ArenaContractError):
        verify.king_decision([king], None, salt)
    with pytest.raises(ArenaContractError):
        verify.king_decision([entry("king", 1.0)], king, salt)


def test_final_ranking_orders_valid_scores_then_king_then_hash():
    salt = "0x" + "ab" * 32
    entries = [
        {"submission_id": "a", "hotkey": _hotkey("a"), "artifact_hash": _sha("a"), "final_score": 40.0, "is_king": False},
        {"submission_id": "king", "hotkey": _hotkey("king"), "artifact_hash": _sha("king"), "final_score": 40.0, "is_king": True},
        {"submission_id": "b", "hotkey": _hotkey("b"), "artifact_hash": _sha("b"), "final_score": None, "is_king": False},
        {"submission_id": "c", "hotkey": _hotkey("c"), "artifact_hash": _sha("c"), "final_score": 55.0, "is_king": False},
    ]
    ranked = verify.final_ranking(entries, salt)
    assert [row["submission_id"] for row in ranked] == ["c", "king", "a", "b"]
    assert [row["rank"] for row in ranked] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _walk_keys(value: Any) -> set:
    keys: set = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _walk_keys(item)
    return keys


def test_redaction_keeps_fp_derivation_and_removes_payload():
    originals = _fp_fixtures()
    redacted = [verify.redact_breakdown(item) for item in originals]
    for flag in (True, False):
        assert evaluator.count_penalizable_false_positives(redacted, icp_has_intent_signals=flag) == evaluator.count_penalizable_false_positives(
            originals, icp_has_intent_signals=flag
        )
    # 3 gate FPs (marker, structured mismatch, contract-incompatible) and 3
    # unverified primaries (bonus-only, content rejection, non-object detail).
    assert evaluator.count_penalizable_false_positives(originals, icp_has_intent_signals=True) == (3, 3)
    assert evaluator.count_penalizable_false_positives(originals, icp_has_intent_signals=False) == (3, 0)
    checks = (
        evaluator.scorer_breakdown_has_retryable_infrastructure_failure,
        evaluator.scorer_breakdown_has_complete_current_company_fit_receipt,
        evaluator.scorer_breakdown_has_structured_company_fit_mismatch,
        evaluator.scorer_breakdown_has_model_contract_incompatibility,
    )
    for original, clean in zip(originals, redacted):
        for check in checks:
            assert check(clean) == check(original), check.__name__
        assert verify.redact_breakdown(clean) == clean  # idempotent
        assert verify.breakdown_is_redacted(clean)
        assert not _walk_keys(clean) & PAYLOAD_KEYS, _walk_keys(clean) & PAYLOAD_KEYS
    # Every branch was exercised by the fixtures.
    assert any(evaluator.scorer_breakdown_has_complete_current_company_fit_receipt(item) for item in redacted)
    assert any(evaluator.scorer_breakdown_has_structured_company_fit_mismatch(item) for item in redacted)
    assert any(evaluator.scorer_breakdown_has_model_contract_incompatibility(item) for item in redacted)
    assert sum(1 for item in redacted if evaluator.scorer_breakdown_has_retryable_infrastructure_failure(item)) == 4
    # Kept fields are exactly the FP inputs plus scalar score components.
    clean = redacted[0]
    assert set(clean) == set(verify.BREAKDOWN_FIELDS) | {"intent_signals_detail", "verifier_gate_receipts"}
    assert set(clean["intent_signals_detail"][0]) == set(verify.SIGNAL_DETAIL_FIELDS) | {"judge_verdict"}
    assert set(clean["intent_signals_detail"][0]["judge_verdict"]) == {"decision", "pipeline_decision"}
    receipt = clean["verifier_gate_receipts"][0]
    assert set(receipt) == set(verify.GATE_RECEIPT_FIELDS) - {"failure_class"} | {"company_fit_dimensions", "dimension_evidence"}
    assert receipt["dimension_evidence"]["industry"] == {"decision": "match"}
    # An unredacted breakdown is detectable and non-objects are neutralized.
    assert not verify.breakdown_is_redacted(originals[0])
    assert verify.redact_breakdown({"final_score": 1.0, "intent_signals_detail": ["garbage"], "verifier_gate_receipts": "x"}) == {
        "final_score": 1.0,
        "intent_signals_detail": [{}],
    }
    with pytest.raises(ArenaContractError):
        verify.redact_breakdown("not-a-breakdown")


# ---------------------------------------------------------------------------
# Score bundle schema
# ---------------------------------------------------------------------------


def test_score_bundle_schema_rejections(full_round):
    bundle = full_round["bundle"]["score_bundles"]["1"]
    verify.validate_score_bundle(bundle)
    accepted_index = next(index for index, row in enumerate(bundle["rows"]) if row["cause"] == "accepted")
    zero_index = next(index for index, row in enumerate(bundle["rows"]) if row["cause"] != "accepted")

    def mutated(change) -> dict:
        document = copy.deepcopy({k: v for k, v in bundle.items() if k not in ("bundle_hash", "signature")})
        change(document)
        return document

    def unredacted(document):
        document["rows"][accepted_index]["breakdowns"][0]["verification_trace"] = {"x": 1}

    def zero_with_breakdowns(document):
        document["rows"][zero_index]["breakdowns"] = [verify.redact_breakdown(_breakdown(1.0))]

    def zero_with_score(document):
        document["rows"][zero_index]["per_icp_score"] = 1.0

    def duplicate_row(document):
        document["rows"].append(copy.deepcopy(document["rows"][0]))

    def wrong_stage_position(document):
        document["rows"][accepted_index]["icp_position"] = 25

    def scores_mismatch(document):
        document["submission_scores"]["ghost"] = 1.0

    def bad_cause(document):
        document["rows"][zero_index]["cause"] = "refused"

    def stage_two_without_binding(document):
        document["stage"] = 2
        for row in document["rows"]:
            row["icp_position"] += 20

    def breakdown_count(document):
        document["rows"][accepted_index]["breakdowns"].append(document["rows"][accepted_index]["breakdowns"][0])

    def policy_hash_missing(document):
        document["scorer_policy"] = {k: v for k, v in POLICY.items() if k != "policy_hash"}

    for change in (
        unredacted,
        zero_with_breakdowns,
        zero_with_score,
        duplicate_row,
        wrong_stage_position,
        scores_mismatch,
        bad_cause,
        stage_two_without_binding,
        breakdown_count,
        policy_hash_missing,
    ):
        with pytest.raises(ArenaContractError):
            verify.validate_score_bundle(mutated(change))
    tampered = copy.deepcopy(bundle)
    tampered["rows"][accepted_index]["per_icp_score"] += 1.0
    with pytest.raises(ArenaContractError, match="bundle_hash"):
        verify.validate_score_bundle(tampered)


# ---------------------------------------------------------------------------
# Whole-round rebuild
# ---------------------------------------------------------------------------


def _assert_ok(report: Dict[str, Any]) -> None:
    assert report["ok"] is True, _failed_details(report)
    assert set(item["status"] for item in report["checks"]) <= {"verified", "failed", "not_checked"}
    assert report["statement"] == verify.VERIFIER_STATEMENT
    assert "aligned" not in report["statement"] and "divergent" not in report["statement"]


def test_full_round_fifteen_challengers_and_one_king(full_round):
    bundle = full_round["bundle"]
    report = verify.rebuild_round(bundle, full_round["key_document"])
    _assert_ok(report)
    verified = {item["check"] for item in report["checks"] if item["status"] == "verified"}
    assert {"stage_1.per_icp_and_stage_scores", "stage_1.finalists", "final.king_decision", "reward_basis.signature_and_decision"} <= verified
    participants = [item["submission_id"] for item in full_round["participants"]]
    assert len(participants) == 16
    stage1 = bundle["score_bundles"]["1"]["rows"]
    stage2 = bundle["score_bundles"]["2"]["rows"]
    for submission_id in participants:
        assert sorted(row["icp_position"] for row in stage1 if row["submission_id"] == submission_id) == list(range(20))
    finalists = full_round["finalists"]
    assert len(finalists) == 10 and "king" not in finalists
    stage1_scores = full_round["stage1_scores"]
    challengers = [sid for sid in participants if sid != "king"]
    assert min(stage1_scores[sid] for sid in finalists) >= max(stage1_scores[sid] for sid in challengers if sid not in finalists)
    for submission_id in participants:
        positions = sorted(row["icp_position"] for row in stage2 if row["submission_id"] == submission_id)
        assert positions == (list(range(20, 50)) if submission_id in finalists or submission_id == "king" else [])
    assert bundle["score_bundles"]["1"]["submission_scores"] == stage1_scores
    assert set(bundle["score_bundles"]["2"]["submission_scores"]) == set(finalists) | {"king"}
    # The lower-scoring king is replaced by the strictly higher top challenger.
    decision = full_round["decision"]
    assert decision["outcome"] == "crowned"
    assert full_round["final_scores"][decision["winner_submission_id"]] > full_round["final_scores"]["king"]
    assert any(row["cause"] == "model_timeout" for row in stage1) and any(row["skipped_company_indexes"] for row in stage1)


@pytest.mark.parametrize(
    "king_mode, challenger_mode, expected_outcome",
    [
        ("tie", "normal", "defended"),
        ("valid_top", "normal", "defended"),
        ("valid_lower", "normal", "crowned"),
        ("preflight", "normal", "crowned"),
        ("preflight", "junk", "retained_ineligible"),
        (None, "junk", "no_king"),
        ("valid_top", "junk", "defended"),
    ],
)
def test_round_outcomes_rebuild(king_mode, challenger_mode, expected_outcome):
    round_data = _build_round(challenger_count=12, king_mode=king_mode, challenger_mode=challenger_mode)
    decision = round_data["decision"]
    assert decision["outcome"] == expected_outcome
    if expected_outcome == "crowned":
        assert round_data["final_scores"][decision["winner_submission_id"]] > (round_data["final_scores"].get("king") or 0.0)
    if king_mode == "tie":
        assert round_data["final_scores"]["king"] == round_data["final_scores"]["sub-11"]
        assert decision["king_submission_id"] == "king"
    if expected_outcome == "no_king":
        assert decision["king_hotkey"] == "" and round_data["bundle"]["reward_basis"]["king_hotkey"] == ""
    _assert_ok(verify.rebuild_round(round_data["bundle"], round_data["key_document"]))


def test_fewer_than_ten_challengers_all_advance():
    round_data = _build_round(challenger_count=6)
    assert sorted(round_data["finalists"]) == ["sub-%02d" % index for index in range(6)]
    stage2 = round_data["bundle"]["score_bundles"]["2"]["rows"]
    assert {row["submission_id"] for row in stage2} == set(round_data["finalists"]) | {"king"}
    _assert_ok(verify.rebuild_round(round_data["bundle"], round_data["key_document"]))


def test_shadow_mode_everyone_runs_stage_two():
    round_data = _build_round(challenger_count=12, all_stage_2=True, with_reward_basis=False)
    stage2 = round_data["bundle"]["score_bundles"]["2"]["rows"]
    assert {row["submission_id"] for row in stage2} == {item["submission_id"] for item in round_data["participants"]}
    report = verify.rebuild_round(round_data["bundle"], round_data["key_document"])
    _assert_ok(report)
    assert any(item["check"] == "reward_basis" and item["status"] == "not_checked" for item in report["checks"])


def test_tampering_is_detected(full_round):
    key_document = full_round["key_document"]

    def tampered(mutate, resign_stage=None) -> Dict[str, Any]:
        # The signer holds an in-memory EC key that cannot be deep-copied.
        clone = {key: (value if key == "signer" else copy.deepcopy(value)) for key, value in full_round.items()}
        mutate(clone)
        if resign_stage is not None:
            _resign_stage(clone, resign_stage)
        return verify.rebuild_round(clone["bundle"], key_document)

    def accepted_row(clone, stage="1"):
        return next(row for row in clone["bundle"]["score_bundles"][stage]["rows"] if row["cause"] == "accepted" and row["breakdowns"])

    # 1. A breakdown edited without re-signing breaks the bundle hash.
    def edit_breakdown(clone):
        accepted_row(clone)["breakdowns"][0]["final_score"] += 1.0

    report = tampered(edit_breakdown)
    assert report["ok"] is False and "bundle_hash" in _failed_details(report)
    assert any(item["check"] == "rebuild" and item["status"] == "not_checked" for item in report["checks"])

    # 2. The same edit re-signed with the Arena key no longer recomputes.
    report = tampered(edit_breakdown, resign_stage=1)
    assert report["ok"] is False and "per-ICP score" in _failed_details(report)

    # 3. A published stage score edited and re-signed does not recompute.
    def edit_score(clone):
        scores = clone["bundle"]["score_bundles"]["1"]["submission_scores"]
        scores["sub-00"] = scores["sub-00"] + 0.5

    report = tampered(edit_score, resign_stage=1)
    assert report["ok"] is False and "stage 1 submission scores do not recompute" in _failed_details(report)

    # 4. A flipped signature byte fails.
    def edit_signature(clone):
        signature = clone["bundle"]["benchmark_commitment"]["signature"]["signature_b64"]
        signature = ("A" if signature[0] != "A" else "B") + signature[1:]
        clone["bundle"]["benchmark_commitment"]["signature"]["signature_b64"] = signature

    report = tampered(edit_signature)
    assert report["ok"] is False and "signature" in _failed_details(report).lower()

    # 5. Another key, or the wrong pinned hash, fails.
    other = signing_key_document(LocalSigner.generate().public_key_der)
    report = verify.rebuild_round(full_round["bundle"], other)
    assert report["ok"] is False and "signing key" in _failed_details(report)

    # 6. A reordered Stage 1 ranking fails.
    def swap_ranking(clone):
        ranking = clone["bundle"]["stage1_ranking"]
        ranking[0], ranking[1] = ranking[1], ranking[0]

    report = tampered(swap_ranking)
    assert report["ok"] is False and "stage1_ranking" in _failed_details(report)

    # 7. A changed king decision fails.
    def edit_decision(clone):
        clone["bundle"]["king_decision"]["outcome"] = "defended"

    report = tampered(edit_decision)
    assert report["ok"] is False and "king decision" in _failed_details(report)

    # 8. A reordered published output no longer hashes to its key.
    def reorder_output(clone):
        row = accepted_row(clone)
        companies = clone["bundle"]["outputs"][row["output_hash"]]["companies"]
        companies.append(companies.pop(0))

    report = tampered(reorder_output)
    assert report["ok"] is False and "does not hash" in _failed_details(report)

    # 9. A finalist missing from Stage 2 (re-signed) is incomplete.
    def drop_finalist_rows(clone):
        finalist = clone["finalists"][0]
        bundle2 = clone["bundle"]["score_bundles"]["2"]
        bundle2["rows"] = [row for row in bundle2["rows"] if row["submission_id"] != finalist]
        del bundle2["submission_scores"][finalist]

    report = tampered(drop_finalist_rows, resign_stage=2)
    assert report["ok"] is False and "incomplete" in _failed_details(report)

    # 10. A non-finalist scored in Stage 2 (re-signed) must not be there.
    def add_non_finalist(clone):
        bundle2 = clone["bundle"]["score_bundles"]["2"]
        outsider = next(sid for sid in clone["stage1_scores"] if sid not in clone["finalists"] and sid != "king")
        finalist = clone["finalists"][0]
        for row in [row for row in bundle2["rows"] if row["submission_id"] == finalist]:
            copied = copy.deepcopy(row)
            copied["submission_id"] = outsider
            bundle2["rows"].append(copied)
        bundle2["submission_scores"][outsider] = bundle2["submission_scores"][finalist]

    report = tampered(add_non_finalist, resign_stage=2)
    assert report["ok"] is False and "must not score" in _failed_details(report)

    # 11. A re-signed plan that dropped a work item no longer binds the bundle.
    def drop_work_item(clone):
        plan = clone["bundle"]["stage_plans"]["1"]
        plan = {k: v for k, v in plan.items() if k not in ("plan_hash", "signature")}
        plan["work_items"] = plan["work_items"][1:]
        clone["bundle"]["stage_plans"]["1"] = sign_document(clone["signer"], finalize_scoring_plan(plan), hash_field="plan_hash")

    report = tampered(drop_work_item)
    assert report["ok"] is False and "plan" in _failed_details(report)

    # 12. A re-signed reward basis naming another outcome contradicts the decision.
    def edit_basis(clone):
        basis = {k: v for k, v in clone["bundle"]["reward_basis"].items() if k not in ("reward_basis_hash", "signature")}
        basis["king_outcome"] = "defended"
        basis["king_start_epoch"] = 24900
        clone["bundle"]["reward_basis"] = sign_document(
            clone["signer"], rewards.finalize_reward_basis(basis), hash_field="reward_basis_hash"
        )

    report = tampered(edit_basis)
    assert report["ok"] is False and "reward basis" in _failed_details(report)

    # 13. A swapped benchmark ICP breaks the committed leaves.
    def swap_icps(clone):
        benchmark = clone["bundle"]["benchmark"]
        benchmark[0], benchmark[1] = benchmark[1], benchmark[0]

    report = tampered(swap_icps)
    assert report["ok"] is False and "committed leaves" in _failed_details(report)

    # 14. A zero row whose cause differs from the plan fails.
    def edit_zero_cause(clone):
        row = next(row for row in clone["bundle"]["score_bundles"]["1"]["rows"] if row["cause"] == "model_timeout")
        row["cause"] = "model_error"

    report = tampered(edit_zero_cause, resign_stage=1)
    assert report["ok"] is False and "scoring plan" in _failed_details(report)

    # The untouched round still verifies after all of the above.
    _assert_ok(verify.rebuild_round(full_round["bundle"], key_document))
