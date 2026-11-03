"""Read-only yield report for a supplied set of private Arena exports.

Counts are evidence coverage, not model quality or permission to train. Reuse
the dataset builders so this report cannot invent a weaker eligibility rule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from lab_arena import contracts, training_coverage, training_dataset, training_steps, training_thinking

SCHEMA_VERSION = "leadpoet.lab_arena.training_readiness.v1"
_DIMENSIONS = ("context", "source_identity", "exact_inputs", "input_output_label", "costs", "research_process")


def build(examples: Iterable[Mapping[str, Any]], *, production_rounds_only: bool = True) -> dict:
    """Report distinct, verified exports; keep incompatible scoring cohorts apart."""
    by_run, rejected, supplied = defaultdict(dict), Counter(), 0
    for example in examples:
        supplied += 1
        if not isinstance(example, dict) or example.get("schema_version") != training_dataset.EXAMPLE_SCHEMA_VERSION:
            rejected["example_schema_unsupported"] += 1
            continue
        run_id, digest = example.get("run_id"), example.get("example_hash")
        if not isinstance(run_id, str) or not run_id:
            rejected["execution_identity_missing"] += 1
            continue
        if digest != contracts.document_hash({key: value for key, value in example.items() if key != "example_hash"}):
            rejected["example_hash_mismatch"] += 1
            continue
        if digest in by_run[run_id]:
            rejected["duplicate_export"] += 1
        by_run[run_id][digest] = example
    valid = []
    for run_id in sorted(by_run):
        versions = by_run[run_id]
        if len(versions) != 1:
            # Do not silently select one of two differently judged/observed
            # versions of a run using filename or caller order.
            rejected["conflicting_export_versions"] += len(versions)
            continue
        valid.append(next(iter(versions.values())))

    dimensions, gaps, call_gaps, calls, recorder_counts = Counter(), Counter(), Counter(), Counter(), Counter()
    recorder_runs, recorder_missing, complete, selected_research_complete, cohorts = 0, 0, 0, 0, defaultdict(list)
    sources = []
    for example in valid:
        proof = example.get("completeness") or {}
        dimensions_for_run = {name: (proof.get("dimensions") or {}).get(name) is True for name in _DIMENSIONS}
        complete_for_run = (proof.get("complete") is True and all(dimensions_for_run.values())
                            and not proof.get("incomplete_reasons"))
        selected_gaps = training_thinking.selected_research_evidence(example)
        diagnostic_gaps = training_thinking.provider_visible_view(example)[1]
        selected_research_complete += selected_gaps is not None
        for name in _DIMENSIONS:
            dimensions[name] += dimensions_for_run[name]
        complete += complete_for_run
        gaps.update(set(proof.get("incomplete_reasons") or []))
        receipts = {"socket_written": 0, "read_reported": 0, "written_without_read_report": 0}
        # Inventory the execution, not its retries and judge calls again.
        for call in example.get("provider_calls") or []:
            if call.get("run_id") != example["run_id"]:
                continue
            calls["calls"] += 1
            calls["full_requests"] += isinstance(call.get("request"), dict)
            calls["retained_latest_responses"] += isinstance(call.get("response"), dict)
            calls["reported_received_responses"] += ((call.get("response_provenance") or {}).get("agent_receipt")
                                                     == "reported_exact_body_received")
            provenance = call.get("response_provenance") or {}
            if provenance.get("delivery_to_worker") == "socket_written":
                receipts["socket_written"] += 1
                received = provenance.get("agent_receipt") == "reported_exact_body_received"
                receipts["read_reported"] += received
                receipts["written_without_read_report"] += not received
            call_gaps.update(set(call.get("evidence_gaps") or []))
        artifact = (example.get("worker_observation_artifacts") or {}).get(example["run_id"]) or {}
        summaries = [row.get("data", {}).get("value") for row in (artifact.get("payload") or {}).get("agent_observations", [])
                     if row.get("kind") == "state_snapshot" and row.get("data", {}).get("name") == "automatic_capture_summary"]
        recorder_complete = None
        if len(summaries) == 1 and isinstance(summaries[0], dict):
            recorder_runs += 1
            if type(summaries[0].get("complete")) is bool:
                recorder_complete = summaries[0]["complete"]
            for key, value in (summaries[0].get("counts") or {}).items():
                if isinstance(key, str) and type(value) is int and value >= 0:
                    recorder_counts[key] += value
        else:
            recorder_missing += 1
        # A recorder can successfully flush an empty buffer while a native
        # child handles every call. Only the joined exporter proves coverage.
        sources.append({"run_id": example["run_id"], "example_hash": example["example_hash"],
                        "capture": {"authority": "joined_export_checks",
                                    "status": "complete" if complete_for_run else "incomplete" if proof else "unverified",
                                    "dimensions": dimensions_for_run,
                                    "incomplete_reasons": sorted(set(proof.get("incomplete_reasons") or [])),
                                    "selected_research_complete": selected_gaps is not None,
                                    "ancillary_judge_gaps": sorted(selected_gaps - diagnostic_gaps) if selected_gaps is not None else [],
                                    **({"scope": "provider_visible_request_results",
                                        "diagnostic_capture_gaps": sorted(diagnostic_gaps)}
                                       if training_coverage.uses_host_result_evidence(example.get("configuration") or {}) else {}),
                                    "recorder_buffer_complete": recorder_complete,
                                    "response_receipts": receipts}})
        config = example.get("configuration") or {}
        cohorts[(config.get("scorer_image_digest"), contracts.document_hash(config.get("scorer_policy") or {}))].append(example)

    qualification, qualification_totals = [], Counter()
    for (image, policy), records in sorted(cohorts.items(), key=lambda item: str(item[0])):
        # Keep the complete verified source graph even when labels are selected
        # from one scorer cohort. Steps and thinking use this same corpus.
        splits, manifest = training_dataset.build(
            records, training_dataset.Rules(production_rounds_only=production_rounds_only),
            split_sources=valid)
        counts = {name: len(items) for name, items in splits.items()}
        qualification_totals.update(counts)
        qualification.append({"scorer_image_digest": image, "scorer_policy_hash": policy,
                              "dataset_version": manifest["dataset_version"], "records": counts,
                              "selection": manifest["counts"],
                              "evaluation_checks": manifest.get("evaluation_checks", {})})
    steps, step_manifest = training_steps.build(valid, training_steps.Rules(production_rounds_only=production_rounds_only))
    thinking_splits, thinking_manifest = training_thinking.build(
        valid, training_steps.Rules(production_rounds_only=production_rounds_only))
    # This inventory counts ICPs with at least one valid thinking pair. It
    # does not establish complete ICP capture or champion eligibility.
    strict_runs = {ref.get("run_id") for split in ("train", "eval")
                   for record in thinking_splits[split] for ref in record.get("source_refs") or []}
    coverage_groups = defaultdict(list)
    for example in valid:
        key = (example.get("round_id"), example.get("submission_id"))
        coverage_groups[key].append(example)
    coverage = []
    for (round_id, submission_id), group in sorted(coverage_groups.items(), key=lambda item: str(item[0])):
        config = group[0].get("configuration") or {}
        try:
            total = contracts.benchmark_icp_count(config)
            percent = training_coverage.percent_from_configuration(config)
            required = training_coverage.required_count(total, percent)
            consistent = all(
                contracts.benchmark_icp_count(item.get("configuration") or {}) == total
                and training_coverage.percent_from_configuration(item.get("configuration") or {}) == percent
                for item in group)
        except (ValueError, contracts.ArenaContractError):
            total, percent, required, consistent = None, None, None, False
        positions = {item.get("icp_position") for item in group if type(item.get("icp_position")) is int}
        strict_positions = {item.get("icp_position") for item in group
                            if item.get("run_id") in strict_runs and type(item.get("icp_position")) is int
                            and total is not None and 0 <= item["icp_position"] < total}
        coverage.append({"round_id": round_id, "submission_id": submission_id,
                         "provided_icps": len(positions), "strict_icps": len(strict_positions),
                         "total_icps": total, "required_icps": required,
                         "minimum_coverage_percent": percent,
                         "supplied_inventory_complete": bool(consistent and total is not None
                                                             and positions == set(range(total))),
                         "strict_coverage_met_in_supplied_inventory": bool(
                             consistent and required is not None and len(strict_positions) >= required)})
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "supplied_exports_only_not_a_full_production_inventory",
        "production_rounds_only": production_rounds_only,
        "input_exports": supplied, "distinct_verified_executions": len(valid),
        "excluded_exports": dict(sorted(rejected.items())),
        "sources": sources,
        "capture": {"executions_passing_all_export_dimensions": complete,
                    "selected_research_complete": selected_research_complete,
                    "passing_dimensions": {name: dimensions[name] for name in _DIMENSIONS},
                    "execution_gap_counts": dict(sorted(gaps.items())),
                    "execution_calls": dict(sorted(calls.items())),
                    "call_gap_counts": dict(sorted(call_gaps.items())),
                    "automatic_capture": {"executions_with_summary": recorder_runs,
                                          "executions_without_verified_summary": recorder_missing,
                                          "counts": dict(sorted(recorder_counts.items())),
                                          "coverage": "supported_clients_only_not_all_local_state"}},
        "qualification": {"candidate_records": qualification_totals["train"] + qualification_totals["eval"],
                          "analysis_records": qualification_totals["analysis"], "cohorts": qualification,
                          "cohorts_must_not_be_combined_for_training": True},
        "agent_steps": {"candidate_records": len(steps["train"]) + len(steps["eval"]),
                        "analysis_records": len(steps["analysis"]),
                        "manifest": step_manifest},
        "full_thinking": {"candidate_records": len(thinking_splits["train"]) + len(thinking_splits["eval"]),
                          "manifest": thinking_manifest, "coverage": coverage,
                          "coverage_scope": "supplied_icps_with_valid_thinking_pairs_not_complete_icps_or_champion_eligibility"},
        "readiness": {"human_label_review": "not_proved_by_storage",
                      "agent_action_quality": "unreviewed",
                      "training_permission": dict(training_dataset.TRAINING_PERMISSION),
                      "training_ready": False,
                      "note": "Candidate counts are not reviewed good examples or a completed evaluation."},
    }
    report["report_hash"] = contracts.document_hash(report)
    return report
