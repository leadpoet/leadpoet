"""Constants for Research Lab observability payloads."""

LANGFUSE_TRACE_SCHEMA_VERSION = "research-lab-langfuse-trace:v1"
LANGFUSE_SCORE_SCHEMA_VERSION = "research-lab-langfuse-score:v1"

SAFE_TRACE_METADATA_KEYS = {
    "run_id",
    "ticket_id",
    "receipt_id",
    "miner_hotkey_hash",
    "island",
    "evaluation_epoch",
    "parent_artifact_hash",
    "candidate_artifact_hash",
    "private_model_manifest_hash",
    "candidate_patch_hash",
    "candidate_source_diff_hash",
    "icp_set_hash",
    "benchmark_split_ref",
    "scoring_version",
    "evaluator_version",
    "reference_evaluation_mode",
    "runtime_image_ref",
    "commit_sha",
    "score_bundle_hash",
    "anchored_hash",
    "cost_ledger_ref",
    "execution_trace_ref",
    "langfuse_trace_id",
    "failure_category",
    "failure_stage",
}
