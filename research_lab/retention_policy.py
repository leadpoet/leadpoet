"""Deliberate S3 retention/erasure policy for the trace prefixes (P20).

trajectoryimprovements.md P20: the trace prefixes accumulate KMS-encrypted
prompt/response text with no lifecycle policy in either direction — nothing
protects them from an ad-hoc cost-driven cleanup, and nothing implements the
v5 §8.5 deletion-with-hash-retention path. This module is the versioned,
in-repo policy plus the tooling that keeps bucket state matched to it:

* ``TRACE_PREFIX_RETENTION_POLICY`` — the dated policy per trace channel.
  Changing retention is a reviewed edit here, never a bucket-console change.
* ``render_s3_lifecycle_rules`` — the exact AWS LifecycleConfiguration rules
  the policy implies for a bucket/prefix.
* ``audit_bucket_lifecycle`` — compares a bucket's live rules against the
  policy: missing rules AND out-of-band rules touching trace prefixes are both
  findings (the deny/alarm half — an unexpected expiry rule would silently
  destroy the corpus raw layer).
* ``verification_state_for_bundle`` — the L0-facing read: a
  ``content_deleted`` bundle with its hash retained verifies as
  ``hash_attested`` instead of failing.

The erasure JOB (delete content → retain hash+anchor → flip state) lives in
``gateway/research_lab/trace_reconciler.py::erase_evidence_bundle_content``
next to the P6 reconciler, which must run after any lifecycle action so
pointer state and object state never diverge silently.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

RETENTION_POLICY_VERSION = "2026-07-02.1"

# Channel → policy. ``expire_days: None`` means retained indefinitely (the
# corpus raw layer is mission data — deletion only via the erasure path).
# ``noncurrent_expire_days`` bounds versioned-object noise; ``ia_days`` moves
# cold objects to STANDARD_IA for cost without destroying anything.
TRACE_PREFIX_RETENTION_POLICY: dict[str, dict[str, Any]] = {
    "trajectories": {
        "description": "engine raw LLM request/response traces (corpus raw layer)",
        "expire_days": None,
        "ia_days": 90,
        "noncurrent_expire_days": 30,
    },
    "telemetry": {
        "description": "legacy-channel OpenRouter traces via openrouter_telemetry",
        "expire_days": None,
        "ia_days": 90,
        "noncurrent_expire_days": 30,
    },
    "scorer-traces": {
        "description": "scorer judgment breakdown artifacts (dense reward layer)",
        "expire_days": None,
        "ia_days": 90,
        "noncurrent_expire_days": 30,
    },
    "incontainer-traces": {
        "description": "private-model in-container provider traces (axis stream)",
        "expire_days": None,
        "ia_days": 90,
        "noncurrent_expire_days": 30,
    },
    "diagnostics": {
        "description": "build-failure diagnostics (supervised negatives)",
        "expire_days": None,
        "ia_days": 60,
        "noncurrent_expire_days": 30,
    },
}


def render_s3_lifecycle_rules(key_prefix: str = "") -> list[dict[str, Any]]:
    """The exact LifecycleConfiguration rules this policy implies.

    ``key_prefix`` is the bucket-relative parent under which the trace
    channels live (e.g. ``research-lab``); rules are deterministic so a diff
    against live bucket rules is meaningful.
    """
    base = key_prefix.strip("/")
    rules: list[dict[str, Any]] = []
    for channel in sorted(TRACE_PREFIX_RETENTION_POLICY):
        policy = TRACE_PREFIX_RETENTION_POLICY[channel]
        prefix = f"{base}/{channel}/" if base else f"{channel}/"
        rule: dict[str, Any] = {
            "ID": f"research-lab-retention-{channel}-{RETENTION_POLICY_VERSION}",
            "Filter": {"Prefix": prefix},
            "Status": "Enabled",
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": int(policy["noncurrent_expire_days"])
            },
        }
        if policy.get("ia_days") is not None:
            rule["Transitions"] = [
                {"Days": int(policy["ia_days"]), "StorageClass": "STANDARD_IA"}
            ]
        if policy.get("expire_days") is not None:
            rule["Expiration"] = {"Days": int(policy["expire_days"])}
        rules.append(rule)
    return rules


def _rule_prefix(rule: Mapping[str, Any]) -> str:
    rule_filter = rule.get("Filter")
    if isinstance(rule_filter, Mapping):
        prefix = rule_filter.get("Prefix")
        if isinstance(prefix, str):
            return prefix
        and_block = rule_filter.get("And")
        if isinstance(and_block, Mapping) and isinstance(and_block.get("Prefix"), str):
            return and_block["Prefix"]
    legacy = rule.get("Prefix")
    return legacy if isinstance(legacy, str) else ""


def audit_bucket_lifecycle(
    live_rules: Sequence[Mapping[str, Any]],
    *,
    key_prefix: str = "",
) -> dict[str, Any]:
    """Compare live bucket lifecycle rules against the versioned policy.

    Findings:
    * ``missing_rules`` — policy rules absent from the bucket;
    * ``out_of_band_rules`` — live rules touching a trace prefix that this
      policy did not author (any such rule can silently destroy the corpus
      raw layer — alarm, never ignore);
    * ``unexpected_expirations`` — out-of-band rules that additionally expire
      current objects (the most destructive class).
    """
    expected = {rule["ID"]: rule for rule in render_s3_lifecycle_rules(key_prefix)}
    base = key_prefix.strip("/")
    channel_prefixes = tuple(
        (f"{base}/{channel}/" if base else f"{channel}/")
        for channel in TRACE_PREFIX_RETENTION_POLICY
    )
    live_by_id = {str(rule.get("ID") or ""): rule for rule in live_rules}
    missing = sorted(set(expected) - set(live_by_id))
    out_of_band: list[dict[str, Any]] = []
    unexpected_expirations: list[dict[str, Any]] = []
    for rule_id, rule in live_by_id.items():
        if rule_id in expected:
            continue
        prefix = _rule_prefix(rule)
        if not any(
            prefix.startswith(channel) or channel.startswith(prefix or "\x00")
            for channel in channel_prefixes
        ):
            continue
        finding = {"rule_id": rule_id, "prefix": prefix}
        out_of_band.append(finding)
        if rule.get("Expiration"):
            unexpected_expirations.append(finding)
    return {
        "schema_version": "1.0",
        "policy_version": RETENTION_POLICY_VERSION,
        "missing_rules": missing,
        "out_of_band_rules": out_of_band,
        "unexpected_expirations": unexpected_expirations,
        "compliant": not missing and not out_of_band,
    }


def verification_state_for_bundle(bundle_row: Mapping[str, Any]) -> str:
    """L0-facing verification state for an evidence bundle (v5 §8.5).

    A bundle whose content was erased under a valid deletion request but whose
    ``bundle_hash`` is retained verifies as ``hash_attested`` — a defined
    attested state, not an error. Anything else reports its stored state.
    """
    state = str(bundle_row.get("verification_state") or "active")
    if (
        state == "content_deleted"
        and str(bundle_row.get("bundle_hash") or "")
        and str(bundle_row.get("deletion_request_ref") or "")
    ):
        return "hash_attested"
    return state
