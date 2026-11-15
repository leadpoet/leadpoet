"""Transaction-local, explicit-owner follow-up to an audited published correction.

This is an operator library, never called by the driver. It does not change a
public function or a normal retry limit. The supplied approval binds the exact
round, failed parents, corrected runtime, deadlines and checkpoint hashes.
Existing recovery code still checks paid-call finality, input identity and costs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROUND_SIGNATURE = "public.lab_arena_correct_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)"
GRANT_SIGNATURE = "public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)"
GRANT_HASH = "2c064da4290e3095821fc3cbbc3f79a4c3cadc0da1b2fe496d5f0907bd09b743"
CALL = "SELECT pg_temp.approved_round_followup(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("reviewed recovery source anchor differs")
    return source.replace(old, new, 1)


def _between(source: str, start: str, end: str, replacement: str = "") -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("reviewed recovery source boundary differs")
    left, right = source.index(start), source.index(end)
    if right <= left:
        raise ValueError("reviewed recovery source order differs")
    return source[:left] + replacement + source[right:]


def install(cursor, approval: dict) -> None:
    """Install only session-local helpers inside the caller's locked transaction.

    The caller must verify release identity and explicit user authority, hold the
    shared deployment lock, validate with rollback, then commit the same inputs.
    Public recovery functions, triggers, ACLs and historical rows stay unchanged.
    """
    required = {"round", "sha", "runtime", "final", "publication", "requests",
                "request", "actor", "approval"}
    if set(approval) != required or not approval["requests"] or not approval["approval"]:
        raise ValueError("exact approval is required")
    cursor.execute("SELECT current_setting('transaction_read_only'), current_setting('session_replication_role')")
    if cursor.fetchone() != ("off", "origin"):
        raise ValueError("normal read-write transaction required")
    cursor.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (ROUND_SIGNATURE,))
    round_source = cursor.fetchone()[0]
    committed = (Path(__file__).with_name("695-lab-arena-published-champion-correction.sql")).read_text()
    expected = re.search(r"AS \$correct695\$(.*?)\$correct695\$;", committed, re.S).group(1)
    if round_source != expected:
        raise ValueError("published correction source differs")
    cursor.execute("SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure", (GRANT_SIGNATURE,))
    grant_source = cursor.fetchone()[0]
    if hashlib.sha256(grant_source.encode()).hexdigest() != GRANT_HASH:
        raise ValueError("finite grant source differs")

    cursor.execute("CREATE TEMP TABLE approved_judge_followup (document jsonb NOT NULL) ON COMMIT DROP")
    cursor.execute("INSERT INTO pg_temp.approved_judge_followup VALUES (%s::jsonb)", (json.dumps(approval),))
    # Exact approval replaces the one-ever reopening restriction. A later round
    # can already be running; this transaction does not edit that round or jobs.
    round_source = _between(round_source,
        "  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id\n",
        "  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND a.status IN('pending','leased','submitted'))",
        """  OR EXISTS(SELECT 1 FROM public.lab_arena_rounds newer WHERE
    (newer.evaluation_date>r.evaluation_date OR(newer.evaluation_date IS NULL AND newer.created_at>r.created_at))
    AND newer.arena_network_name=r.arena_network_name AND newer.arena_netuid=r.arena_netuid
    AND newer.configuration_doc->>'mode'='live' AND (newer.status='published' OR newer.reward_activated_at IS NOT NULL))
""")
    # Use the current grant's stricter native timeout / retained-provider proof,
    # instead of695's older subset of failure envelopes. It is checked again
    # atomically before inserting any child, and an error rolls back everything.
    round_source = _between(round_source,
        "   OR NOT((parent.result_doc#>>'{failure_diagnostic,stage}'",
        "   OR parent.scoring_generation_floor<>r.scoring_generation_floor")
    round_source = replace_once(round_source,
        "    (h.request_id=item->>'request_id'OR(h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'\n     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)))THEN",
        "    h.request_id=item->>'request_id')THEN")
    round_source = replace_once(round_source,
        "public.lab_arena_grant_scorer_bug_recovery_v1(item->>'parent_run_id',item->>'parent_sha256',\n   p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,FALSE)",
        "pg_temp.approved_grant_followup(item->>'parent_run_id',item->>'parent_sha256',\n   p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,FALSE,'operator_scorer_bug_recovery_v1',NULL)")
    # The approved parent list replaces only prior-grant caps. All diagnostic,
    # latest-parent, accepted-work, settled-call, deadline and runtime gates stay.
    grant_source = _between(grant_source,
        "  OR (p_recovery_kind<>'operator_judge_budget_exception_v1' AND",
        "  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id\n")
    round_guard = """
 IF NOT EXISTS(SELECT 1 FROM pg_temp.approved_judge_followup a WHERE
  (a.document-ARRAY['final','publication'])=jsonb_build_object('round',p_round_id,
   'sha',p_round_sha256,'runtime',p_runtime_image_digest,'requests',p_requests,
   'request',p_request_id,'actor',p_actor_ref,'approval',p_approval_ref)
  AND (a.document->>'final')::timestamptz=p_final_scoring_close
  AND (a.document->>'publication')::timestamptz=p_publication_deadline) THEN
  RAISE EXCEPTION 'exact owner followup approval differs'; END IF;
"""
    grant_guard = """
 IF p_recovery_kind IS DISTINCT FROM 'operator_scorer_bug_recovery_v1'
  OR p_provider_entry_id IS NOT NULL OR NOT EXISTS(
  SELECT 1 FROM pg_temp.approved_judge_followup a,
   LATERAL jsonb_array_elements(a.document->'requests') item
  WHERE item->>'parent_run_id'=p_parent_run_id AND item->>'parent_sha256'=p_parent_sha256
   AND item->>'request_id'=p_request_id AND a.document->>'runtime'=p_runtime_image_digest
   AND a.document->>'actor'=p_actor_ref AND a.document->>'approval'=p_correction_ref
   AND EXISTS(SELECT 1 FROM public.lab_arena_runs authorized_parent WHERE authorized_parent.run_id=p_parent_run_id
    AND authorized_parent.round_id=a.document->>'round')) THEN
  RAISE EXCEPTION 'exact owner child approval differs'; END IF;
"""
    round_source = replace_once(round_source, "BEGIN\n", "BEGIN\n" + round_guard)
    grant_source = replace_once(grant_source, "BEGIN\n", "BEGIN\n" + grant_guard)
    cursor.execute("""CREATE OR REPLACE FUNCTION pg_temp.approved_grant_followup(
 p_parent_run_id text,p_parent_sha256 text,p_runtime_image_digest text,p_request_id text,
 p_actor_ref text,p_correction_ref text,p_dry_run boolean,p_recovery_kind text,p_provider_entry_id bigint)
 RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog,public AS %s""", (grant_source,))
    cursor.execute("""CREATE OR REPLACE FUNCTION pg_temp.approved_round_followup(
 p_round_id text,p_round_sha256 text,p_runtime_image_digest text,p_final_scoring_close timestamptz,
 p_publication_deadline timestamptz,p_requests jsonb,p_request_id text,p_actor_ref text,
 p_approval_ref text,p_dry_run boolean)
 RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog,public AS %s""", (round_source,))

    cursor.execute("REVOKE ALL ON FUNCTION pg_temp.approved_grant_followup(text,text,text,text,text,text,boolean,text,bigint) FROM PUBLIC")
    cursor.execute("REVOKE ALL ON FUNCTION pg_temp.approved_round_followup(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean) FROM PUBLIC")


def correct(cursor, approval: dict, *, dry_run: bool = True) -> dict:
    values = [approval[k] for k in ("round", "sha", "runtime", "final", "publication")]
    values += [json.dumps(approval["requests"])]
    values += [approval[k] for k in ("request", "actor", "approval")]
    cursor.execute(CALL, values + [dry_run])
    result = cursor.fetchone()[0]
    if not dry_run and not result.get("idempotent"):
        for item, grant in zip(approval["requests"], result["grants"], strict=True):
            for provider in ("openrouter", "deepline"):
                cursor.execute("SELECT public.lab_arena_provider_funding(%s,%s)->>'funding_source', public.lab_arena_provider_funding(%s,%s)->>'funding_source'",
                               (item["parent_run_id"], provider, grant["child_identity"]["run_id"], provider))
                parent_funding, child_funding = cursor.fetchone()
                if parent_funding is None or parent_funding != child_funding:
                    raise ValueError("followup must preserve existing payer")
    return result
