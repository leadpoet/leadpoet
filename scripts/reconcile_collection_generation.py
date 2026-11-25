"""Look up an unsettled synthetic call on the exact original OpenRouter account."""

import argparse
import hashlib
import json
import os
import time
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

from lab_arena import broker, provider_costs, store
from lab_arena.credentials import CredentialError, CredentialManager
from scripts.run_lab_arena_service import load_scoped_environment

store.FUNCTION_SIGNATURES.update(
    {
        "collection_pending_billing_v1": (
            ("p_after_job", "uuid"),
            ("p_after_request", "text"),
        ),
        "collection_billing_subject_v1": (("p_job", "uuid"), ("p_request", "text")),
        "collection_settle_generation_v1": (
            ("p_job", "uuid"),
            ("p_request", "text"),
            ("p_generation", "text"),
            ("p_response_hash", "text"),
            ("p_lookup_hash", "text"),
            ("p_cost", "bigint"),
        ),
    }
)


def exact_generation(http, secret, generation):
    response = http.send(
        method="GET",
        url="https://openrouter.ai/api/v1/generation?" + urlencode({"id": generation}),
        headers={"authorization": "Bearer " + secret},
        body=b"",
        timeout_seconds=12,
    )
    if response.status != 200:
        return None, {"status": "unresolved", "generation_lookup_http": response.status}
    try:
        data = json.loads(response.body)
    except (ValueError, UnicodeError):
        return None, {"status": "unresolved", "reason": "invalid_billing_response"}
    cost = provider_costs.openrouter_generation_cost(data, generation_id=generation)
    if cost is None:
        return None, {"status": "unresolved", "reason": "no_exact_cost"}
    return cost, hashlib.sha256(response.body).hexdigest()


def reconcile(db, vault, http, job, request):
    original = {"p_job": str(UUID(job)), "p_request": request}
    subject = db.rpc("collection_billing_subject_v1", original)
    if not subject:
        return {"status": "unresolved", "reason": "no_known_original_generation"}
    if subject["cost_microusd"] is not None:
        return {"status": "settled", "cost_microusd": subject["cost_microusd"]}
    if not subject["credential"]:
        return {"status": "unresolved", "reason": "funding_revoked"}
    generation = subject["generation"]
    secret = vault.runtime_key(subject["credential"], "openrouter")
    cost, evidence = exact_generation(http, secret, generation)
    if cost is None:
        return evidence
    return db.rpc(
        "collection_settle_generation_v1",
        {
            **original,
            "p_generation": generation,
            "p_response_hash": subject["response_hash"],
            "p_lookup_hash": evidence,
            "p_cost": cost.microusd,
        },
    )


def reconcile_owner(db, http, subject, owner_key):
    # Owner calls do not store a miner key ID. Verify access to a positive native
    # charge from this same job before looking up or crediting its missing charge.
    if not owner_key:
        return {"status": "unresolved", "reason": "owner_key_unavailable"}
    known, _ = exact_generation(http, owner_key, subject["control_generation"])
    if known is None or known.microusd != subject["control_cost"]:
        return {"status": "unresolved", "reason": "owner_account_unverified"}
    cost, evidence = exact_generation(http, owner_key, subject["generation"])
    if cost is None:
        return evidence
    from lab_arena import (
        collection_launch_review,
    )  # registers the restricted owner-ledger RPC

    result = db.rpc(
        "collection_reconcile_launch_judge_v1",
        {
            "p_job": subject["job"],
            "p_source": subject["source"],
            "p_request": subject["request"],
            "p_generation": subject["generation"],
            "p_cost": cost.microusd,
            "p_hash": evidence,
        },
    )
    if result.get("status") != "reconciled":
        raise ValueError("owner receipt was not registered")
    return {"status": "settled", "cost_microusd": cost.microusd}


def reconcile_pending(db, look_up, state_file, *, now):
    state = json.loads(state_file.read_bytes()) if state_file.exists() else {}
    cursor = state.get("cursor")
    if cursor is None and now < state.get("next_sweep_at", 0):
        return {"checked": 0, "status": "waiting"}
    page = db.rpc(
        "collection_pending_billing_v1",
        {
            "p_after_job": cursor[0] if cursor else None,
            "p_after_request": cursor[1] if cursor else None,
        },
    )
    if not isinstance(page, list) or len(page) > 2:
        raise ValueError("invalid collection billing page")
    outcomes = []
    for item in page:
        try:
            result = look_up(item)
            reason = result.get("reason", result.get("generation_lookup_http"))
            outcomes.append(
                {
                    "provider": item["provider"],
                    "status": result["status"],
                    "reason": reason,
                }
            )
        except (
            broker.ProviderTransportError,
            CredentialError,
            store.ArenaStoreError,
        ) as exc:
            outcomes.append(
                {
                    "provider": item["provider"],
                    "status": "unresolved",
                    "reason": type(exc).__name__,
                }
            )
    # The SQL ledger makes settlement idempotent if we restart before saving this
    # cursor. An old persistent 404 cannot starve a later job or cause a fast loop.
    state = (
        {"cursor": [page[-1]["job"], page[-1]["request"]]}
        if len(page) == 2
        else {"next_sweep_at": now + 3600}
    )
    with tempfile.NamedTemporaryFile(
        mode="w", dir=state_file.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(state_file)
    return {"checked": len(page), "outcomes": outcomes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job")
    mode.add_argument(
        "--pending-state",
        type=Path,
        help="Check one bounded billing page from the existing watcher",
    )
    p.add_argument("--request")
    p.add_argument("--environment-file", type=Path, required=True)
    args = p.parse_args(argv)
    if bool(args.job) != bool(args.request):
        p.error("--request is required with --job only")
    os.umask(0o077)
    load_scoped_environment(args.environment_file)
    from lab_arena.collection_intake import live_finney_arena

    if not live_finney_arena():
        p.error("Finney SN71 gateway environment required")
    db = store.PostgrestTransport(
        os.environ["LAB_ARENA_SUPABASE_URL"],
        service_key=os.environ["LAB_ARENA_SERVICE_KEY"],
    )
    vault = CredentialManager(kms_key_id=os.environ["LAB_ARENA_CREDENTIAL_KMS_KEY_ID"])
    http = broker.HttpxProviderTransport()
    try:
        if args.pending_state:

            def look_up(item):
                if item["provider"] == "judge":
                    key = os.getenv("LEADPOET_COLLECTION_JUDGE_KEY") or os.getenv(
                        "LAB_ARENA_OPENROUTER_API_KEY"
                    )
                    return reconcile_owner(db, http, item, key)
                return reconcile(db, vault, http, item["job"], item["request"])

            result = reconcile_pending(
                db, look_up, args.pending_state, now=int(time.time())
            )
            print(json.dumps({"collection_billing": result}, sort_keys=True))
            return 0
        result = reconcile(db, vault, http, args.job, args.request)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "settled" else 2
    finally:
        http.close()


if __name__ == "__main__":
    raise SystemExit(main())
