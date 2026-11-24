"""Look up an unsettled synthetic call on the exact original OpenRouter account."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

from lab_arena import broker, provider_costs, store
from lab_arena.credentials import CredentialManager
from scripts.run_lab_arena_service import load_scoped_environment

store.FUNCTION_SIGNATURES.update(
    {
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
    response = http.send(
        method="GET",
        url="https://openrouter.ai/api/v1/generation?" + urlencode({"id": generation}),
        headers={"authorization": "Bearer " + secret},
        body=b"",
        timeout_seconds=12,
    )
    if response.status != 200:
        return {"status": "unresolved", "generation_lookup_http": response.status}
    try:
        data = json.loads(response.body)
    except (ValueError, UnicodeError):
        return {"status": "unresolved", "reason": "invalid_billing_response"}
    cost = provider_costs.openrouter_generation_cost(data, generation_id=generation)
    if cost is None:
        return {"status": "unresolved", "reason": "no_exact_cost"}
    return db.rpc(
        "collection_settle_generation_v1",
        {
            **original,
            "p_generation": generation,
            "p_response_hash": subject["response_hash"],
            "p_lookup_hash": hashlib.sha256(response.body).hexdigest(),
            "p_cost": cost.microusd,
        },
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--environment-file", type=Path, required=True)
    args = p.parse_args(argv)
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
        result = reconcile(db, vault, http, args.job, args.request)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "settled" else 2
    finally:
        http.close()


if __name__ == "__main__":
    raise SystemExit(main())
