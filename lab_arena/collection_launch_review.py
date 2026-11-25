"""Owner-funded judge calls for the existing, completed first live job."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlencode
from uuid import UUID

from lab_arena import broker, provider_costs, store
from lab_arena.collection_broker import CollectionBroker, CollectionCallError
from lab_arena.collection_intake import _validate_quality

_SCOPE = (("p_job", "uuid"), ("p_source", "text"), ("p_attempt", "uuid"))
_CALL = _SCOPE + (("p_request", "text"), ("p_hash", "text"))
store.FUNCTION_SIGNATURES.update(
    {
        "collection_reconcile_launch_judge_v1": (
            ("p_job", "uuid"),
            ("p_source", "text"),
            ("p_request", "text"),
            ("p_generation", "text"),
            ("p_cost", "bigint"),
            ("p_hash", "text"),
        ),
        "collection_reserve_launch_judge_v1": _CALL + (("p_amount", "bigint"),),
        "collection_record_launch_judge_v1": _CALL
        + (
            ("p_provider_id", "text"),
            ("p_cost", "bigint"),
            ("p_status", "integer"),
        ),
        "collection_finish_launch_review_v1": _SCOPE
        + (
            ("p_artifact", "text"),
            ("p_quality", "jsonb"),
            ("p_requests", "jsonb"),
        ),
    }
)


def _scope(pending, attempt_id):
    return {
        "p_job": pending["job_id"],
        "p_source": pending["artifact_sha256"],
        "p_attempt": str(UUID(str(attempt_id))),
    }


class LaunchReviewBroker(CollectionBroker):
    """Use the locked judge adapter while granting only owner-funded requests."""

    def __init__(
        self, database, transport, pending, pricing, *, routes, judge_key, attempt_id
    ):
        super().__init__(
            None,
            database,
            transport,
            {**pending, "mode": pending.get("mode", "live")},
            pricing,
            routes=routes,
            judge_key=judge_key,
        )
        self._scope = _scope(pending, attempt_id)
        self.completed_requests = []

    def _call(self, request_id, provider, url, content, reserve, **kwargs):
        if provider != "judge":
            raise CollectionCallError(
                "launch review allows only owner-funded judgments"
            )
        return super()._call(request_id, provider, url, content, reserve, **kwargs)

    def _rpc(self, name, request_id, **params):
        if name == "collection_reserve_call_v1" and params.pop("p_provider") == "judge":
            target = "collection_reserve_launch_judge_v1"
        elif name == "collection_record_call_v1":
            target = "collection_record_launch_judge_v1"
        else:
            raise CollectionCallError("unexpected launch review operation")
        arguments = {**self._scope, "p_request": request_id, **params}
        result = self.database.rpc(target, arguments)
        if (
            target == "collection_reserve_launch_judge_v1"
            and result.get("status") == "unresolved_prior_charge"
        ):
            if self._recover_owner_costs(result["judge_receipts"]):
                result = self.database.rpc(target, arguments)
        return result

    def _recover_owner_costs(self, receipts):
        recovered = False
        for receipt in receipts:
            generation = receipt["generation_id"]
            try:
                lookup = self.transport.send(
                    method="GET",
                    url="https://openrouter.ai/api/v1/generation?"
                    + urlencode({"id": generation}),
                    headers={"authorization": "Bearer " + self.judge_key},
                    body=b"",
                    timeout_seconds=8,
                )
                if lookup.status != 200:
                    continue
                cost = provider_costs.openrouter_generation_cost(
                    json.loads(lookup.body), generation_id=generation
                )
            except (broker.ProviderTransportError, UnicodeError, ValueError):
                continue
            if cost is not None:
                self.database.rpc(
                    "collection_reconcile_launch_judge_v1",
                    {
                        "p_job": self._scope["p_job"],
                        "p_source": self._scope["p_source"],
                        "p_request": receipt["request_id"],
                        "p_generation": generation,
                        "p_cost": cost.microusd,
                        "p_hash": hashlib.sha256(lookup.body).hexdigest(),
                    },
                )
                recovered = True
        return recovered

    def judge(self, **kwargs):
        reply = super().judge(**kwargs)
        if reply:
            self.completed_requests.append(kwargs["request_id"])
        return reply


def finish_review(
    database, pending, attempt_id, artifact_sha256, public_quality, completed_requests
):
    _validate_quality(public_quality)
    result = database.rpc(
        "collection_finish_launch_review_v1",
        {
            **_scope(pending, attempt_id),
            "p_artifact": artifact_sha256,
            "p_quality": public_quality,
            "p_requests": list(completed_requests),
        },
    )
    if result.get("artifact_sha256") != artifact_sha256:
        raise CollectionCallError("launch review manifest was not registered")
    return result
