"""One complete source review, paid by its submitting miner, before evaluation."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from lab_arena import broker, code_review, contracts, source_bundle


class SubmissionCodeReviewer:
    """Keep review work outside the scheduler and reuse Arena's cost ledger."""

    def __init__(
        self, *, store: Any, objects: Any,
        credential_for: Callable[[Mapping[str, Any]], str],
        price_table: Mapping[str, Any], transport: broker.ProviderTransport,
    ) -> None:
        self._store = store
        self._objects = objects
        self._credential_for = credential_for
        self._prices = broker.validate_price_table(price_table)
        self._transport = transport

    def review(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        state = row.get("code_review_status")
        if state in ("passed", "rejected"):
            return {"status": "existing", "code_review_status": state}
        if state == "error" and int(row.get("code_review_attempts") or 0) >= 3:
            return {"status": "exhausted", "code_review_status": state}
        timestamp = row.get("code_review_expires_at" if state == "reviewing" else "code_review_started_at")
        if state in ("reviewing", "error") and timestamp:
            moment = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            ready_at = moment if state == "reviewing" else moment + timedelta(seconds=60)
            if datetime.now(timezone.utc) < ready_at:
                return {"status": "busy" if state == "reviewing" else "backoff", "code_review_status": state}
        submission_id = str(row["submission_id"])
        hotkey = str(row["miner_hotkey"])
        prepared = None
        preparation_error = None
        reservation = 0
        try:
            payload = self._objects.get_bounded(
                str(row["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
            if len(payload) != int(row["source_size_bytes"]):
                raise code_review.CodeReviewError("code_review_source_size_mismatch")
            prepared = code_review.prepare_request(payload)
            reservation = broker.max_openrouter_cost_microusd(
                self._prices, prepared.parameters["model"], prepared.parameters,
                max_output_tokens=prepared.parameters["max_tokens"],
            )
        except code_review.CodeReviewError as exc:
            preparation_error = exc.code
        except Exception:
            # Storage/catalog failure is an incomplete review, never a pass.
            preparation_error = "code_review_preparation_unavailable"
        token = secrets.token_hex(32)
        claim = self._store.begin_submission_review(
            submission_id, hotkey, token, reservation,
            code_review.DEFAULT_REVIEW_MODEL,
            len(prepared.reviewed_files) if prepared else 0,
            prepared.reviewed_file_bytes if prepared else 0,
        )
        if claim.get("status") != "claimed":
            return claim
        if preparation_error:
            return self._store.finish_submission_review(
                submission_id, hotkey, token, "error",
                {"error_code": preparation_error, "model": code_review.DEFAULT_REVIEW_MODEL,
                 "file_count": len(prepared.reviewed_files) if prepared else 0,
                 "source_bytes": prepared.reviewed_file_bytes if prepared else 0}, 0,
            )
        actual = 0
        status = "error"
        document: dict[str, Any] = {"error_code": "code_review_provider_unavailable"}
        try:
            secret = self._credential_for(row)
            if not isinstance(secret, str) or not secret:
                raise ValueError("review credential unavailable")
            actual = None
            response = self._transport.send(
                method="POST", url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer " + secret,
                         "Content-Type": "application/json"},
                body=contracts.canonical_json(prepared.parameters).encode("utf-8"),
                timeout_seconds=300,
            )
            if broker._response_contains_credential(response, secret):
                document = {"error_code": "code_review_credential_echo"}
            else:
                parsed = json.loads(response.body.decode("utf-8"))
                actual = broker.actual_openrouter_cost_microusd(
                    self._prices, prepared.parameters["model"], parsed,
                )
                if response.status == 200:
                    result = code_review.parse_response(parsed, prepared)
                    # Validate source excerpts in memory, then persist only
                    # counts and category codes. No submitted instructions or
                    # provider-authored prose enters the durable review row.
                    document = {
                        "passed": result.passed, "verdict": result.verdict,
                        "categories": sorted({item["category"] for item in result.findings}),
                    }
                    status = "passed" if result.passed else "rejected"
                else:
                    document = {"error_code": "code_review_provider_unavailable"}
        except code_review.CodeReviewError as exc:
            document = {"error_code": exc.code}
            if exc.response_reason is not None:
                document["error_reason"] = exc.response_reason
        except Exception:
            # Provider/credential/JSON errors must not enter logs or results.
            document = {"error_code": "code_review_provider_unavailable"}
        # A successful verdict without a confirmed charge cannot be called a
        # completed, accounted review. Preserve the reservation as uncertain.
        if actual is None and status == "passed":
            status = "error"
            document = {"error_code": "code_review_cost_unavailable"}
        document.update({
            "model": code_review.DEFAULT_REVIEW_MODEL,
            "file_count": len(prepared.reviewed_files),
            "source_bytes": prepared.reviewed_file_bytes,
        })
        return self._store.finish_submission_review(
            submission_id, hotkey, token, status, document, actual,
        )
