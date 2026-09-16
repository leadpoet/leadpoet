"""One complete source review, paid by its submitting miner, before evaluation."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from lab_arena import broker, code_review, code_review_policy, contracts, source_bundle


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
        if state == "error" and not code_review_policy.review_retry_available(row):
            return {"status": "exhausted", "code_review_status": state}
        timestamp = row.get("code_review_expires_at" if state == "reviewing" else "code_review_started_at")
        if state in ("reviewing", "error") and timestamp:
            moment = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            ready_at = (
                moment if state == "reviewing"
                else code_review_policy.retry_ready_at(row)
            )
            if ready_at is None:
                return {"status": "exhausted", "code_review_status": state}
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
            preparation_error = code_review_policy.source_diagnostic(exc.code)
        except Exception:
            # Storage/catalog failure is an incomplete review, never a pass.
            preparation_error = code_review_policy.diagnostic(
                "code_review_preparation_unavailable", retryable=True
            )
        token = secrets.token_hex(32)
        claim = self._store.begin_submission_review(
            submission_id, hotkey, token, reservation,
            code_review.DEFAULT_REVIEW_MODEL,
            len(prepared.reviewed_files) if prepared else 0,
            prepared.reviewed_file_bytes if prepared else 0,
        )
        if claim.get("status") != "claimed":
            return claim
        # An exact claim replay proves that the original caller may already
        # have sent the paid request. Never dispatch that identity again.
        if claim.get("idempotent") is True:
            return claim
        if preparation_error:
            return self._store.finish_submission_review(
                submission_id, hotkey, token, "error",
                {**preparation_error, "model": code_review.DEFAULT_REVIEW_MODEL,
                 "file_count": len(prepared.reviewed_files) if prepared else 0,
                 "source_bytes": prepared.reviewed_file_bytes if prepared else 0}, 0,
            )
        actual = 0
        status = "error"
        document: dict[str, Any] = code_review_policy.diagnostic(
            "code_review_provider_unavailable"
        )
        try:
            secret = self._credential_for(row)
            if not isinstance(secret, str) or not secret:
                raise ValueError("review credential unavailable")
        except broker.BrokerError as exc:
            secret = ""
            if exc.code == "broker_unavailable":
                document = code_review_policy.diagnostic(
                    "code_review_preparation_unavailable", retryable=True
                )
            elif exc.code == "miner_credentials_unavailable":
                document = code_review_policy.diagnostic(
                    "code_review_provider_authentication", retryable=False
                )
            else:
                document = code_review_policy.diagnostic(
                    "code_review_provider_unavailable"
                )
        except Exception:
            secret = ""
            # Unknown local failures retain the legacy cap without exposing
            # exception text or claiming that the miner's key is invalid.
            document = code_review_policy.diagnostic(
                "code_review_provider_unavailable"
            )
        if secret:
            actual = None
            try:
                response = self._transport.send(
                    method="POST", url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": "Bearer " + secret,
                             "Content-Type": "application/json"},
                    body=contracts.canonical_json(prepared.parameters).encode("utf-8"),
                    timeout_seconds=300,
                )
                if broker._response_contains_credential(response, secret):
                    document = code_review_policy.diagnostic(
                        "code_review_credential_echo", retryable=False
                    )
                else:
                    try:
                        parsed = json.loads(response.body.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        parsed = None
                    if isinstance(parsed, Mapping):
                        actual = broker.actual_openrouter_cost_microusd(
                            self._prices, prepared.parameters["model"], parsed,
                        )
                    if response.status == 200:
                        if not isinstance(parsed, Mapping):
                            raise code_review.CodeReviewError(
                                "review_response_invalid", response_reason="envelope"
                            )
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
                        document = code_review_policy.provider_http_diagnostic(
                            response.status
                        )
            except code_review.CodeReviewError as exc:
                # A malformed successful generation may already have been paid.
                # Retain the legacy three-attempt cap rather than expanding it.
                document = code_review_policy.diagnostic(
                    exc.code, error_reason=exc.response_reason
                )
            except broker.ProviderTransportError:
                # The request may have reached the provider, so its cost is unknown.
                document = code_review_policy.diagnostic(
                    "code_review_transport_error", retryable=True
                )
            except Exception:
                # Unknown failures retain the legacy cap and no raw detail.
                document = code_review_policy.diagnostic(
                    "code_review_provider_unavailable"
                )
        # A successful verdict without a confirmed charge cannot be called a
        # completed, accounted review. Preserve the reservation as uncertain.
        if actual is None and status == "passed":
            status = "error"
            # Missing accounting can mean a completed paid generation. Keep
            # its conservative uncertain charge and the legacy retry cap.
            document = code_review_policy.diagnostic(
                "code_review_cost_unavailable"
            )
        document.update({
            "model": code_review.DEFAULT_REVIEW_MODEL,
            "file_count": len(prepared.reviewed_files),
            "source_bytes": prepared.reviewed_file_bytes,
        })
        return self._store.finish_submission_review(
            submission_id, hotkey, token, status, document, actual,
        )
