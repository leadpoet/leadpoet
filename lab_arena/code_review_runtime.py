"""One complete source review, paid by its submitting miner, before evaluation."""

from __future__ import annotations

import json
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from lab_arena import broker, code_review, code_review_policy, contracts, operations, source_bundle, submission_similarity


class SubmissionCodeReviewer:
    """Keep review work outside the scheduler and reuse Arena's cost ledger."""

    def __init__(
        self, *, store: Any, objects: Any,
        credential_for: Callable[[Mapping[str, Any]], str],
        price_table: Mapping[str, Any], transport: broker.ProviderTransport,
        similarity_references_for: Callable[[Mapping[str, Any]], list[Mapping[str, Any]]] | None = None,
    ) -> None:
        self._store = store
        self._objects = objects
        self._credential_for = credential_for
        self._prices = broker.validate_price_table(price_table)
        self._transport = transport
        self._similarity_references_for = similarity_references_for

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
        exact_duplicate = False
        reservation = 0
        try:
            payload = self._objects.get_bounded(
                str(row["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
            if len(payload) != int(row["source_size_bytes"]):
                raise code_review.CodeReviewError("code_review_source_size_mismatch")
            # First validate the full archive, even when a local duplicate can
            # skip the paid review. Persisted digests bind rows to object bytes.
            prepared = code_review.prepare_request(payload)
            if (self._similarity_references_for is not None or row.get("source_archive_sha256")
                    or row.get("source_normalized_sha256")):
                candidate = submission_similarity.inspect_archive(payload)
                if row.get("source_archive_sha256") and row["source_archive_sha256"] != candidate.archive_sha256:
                    raise code_review.CodeReviewError("code_review_source_digest_mismatch")
                if row.get("source_normalized_sha256") and row["source_normalized_sha256"] != candidate.normalized_sha256:
                    raise code_review.CodeReviewError("code_review_source_digest_mismatch")
            if self._similarity_references_for is not None:
                contexts = []
                references = self._similarity_references_for(row)
                if not isinstance(references, (list, tuple)) or len(references) > 257:
                    raise ValueError("similarity reference set unavailable")
                for entry in references:
                    reference_row = entry["row"]
                    reference_id = str(reference_row.get("submission_id") or "public-bootstrap")
                    if reference_id == submission_id:
                        continue
                    if "payload" in entry and not isinstance(entry["payload"], bytes):
                        raise ValueError("bootstrap reference invalid")
                    reference_payload = (
                        entry["payload"] if "payload" in entry else
                        self._objects.get_bounded(
                            str(reference_row["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
                        )
                    )
                    if len(reference_payload) != int(reference_row["source_size_bytes"]):
                        raise ValueError("reference source size mismatch")
                    reference = submission_similarity.inspect_archive(reference_payload)
                    if reference_row.get("source_archive_sha256") and reference_row["source_archive_sha256"] != reference.archive_sha256:
                        raise ValueError("reference source digest mismatch")
                    if reference_row.get("source_normalized_sha256") and reference_row["source_normalized_sha256"] != reference.normalized_sha256:
                        raise ValueError("reference normalized digest mismatch")
                    comparison = submission_similarity.compare(candidate, reference)
                    if comparison.status == "exact":
                        exact_duplicate = True
                    if comparison.status != "exact" and (
                        comparison.status == "ambiguous" or entry["source_public"] is True
                    ):
                        comparison_id = hashlib.sha256(
                            f"{submission_id}:{reference_id}".encode("utf-8")
                        ).hexdigest()[:24]
                        context = submission_similarity.comparison_context(
                            candidate, reference,
                            reference_public=entry["source_public"] is True,
                            comparison_id=comparison_id,
                        )
                        if context is None and comparison.status == "ambiguous":
                            raise ValueError("similarity context unavailable")
                        if context is not None:
                            contexts.append(context)
                if contexts and not exact_duplicate:
                    prepared = code_review.prepare_request(
                        payload, similarity_contexts=tuple(contexts)
                    )
            if not exact_duplicate:
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
        if exact_duplicate:
            return self._store.finish_submission_review(
                submission_id, hotkey, token, "rejected",
                {"passed": False, "verdict": "reject",
                 "categories": ["duplicate_submission"],
                 "model": code_review.DEFAULT_REVIEW_MODEL,
                 "file_count": len(prepared.reviewed_files),
                 "source_bytes": prepared.reviewed_file_bytes}, 0,
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
                             "Content-Type": "application/json",
                             "X-OpenRouter-Metadata": "enabled"},
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
                    try:
                        response = broker._openrouter_effective_response(response)
                    except operations.OperationResponseError as exc:
                        raise code_review.CodeReviewError(
                            "review_response_invalid", response_reason="envelope"
                        ) from exc
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
                    elif (
                        response.status == 403
                        and broker._openrouter_request_policy_refusal(response)
                    ):
                        document = code_review_policy.diagnostic(
                            "code_review_provider_request_rejected",
                            provider_http_status=403,
                            retryable=False,
                        )
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
