"""One complete source review, paid by its submitting miner, before evaluation."""

from __future__ import annotations

import json
import hashlib
import secrets
import time
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from lab_arena import broker, code_review, code_review_policy, contracts, operations, source_bundle, submission_similarity, telemetry, reasoning_capture, training_coverage


_ROUND_NOT_PROVIDED = object()


class SubmissionCodeReviewer:
    """Keep review work outside the scheduler and reuse Arena's cost ledger."""

    def __init__(
        self, *, store: Any, objects: Any,
        credential_for: Callable[[Mapping[str, Any]], str],
        price_table: Mapping[str, Any], transport: broker.ProviderTransport,
        similarity_references_for: Callable[[Mapping[str, Any]], list[Mapping[str, Any]]] | None = None,
        capture: Any = None,
        prospective_primary_model: str = code_review.DEFAULT_REVIEW_MODEL,
    ) -> None:
        if prospective_primary_model not in {code_review.DEFAULT_REVIEW_MODEL, code_review.PROSPECTIVE_REVIEW_MODEL}:
            raise ValueError("unsupported admission primary model")
        self._store = store
        self._objects = objects
        self._credential_for = credential_for
        self._prices = broker.validate_price_table(price_table)
        self._transport = transport
        self._similarity_references_for = similarity_references_for
        self._capture = capture
        self._prospective_primary_model = prospective_primary_model

    @staticmethod
    def _capture_outcome(call, document, result, actual):
        if call is not None:
            try:
                call.finish(outcome={"host_decision": dict(document),
                                     "committed_review": dict(result)},
                            cost_microusd=actual)
            except Exception:
                # Private training evidence never changes admission authority.
                pass

    def _finish_review(
        self, row: Mapping[str, Any], token: str, status: str,
        document: Mapping[str, Any], actual: int | None,
        *, attempt: Any, provider_http_status: int = 0,
        funding_route: str = "none",
        capture_call: Any = None,
    ) -> Mapping[str, Any]:
        try:
            result = self._store.finish_submission_review(
                str(row["submission_id"]), str(row["miner_hotkey"]),
                token, status, document, actual,
            )
        except Exception:
            self._capture_outcome(capture_call, document, {"status": "unavailable"}, actual)
            raise
        self._capture_outcome(capture_call, document, result, actual)
        try:
            # The database may replace a late pass with a terminal error.
            # Emit only its committed decision, never a stale or replay reply.
            if result.get("status") not in ("passed", "rejected", "error", "deadline", "round_closed", "submission_ineligible"):
                return result
            committed = result.get("code_review_status")
            if committed not in ("passed", "rejected", "error"):
                return result
            committed_attempt = int(result.get("attempt") or attempt)
            error_code = document.get("error_code", telemetry.NO_ERROR)
            retryable = False
            if committed == "error":
                retryable = code_review_policy.review_retry_available({
                    "code_review_status": committed,
                    "code_review_attempts": committed_attempt,
                    "code_review_doc": document,
                })
            if (result.get("refusal_recovery") is True
                or type(result.get("review_invalid_attempts")) is int
            ) and type(result.get("retryable")) is bool:
                retryable = result["retryable"]
            if result["status"] in ("deadline", "round_closed", "submission_ineligible"):
                error_code = {
                    "deadline": "code_review_deadline_exceeded",
                    "round_closed": "code_review_round_closed",
                    "submission_ineligible": "code_review_submission_ineligible",
                }[result["status"]]
                retryable = False
            raw_categories = document.get("categories") if committed == "rejected" else None
            categories = (
                ";".join(sorted(set(raw_categories)))
                if isinstance(raw_categories, list)
                and raw_categories
                and all(
                    isinstance(category, str) and category in code_review.REVIEW_CATEGORIES
                    for category in raw_categories
                )
                else telemetry.NO_ERROR
            )
            telemetry.record_review(
                committed,
                error_code=error_code,
                categories=categories,
                provider_http_status=int(document.get("provider_http_status") or provider_http_status),
                retryable=retryable,
                funding_route=funding_route,
                round_id=str(row["round_id"]),
                submission_id=str(row["submission_id"]),
                miner_hotkey=str(row["miner_hotkey"]),
                attempt=committed_attempt,
            )
        except BaseException:
            # Observability cannot turn a persisted review into a worker error.
            pass
        return result

    def review_with_frozen_round(
        self, row: Mapping[str, Any], round_row: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reuse the exact round already read and checked by ArenaService.

        This internal entry point receives service state, never submission
        annotations. Standalone review still reads its authoritative round.
        """
        return self.review(row, _frozen_round=round_row)

    def review(
        self, row: Mapping[str, Any], *, _frozen_round: Any = _ROUND_NOT_PROVIDED,
    ) -> Mapping[str, Any]:
        if code_review_policy.legacy_refusal_candidate(row):
            hydrate = getattr(self._store, "_hydrate_legacy_review", None)
            if callable(hydrate):
                row = hydrate(row)
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
        review_model = code_review_policy.review_model_for(
            row, prospective_primary_model=self._prospective_primary_model)
        review_output_tokens = code_review_policy.review_output_tokens_for(row)
        review_context = (
            code_review.FALLBACK_CONTEXT_WINDOW_TOKENS
            if review_model == code_review.FALLBACK_REVIEW_MODEL
            else code_review.PROSPECTIVE_CONTEXT_WINDOW_TOKENS
            if review_model == code_review.PROSPECTIVE_REVIEW_MODEL
            else code_review.DEFAULT_CONTEXT_WINDOW_TOKENS
        )
        prepared = None
        preparation_error = None
        exact_duplicate = False
        reservation = 0
        research_control = False
        try:
            round_row = _frozen_round
            if round_row is _ROUND_NOT_PROVIDED:
                get_round = getattr(self._store, "get_round", None)
                if callable(get_round):
                    round_row = get_round(str(row["round_id"]))
            if round_row is not _ROUND_NOT_PROVIDED:
                if (not isinstance(round_row, Mapping)
                        or not isinstance(round_row.get("configuration_doc"), Mapping)
                        or (_frozen_round is not _ROUND_NOT_PROVIDED
                            and round_row.get("round_id") != row["round_id"])
                        or round_row.get("round_id", row["round_id"]) != row["round_id"]):
                    raise ValueError("round configuration unavailable")
                configuration = round_row["configuration_doc"]
                research_control = reasoning_capture.enabled(configuration)
                training_coverage.missing_reasoning_percent_from_configuration(configuration)
            payload = self._objects.get_bounded(
                str(row["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
            if len(payload) != int(row["source_size_bytes"]):
                raise code_review.CodeReviewError("code_review_source_size_mismatch")
            # First validate the full archive, even when a local duplicate can
            # skip the paid review. Persisted digests bind rows to object bytes.
            prepared = code_review.prepare_request(
                payload, model=review_model, context_window_tokens=review_context,
                max_output_tokens=review_output_tokens,
                research_control=research_control,
            )
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
                        payload, similarity_contexts=tuple(contexts),
                        model=review_model, context_window_tokens=review_context,
                        max_output_tokens=review_output_tokens,
                        research_control=research_control,
                    )
            if not exact_duplicate:
                if review_model == code_review.FALLBACK_REVIEW_MODEL:
                    # Reuse the existing conservative Sol Azure price ceiling,
                    # including the long-input tier and regional/cache premium.
                    # Admission keeps its own single pinned endpoint policy.
                    route = broker._openrouter_host_route(
                        kind="score", operation_id="openrouter.chat",
                        model=review_model, pricing=self._prices["models"][review_model],
                        parameters=prepared.parameters, industry_review_model=review_model,
                    )
                    reservation = broker._max_openrouter_cost_for_pricing(
                        route.reservation_pricing, prepared.parameters,
                        max_output_tokens=prepared.parameters["max_tokens"], model=review_model,
                    )
                elif review_model == code_review.PROSPECTIVE_REVIEW_MODEL:
                    pricing = dict(self._prices["models"][review_model])
                    for component, ceiling in code_review.PROSPECTIVE_PRICE_CEILING.items():
                        pricing[component] = str(max(Decimal(pricing[component]), Decimal(ceiling)))
                    reservation = broker._max_openrouter_cost_for_pricing(
                        pricing, prepared.parameters,
                        max_output_tokens=prepared.parameters["max_tokens"], model=review_model)
                else:
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
            review_model,
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
            return self._finish_review(
                row, token, "error",
                {**preparation_error, "model": review_model,
                 "file_count": len(prepared.reviewed_files) if prepared else 0,
                 "source_bytes": prepared.reviewed_file_bytes if prepared else 0}, 0,
                attempt=claim.get("attempt"),
            )
        if exact_duplicate:
            return self._finish_review(
                row, token, "rejected",
                {"passed": False, "verdict": "reject",
                 "categories": ["duplicate_submission"],
                 "model": review_model,
                 "file_count": len(prepared.reviewed_files),
                 "source_bytes": prepared.reviewed_file_bytes}, 0,
                attempt=claim.get("attempt"),
            )
        # Historical standalone callers have no recommendation field and use
        # only the default plan. Present malformed fields and elevated plans
        # require an exact match. Production wiring always requires624 first.
        if claim.get("review_output_token_recommendation", code_review.DEFAULT_MAX_OUTPUT_TOKENS) != review_output_tokens:
            # A stale reader can reserve a smaller output plan. Settle the
            # undispatched claim at zero, retaining SQL's authoritative plan
            # for the next bounded attempt; never send an under-reserved call.
            return self._finish_review(
                row, token, "error",
                {**code_review_policy.diagnostic("code_review_preparation_unavailable", retryable=True),
                 "model": review_model, "file_count": len(prepared.reviewed_files),
                 "source_bytes": prepared.reviewed_file_bytes}, 0,
                attempt=claim.get("attempt"),
            )
        actual = 0
        status = "error"
        provider_http_status = 0
        provider_observation: dict[str, Any] = {}
        diagnostic_response: tuple[Any, Mapping[str, Any]] | None = None
        capture_call = None
        funding_route = "none"
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
            # A local failure is incomplete; never expose exception text or
            # claim that the miner's key is invalid without a typed failure.
            document = code_review_policy.diagnostic(
                "code_review_provider_unavailable"
            )
        if secret:
            actual = None
            try:
                request_body = contracts.canonical_json(prepared.parameters).encode("utf-8")
                # This is the owner's stored OpenRouter key; no organizer key
                # is available on this review route.
                funding_route = "miner_key"
                if self._capture is not None:
                    try:
                        capture_call = self._capture.start(
                            purpose="admission_review", api="openrouter.chat",
                            context={
                                "round_id": row["round_id"], "submission_id": submission_id,
                                "attempt": claim.get("attempt"), "funding_source": funding_route,
                                "source_archive_sha256": contracts.hash_bytes(payload),
                                "source_manifest_sha256": prepared.source_manifest_sha256,
                                "research_policy_hash": code_review.RESEARCH_POLICY_HASH if research_control else None,
                            },
                            request=prepared.parameters, secrets=(secret,),
                        )
                    except Exception:
                        pass
                request_started = time.monotonic()
                response = self._transport.send(
                    method="POST", url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": "Bearer " + secret,
                             "Content-Type": "application/json",
                             "X-OpenRouter-Metadata": "enabled"},
                    body=request_body,
                    timeout_seconds=300,
                )
                if capture_call is not None:
                    try:
                        capture_call.response(body=response.body, status=response.status,
                                              elapsed_seconds=time.monotonic() - request_started)
                    except Exception:
                        pass
                provider_http_status = response.status
                if broker._response_contains_credential(response, secret):
                    document = code_review_policy.diagnostic(
                        "code_review_credential_echo", retryable=False
                    )
                else:
                    try:
                        parsed = json.loads(response.body.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        parsed = None
                    diagnostic_response = (parsed, response.headers)
                    if response.internal_provenance is None:
                        provider_observation = code_review_policy.provider_response_observation(
                            parsed, raw_http_status=provider_http_status,
                        )
                    if isinstance(parsed, Mapping):
                        actual = broker.actual_openrouter_cost_microusd(
                            self._prices, prepared.parameters["model"], parsed,
                        )
                    # Reuse the broker's established direct credit-refusal
                    # proof. A complete HTTP 402 before any generation is not
                    # a missing paid result. Embedded failures and ambiguous
                    # usage remain uncertain, as do transport failures.
                    if actual is None and response.status == 402:
                        generation_present, _ = broker._openrouter_generation_identity(
                            parsed, response.headers,
                        )
                        if (
                            broker._openrouter_complete_provider_error(response, parsed, response)
                            and set(parsed) <= {"error", "openrouter_metadata"}
                            and broker._confirmed_credit_failure_proof(
                                provider="openrouter", funding_source="miner_key",
                                response=response, raw_document=parsed,
                                raw_actual=None, actual=0, call_succeeded=False,
                                openrouter_generation_present=generation_present,
                            ) is not None
                        ):
                            actual = 0
                    try:
                        response = broker._openrouter_effective_response(response)
                    except operations.OperationResponseError as exc:
                        raise code_review.CodeReviewError(
                            "review_response_invalid", response_reason="envelope"
                        ) from exc
                    if response.internal_provenance is None:
                        provider_observation = code_review_policy.provider_response_observation(
                            parsed, raw_http_status=provider_http_status,
                            effective_http_status=response.status,
                        )
                    if response.status == 200 and code_review_policy.explicit_provider_refusal(parsed):
                        # Refusal is a provider request failure, never evidence
                        # of source misconduct. Account this call before a new
                        # durable claim can select the one full-source alternate.
                        document = code_review_policy.diagnostic(
                            "code_review_provider_refused",
                            provider_http_status=200,
                            retryable=review_model != code_review.FALLBACK_REVIEW_MODEL,
                        )
                    elif response.status == 200:
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
                        if research_control:
                            document["research_control"] = dict(result.research_control)
                            classification = result.research_control["classification"]
                            if classification == "workflow" and status == "passed":
                                status = "rejected"
                                document.update({"passed": False, "verdict": "reject"})
                            elif classification == "unverified" and status == "passed":
                                status = "error"
                                document.update(code_review_policy.diagnostic(
                                    "code_review_research_unverified", retryable=True
                                ))
                    elif (
                        response.status == 403
                        and broker._openrouter_request_policy_refusal(response)
                    ):
                        document = code_review_policy.diagnostic(
                            "code_review_provider_refused",
                            provider_http_status=403,
                            retryable=review_model != code_review.FALLBACK_REVIEW_MODEL,
                        )
                    else:
                        document = code_review_policy.provider_http_diagnostic(
                            response.status
                        )
                        if provider_http_status == 402 and response.status == 429:
                            document["provider_http_status"] = provider_http_status
            except code_review.CodeReviewError as exc:
                # An unusable provider review is not a negative code verdict.
                # Account this call; SQL also bounds any invalid-review episode
                # by durable reservations. Never loop within a call.
                document = code_review_policy.diagnostic(
                    exc.code, error_reason=exc.response_reason,
                    retryable=(False if exc.response_reason == "truncated_at_limit" else
                               True if exc.code in code_review_policy.TRANSIENT_ERROR_CODES else None),
                )
            except broker.ProviderTransportError:
                # The request may have reached the provider, so its cost is unknown.
                document = code_review_policy.diagnostic(
                    "code_review_transport_error", retryable=True
                )
            except Exception:
                # A local/provider failure remains incomplete, with no raw detail.
                document = code_review_policy.diagnostic(
                    "code_review_provider_unavailable"
                )
        # A successful verdict without a confirmed charge cannot be called a
        # completed, accounted review. Preserve the reservation as uncertain.
        if actual is None and status == "passed":
            status = "error"
            # Missing accounting can mean a completed paid generation. Keep
            # its conservative uncertain charge; never replay a valid verdict for billing.
            document = code_review_policy.diagnostic(
                "code_review_cost_unavailable", retryable=False
            )
        if status == "error" and any(
            provider_observation.get(name, 0) >= 400
            for name in ("provider_raw_http_status", "provider_effective_http_status")
        ):
            document.update(provider_observation)
        previous = row.get("code_review_doc") or {}
        if code_review_policy.refusal_recovery(previous) or document.get("error_code") == "code_review_provider_refused":
            primary_before = int(previous.get("review_primary_attempts") or 0) if code_review_policy.refusal_recovery(previous) else 0
            alternate_before = int(previous.get("review_alternate_attempts") or 0)
            document.update({"refusal_recovery": True,
                "refusal_episode_start": int(previous.get("refusal_episode_start") or claim.get("attempt") or 1),
                "review_primary_attempts": primary_before + (review_model in {
                    code_review.DEFAULT_REVIEW_MODEL, code_review.PROSPECTIVE_REVIEW_MODEL}),
                "review_alternate_attempts": alternate_before + (review_model == code_review.FALLBACK_REVIEW_MODEL)})
        if status == "error" and review_model == code_review.FALLBACK_REVIEW_MODEL:
            previous = row.get("code_review_doc") or {}
            document["retryable"] = (int(previous.get("review_alternate_attempts") or 0) + 1 < 2
                and code_review_policy.recovery_retryable(document))
        document.update({
            "model": review_model,
            "file_count": len(prepared.reviewed_files),
            "source_bytes": prepared.reviewed_file_bytes,
        })
        if diagnostic_response is not None:
            # Decide the review and cost first. Optional diagnostics cannot
            # replace either outcome if their own projection fails.
            try:
                parsed, headers = diagnostic_response
                _present, generation_id = broker._openrouter_generation_identity(parsed, headers)
                document["provider_diagnostic"] = code_review_policy.private_response_diagnostic(
                    parsed, prepared, generation_id=generation_id,
                )
            except Exception:
                pass
        return self._finish_review(
            row, token, status, document, actual,
            attempt=claim.get("attempt"),
            provider_http_status=provider_http_status,
            funding_route=funding_route,
            capture_call=capture_call,
        )
