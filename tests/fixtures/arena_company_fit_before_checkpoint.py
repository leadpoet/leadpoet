"""Frozen pre-fix scorer from b56af2539; offline regression comparison only."""
async def score_company_competition_intent(
    company: CompanyOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str],
    force_fail_reason: Optional[str] = None,
    is_reference_model: bool = False,
    integrity_policy: bool = False,
    company_quality: bool = False,
    evidence_investigator: bool = False,
    required_attribute_retry_source_cache: Optional[
        dict[str, dict[str, Any]]
    ] = None,
    matched_company_retry_source_cache: Optional[
        MutableMapping[str, Any]
    ] = None,
    verified_company_fit_retry_cache: Optional[MutableMapping[str, Any]] = None,
    intent_terminal_retry_cache: Optional[MutableMapping[str, Any]] = None,
    retry_evidence_context_key: str = "",
    provider_observation: Optional[Mapping[str, Any]] = None,
) -> LeadScoreBreakdown:
    """Score one Arena company with binary fit gates and 0-100 intent score.

    Company-fit gates run before intent evidence verification.
    """
    if force_fail_reason:
        logger.info(
            f"Competition company forced to fail: {force_fail_reason}"
        )
        return _zero_company_breakdown(force_fail_reason)

    linkedin_profile_source_candidate: dict[str, str] = {}
    matched_company_source_cache: dict[str, dict[str, str]] = {}
    # The adapter binds this private map to the complete frozen company, ICP,
    # policy and evaluation date. It lasts for one score_work_item invocation.
    # Retrying intent or paragraph verification must not replace a completed
    # fit and change the evidence already bound to its signal-cache claims.
    fit_cache = (
        verified_company_fit_retry_cache
        if integrity_policy
        and isinstance(verified_company_fit_retry_cache, MutableMapping)
        else None
    )
    retained_fit = fit_cache.get("verified_fit") if fit_cache is not None else None
    if retained_fit is not None:
        (company_fit, linkedin_profile_source_candidate,
         matched_company_source_cache) = deepcopy(retained_fit)
    else:
        company_fit = await _verify_company_fit(
            company,
            icp,
            run_cost_usd,
            run_time_seconds,
            seen_companies,
            require_https_transport=True,
            company_quality=company_quality,
            evidence_investigator=evidence_investigator,
            required_attribute_retry_source_cache=(
                required_attribute_retry_source_cache
            ),
            linkedin_profile_source_sink=linkedin_profile_source_candidate,
            matched_company_source_sink=matched_company_source_cache,
            matched_company_retry_source_cache=(
                matched_company_retry_source_cache
            ),
        )
        if fit_cache is not None and company_fit.decision == COMPANY_FIT_MATCH:
            fit_cache["verified_fit"] = deepcopy((
                company_fit, linkedin_profile_source_candidate,
                matched_company_source_cache,
            ))
    _retain_matched_company_retry_sources(
        matched_company_retry_source_cache, matched_company_source_cache, company_fit,
    )
    gate_receipts = [company_fit.receipt("company_fit")]
    if company_fit.decision != COMPANY_FIT_MATCH:
        return _zero_company_breakdown(
            _company_fit_failure_reason("Company fit", company_fit),
            verifier_gate_receipts=gate_receipts,
        )

    if company.company_name:
        seen_companies.add(company.company_name.lower().strip())

    funding_stage_context = [
        context for context in _matched_investigator_source_contexts(
            company_fit, matched_company_source_cache,
        ) if context["dimension"] == "stage"
    ] if integrity_policy and any(
        str(evidence_type or "").upper() == "FUNDING"
        for evidence_type in getattr(icp, "intent_signal_evidence_types", None) or []
    ) else []
    if funding_stage_context:
        attribute = getattr(company, "required_attribute", None)
        attribute_url = _valid_web_evidence_url(getattr(attribute, "evidence_url", None))
        attribute_entry = matched_company_source_cache.get(attribute_url)
        attribute_context = _validated_first_party_source_context(
            attribute_url, attribute_entry, verified_identity_receipt(gate_receipts),
            company_name=company.company_name,
        )
        excluded = [attribute_url] if attribute_context and _quote_occurs(
            getattr(attribute, "evidence_quote", ""), attribute_context["text"],
        ) else []
        auxiliary = _matched_first_party_source_context(
            company_fit, matched_company_source_cache, company,
            paragraph=company.intent_details or "", excluded_urls=excluded,
        )
        if auxiliary is not None:
            funding_stage_context.append(auxiliary)
            _retain_matched_company_retry_sources(
                matched_company_retry_source_cache, matched_company_source_cache, company_fit,
                funding_contexts=funding_stage_context,
            )
    try:
        (
            intent_raw,
            intent_final,
            decay_multiplier,
            _max_confidence,
            all_signals_unverified,
            signal_results,
        ) = await score_company_competition_intent_signal(
            company,
            icp,
            integrity_policy=integrity_policy,
            company_quality=company_quality,
            verified_company_identity=verified_identity_receipt(gate_receipts),
            **({"verified_funding_stage_context": funding_stage_context}
               if funding_stage_context else {}),
            intent_terminal_retry_cache=intent_terminal_retry_cache,
            retry_evidence_context_key=retry_evidence_context_key,
            **({"verified_source_pages": matched_company_source_cache}
               if integrity_policy and matched_company_source_cache else {}),
        )
        if _intent_verifier_unavailable(
            signal_results, integrity_policy=integrity_policy
        ):
            return _zero_company_breakdown(
                "Intent verification unavailable: verifier provider error",
                intent_signals_detail=signal_results,
                verifier_gate_receipts=gate_receipts or None,
            )
        if all_signals_unverified:
            logger.warning(
                f"All competition intent signals failed for company "
                f"{company.company_name!r} — zeroing entire score"
            )
            return _zero_company_breakdown(
                _competition_intent_failure_reason(signal_results),
                intent_signals_detail=signal_results,
                verifier_gate_receipts=gate_receipts or None,
            )
    except SignalJudgmentCacheError:
        # An invalid trusted cache response is an infrastructure failure, not
        # a terminal company or intent mismatch.
        raise
    except Exception as e:
        logger.error(f"Competition intent scoring failed: {e}")
        return _zero_company_breakdown(f"LLM scoring error: {str(e)[:100]}")

    final_score = max(0.0, min(float(MAX_COMPETITION_INTENT_SCORE), intent_final))
    if final_score > 0 and getattr(company, "intent_details", None) is not None:
        from qualification.scoring.intent_details import review_intent_details

        matched_provider_observation = _matched_provider_observation(
            company,
            company_fit,
            signal_results,
            provider_observation,
        )
        details_receipt = await review_intent_details(
            company,
            icp,
            signal_results,
            company_fit.receipt("company_fit"),
            company_source_contexts=_matched_company_source_contexts(
                company_fit,
                required_attribute_retry_source_cache,
                linkedin_profile_source_candidate,
                matched_company_source_cache,
                paragraph=str(company.intent_details or ""),
                company=company,
                verified_signal_source_urls=_verified_signal_context_urls(
                    signal_results
                ),
            ),
            **(
                {
                    "authenticated_provider_observation":
                        matched_provider_observation
                }
                if matched_provider_observation is not None else {}
            ),
        )
        gate_receipts.append(details_receipt)
        if details_receipt["decision"] != COMPANY_FIT_MATCH:
            return _zero_company_breakdown(
                (
                    "Intent Details verification unavailable: review could not complete"
                    if details_receipt["decision"] == COMPANY_FIT_UNAVAILABLE
                    else "Intent Details do not satisfy the grounded client paragraph contract"
                ),
                intent_signals_detail=signal_results,
                verifier_gate_receipts=gate_receipts,
            )
    role_tag = "reference" if is_reference_model else "miner"
    logger.info(
        f"Competition company scored [{role_tag}]: {final_score:.2f} "
        f"(IntentV2:{intent_final:.2f}, cost=${run_cost_usd:.4f}, "
        f"time={run_time_seconds:.1f}s)"
    )
    return LeadScoreBreakdown(
        icp_fit=0,
        decision_maker=0,
        intent_signal_raw=intent_raw,
        time_decay_multiplier=decay_multiplier,
        intent_signal_final=final_score,
        cost_penalty=0,
        time_penalty=0,
        final_score=final_score,
        failure_reason=(
            _competition_intent_failure_reason(signal_results)
            if final_score == 0.0
            and icp.intent_signals
            and not required_intent_satisfied(signal_results)
            else None
        ),
        intent_signals_detail=signal_results,
        verifier_gate_receipts=gate_receipts or None,
    )
