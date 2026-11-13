"""Apply reviewed feedback and private failure-source preservation on50de.

The complete preimage and postimage are checksum-bound. All other frozen
source bytes, grader policy and process configuration stay unchanged.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "investigator_prompt_feedback_v2"
FROZEN_IMAGE_DIGEST = "sha256:50deca7b8bf5129816abaeb59fd5bc1aabb554c529cc4151b8175f9edd0b8144"
REVIEWED_SOURCE_COMMIT = "b383dd1815cc5b64ee3fcaf3a781d21d5f44761b"
ORIGINAL_SHA256 = "eaa205dfe6a1707199246008f7108c5883e1d298b6c7c682d41b16c55e677483"
REPAIRED_SHA256 = "14f2bf1fbe3dbcf384c9fb5760afb04c0c412028226fdca62b94b1ad37e9a37d"

# Preserve the reviewed342545-to-25cc feedback delta, then apply only the six
# reviewed private source handoffs from b383dd181. Context makes each replacement
# unique; whole-file checks reject all other drift. No verdict changes.
DELTAS = (
    (
        (
            b'when unproven.\n'
            b'Cite PE/private-markets firm proof in primary or support 1; control or\n'
            b'private-markets allocation activity alone is insufficient.\n'
            b'The primary quote must name the owner and prove current control. It must name\n'
            b'the company or use a company alias explicitly bound in same-page cited support.\n'
            b'Supporting quote 1 must classify the owner if the primary does not; bind owner\n'
        ),
        (
            b'when unproven.\n'
            b'Cite PE/private-markets firm proof in primary or support 1; control or\n'
            b'private-markets allocation activity alone is insufficient.\n'
            b'Absent accepted direct/linked literal ownership proof, primary or support 1 must\n'
            b'literally say "private equity" (spaces/hyphens). Generic equity funds do not qualify.\n'
            b'The primary quote must name the owner and prove current control. It must name\n'
            b'the company or use a company alias explicitly bound in same-page cited support.\n'
            b'Supporting quote 1 must classify the owner if the primary does not; bind owner\n'
        ),
    ),
    (
        (
            b'                                        "owner_type": {\n'
            b'                                            "type": "string",\n'
            b'                                            "description": (\n'
            b'                                                "One of: private_equity, other, unproven."\n'
            b'                                            ),\n'
            b'                                        },\n'
            b'                                    },\n'
        ),
        (
            b'                                        "owner_type": {\n'
            b'                                            "type": "string",\n'
            b'                                            "description": (\n'
            b'                                                "One of: private_equity, other, unproven. "\n'
            b'                                                "Unless accepted direct or linked literal ownership proof "\n'
            b'                                                "already qualifies, private_equity requires the literal "\n'
            b'                                                "words \'private equity\' (spaces or hyphens allowed) in the "\n'
            b'                                                "primary quote or supporting_evidence_quote_1. Investment "\n'
            b'                                                "firm or dedicated equity/credit funds alone is insufficient. "\n'
            b'                                                "Existing literal private-markets firm proofs remain valid; "\n'
            b'                                                "all company, same-owner, control and current-stage checks apply."\n'
            b'                                            ),\n'
            b'                                        },\n'
            b'                                    },\n'
        ),
    ),
    (
        (
            b'        }\n'
            b"        parameters['required'].append('industry_brand_source')\n"
            b'    return tools\n'
            b'\n'
            b'\n'
            b'def _record_failure(diagnostic: Optional[dict[str, str]], reason: str) -> None:\n'
        ),
        (
            b'        }\n'
            b"        parameters['required'].append('industry_brand_source')\n"
            b'    return tools\n'
            b'\n'
            b'\n'
            b'def _final_submission_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:\n'
            b'    """Align stale ordinary-repair advice with this request\'s submit-only admission."""\n'
            b'\n'
            b'    result = dict(feedback)\n'
            b'    instruction = str(result.get("instruction") or "")\n'
            b'    result["instruction"] = instruction.replace(\n'
            b'        "First repair the quote from an already fetched page. Fetch another useful "\n'
            b'        "source only if no continuous company-bound span on that page proves the fact. ",\n'
            b'        "This request is now submit-only; do not search or fetch. Use only already "\n'
            b'        "loaded evidence and preserve all existing evidence rules. ",\n'
            b'    )\n'
            b'    return result\n'
            b'\n'
            b'\n'
            b'def _record_failure(diagnostic: Optional[dict[str, str]], reason: str) -> None:\n'
        ),
    ),
    (
        (
            b'    same_page_identity_quotes: Sequence[str] = (),\n'
            b'    classification_quote: str = "",\n'
            b'    literal_ownership_proved: bool = False,\n'
            b') -> bool:\n'
            b'    """Bind the investigator\'s ownership judgment to literal source entities.\n'
            b'\n'
        ),
        (
            b'    same_page_identity_quotes: Sequence[str] = (),\n'
            b'    classification_quote: str = "",\n'
            b'    literal_ownership_proved: bool = False,\n'
            b'    failure_details: Optional[list[str]] = None,\n'
            b') -> bool:\n'
            b'    """Bind the investigator\'s ownership judgment to literal source entities.\n'
            b'\n'
        ),
    ),
    (
        (
            b'    if not isinstance(raw, Mapping) or set(raw) != {\n'
            b'        "controlling_owner", "control_status", "owner_type",\n'
            b'    }:\n'
            b'        return False\n'
            b'    owner = raw.get("controlling_owner")\n'
            b'    if (\n'
            b'        not isinstance(owner, str) or not owner.strip() or len(owner) > 200\n'
            b'        or owner != owner.strip()\n'
            b'        or raw.get("control_status") != "current_control"\n'
            b'        or raw.get("owner_type") != "private_equity"\n'
            b'    ):\n'
            b'        return False\n'
            b'    # Retain existing literal ownership proofs; otherwise require quoted PE\n'
            b'    # grounding, which alone does not prove classification or the same owner.\n'
            b'    # Support slot 2 and uncited page text cannot replace classification slot 1.\n'
        ),
        (
            b'    if not isinstance(raw, Mapping) or set(raw) != {\n'
            b'        "controlling_owner", "control_status", "owner_type",\n'
            b'    }:\n'
            b'        if failure_details is not None:\n'
            b'            failure_details.append(\n'
            b'                "ownership must contain exactly controlling_owner, control_status, and owner_type."\n'
            b'            )\n'
            b'        return False\n'
            b'    owner = raw.get("controlling_owner")\n'
            b'    failures: list[str] = []\n'
            b'    if (\n'
            b'        not isinstance(owner, str) or not owner.strip() or len(owner) > 200\n'
            b'        or owner != owner.strip()\n'
            b'    ):\n'
            b'        if failure_details is not None:\n'
            b'            failure_details.append(\n'
            b'                "ownership.controlling_owner must be a nonempty trimmed name of at most 200 characters."\n'
            b'            )\n'
            b'        return False\n'
            b'    if raw.get("control_status") != "current_control":\n'
            b'        failures.append(\n'
            b'            "ownership.control_status must be current_control; minority, pending, and former interests do not prove current control."\n'
            b'        )\n'
            b'    if raw.get("owner_type") != "private_equity":\n'
            b'        failures.append("ownership.owner_type must be private_equity.")\n'
            b'    # Retain existing literal ownership proofs; otherwise require quoted PE\n'
            b'    # grounding, which alone does not prove classification or the same owner.\n'
            b'    # Support slot 2 and uncited page text cannot replace classification slot 1.\n'
        ),
    ),
    (
        (
            b'        re.search(r"\\bprivate[\\s\\-\\u2010-\\u2015]+equity\\b", _normalized_span(quote))\n'
            b'        for quote in (control_quote, classification_quote)\n'
            b'    ):\n'
            b'        return False\n'
            b'    # Stage identity keys discard non-ASCII letters as well as punctuation and\n'
            b'    # spaces. Allow those characters inside the same literal identity, while\n'
            b'    # retaining outer word boundaries on the original Unicode quote.\n'
        ),
        (
            b'        re.search(r"\\bprivate[\\s\\-\\u2010-\\u2015]+equity\\b", _normalized_span(quote))\n'
            b'        for quote in (control_quote, classification_quote)\n'
            b'    ):\n'
            b'        failures.append(\n'
            b'            "The primary quote did not qualify under an existing direct or linked literal ownership proof, "\n'
            b'            "and neither it nor supporting_evidence_quote_1 contains the literal words \'private equity\' "\n'
            b'            "(spaces or hyphens allowed). Investment firm or dedicated equity/credit funds alone does not "\n'
            b'            "meet this grounding requirement; supporting_evidence_quote_2 cannot replace that slot."\n'
            b'        )\n'
            b'    # Stage identity keys discard non-ASCII letters as well as punctuation and\n'
            b'    # spaces. Allow those characters inside the same literal identity, while\n'
            b'    # retaining outer word boundaries on the original Unicode quote.\n'
        ),
    ),
    (
        (
            b'    owner_identity_names = {\n'
            b'        _company_name(owner), re.sub(r"[^a-z0-9]+", "", owner.casefold()),\n'
            b'    }\n'
            b'    if owner_identity_names & company_names or not any(\n'
            b'        re.search(\n'
            b'            r"(?<!\\w)" + r"[^a-z0-9]*".join(re.escape(c) for c in name) + r"(?!\\w)",\n'
            b'            quote.casefold(),\n'
        ),
        (
            b'    owner_identity_names = {\n'
            b'        _company_name(owner), re.sub(r"[^a-z0-9]+", "", owner.casefold()),\n'
            b'    }\n'
            b'    if owner_identity_names & company_names:\n'
            b'        failures.append("The investigated company cannot be its own controlling owner.")\n'
            b'    if not any(\n'
            b'        re.search(\n'
            b'            r"(?<!\\w)" + r"[^a-z0-9]*".join(re.escape(c) for c in name) + r"(?!\\w)",\n'
            b'            quote.casefold(),\n'
        ),
    ),
    (
        (
            b'        for name in company_names if len(name) >= 2\n'
            b'        for quote in (control_quote, *same_page_identity_quotes)\n'
            b'    ):\n'
            b'        return False\n'
            b'\n'
            b'    owner_words = " ".join(re.findall(r"\\w+", owner.casefold()))\n'
            b'    quote_words = " ".join(re.findall(r"\\w+", control_quote.casefold()))\n'
            b'    return bool(owner_words) and f" {owner_words} " in f" {quote_words} "\n'
            b'\n'
            b'\n'
            b'def _retained_requested_source_url(\n'
        ),
        (
            b'        for name in company_names if len(name) >= 2\n'
            b'        for quote in (control_quote, *same_page_identity_quotes)\n'
            b'    ):\n'
            b'        failures.append(\n'
            b'            "The primary quote or same-page cited identity support must name the investigated company under an already established identity."\n'
            b'        )\n'
            b'\n'
            b'    owner_words = " ".join(re.findall(r"\\w+", owner.casefold()))\n'
            b'    quote_words = " ".join(re.findall(r"\\w+", control_quote.casefold()))\n'
            b'    if not (bool(owner_words) and f" {owner_words} " in f" {quote_words} "):\n'
            b'        failures.append("The primary quote must name ownership.controlling_owner as the controlling owner.")\n'
            b'    if failure_details is not None:\n'
            b'        failure_details.extend(failures)\n'
            b'    return not failures\n'
            b'\n'
            b'\n'
            b'def _retained_requested_source_url(\n'
        ),
    ),
    (
        (
            b'    priority_headquarters_url: str = "",\n'
            b'    structured_headquarters_evidence: Optional[Mapping[str, Any]] = None,\n'
            b'    classified_industry_mismatch: bool = False,\n'
            b') -> Optional[dict[str, dict[str, Any]]]:\n'
            b'    # Host-only authority from _review_industry_sources after packet resolution.\n'
            b"    # Never derive this permission from a research model's finding or receipt.\n"
            b'    if not isinstance(arguments, Mapping) or set(arguments) != {"findings"}:\n'
            b'        return None\n'
            b'    raw_findings = arguments.get("findings")\n'
        ),
        (
            b'    priority_headquarters_url: str = "",\n'
            b'    structured_headquarters_evidence: Optional[Mapping[str, Any]] = None,\n'
            b'    classified_industry_mismatch: bool = False,\n'
            b'    rejection_details: Optional[dict[str, list[str]]] = None,\n'
            b'    attribution_rejection_keys: Optional[dict[str, list[str]]] = None,\n'
            b') -> Optional[dict[str, dict[str, Any]]]:\n'
            b'    # Host-only authority from _review_industry_sources after packet resolution.\n'
            b"    # Never derive this permission from a research model's finding or receipt.\n"
            b'    if rejection_details is not None:\n'
            b'        rejection_details.clear()\n'
            b'    if attribution_rejection_keys is not None:\n'
            b'        attribution_rejection_keys.clear()\n'
            b'    if not isinstance(arguments, Mapping) or set(arguments) != {"findings"}:\n'
            b'        return None\n'
            b'    raw_findings = arguments.get("findings")\n'
        ),
    ),
    (
        (
            b'            if raw.get(f"supporting_evidence_url_{index}")\n'
            b'            or raw.get(f"supporting_evidence_quote_{index}")\n'
            b'        ]\n'
            b'        if status == "UNPROVEN":\n'
            b'            finding.update(\n'
            b'                evidence_url="",\n'
        ),
        (
            b'            if raw.get(f"supporting_evidence_url_{index}")\n'
            b'            or raw.get(f"supporting_evidence_quote_{index}")\n'
            b'        ]\n'
            b'        ownership_failures: list[str] = []\n'
            b'        if status == "UNPROVEN":\n'
            b'            finding.update(\n'
            b'                evidence_url="",\n'
        ),
    ),
    (
        (
            b'                                identity_names=tuple(stage_attribution_names),\n'
            b'                            )\n'
            b'                        ),\n'
            b'                        same_page_identity_quotes=tuple(\n'
            b'                            item["quote"] for item in linked_pe_support\n'
            b'                            if item["url"] == evidence_url\n'
        ),
        (
            b'                                identity_names=tuple(stage_attribution_names),\n'
            b'                            )\n'
            b'                        ),\n'
            b'                        failure_details=ownership_failures,\n'
            b'                        same_page_identity_quotes=tuple(\n'
            b'                            item["quote"] for item in linked_pe_support\n'
            b'                            if item["url"] == evidence_url\n'
        ),
    ),
    (
        (
            b'            else:\n'
            b'                finding["supporting_evidence"] = validated_support\n'
            b'        findings[target] = finding\n'
            b'        if target == "rebrand" and finding["status"] == "VERIFIED":\n'
            b'            acquisition_short_identity_names.update(\n'
            b'                re.sub(r"[^a-z0-9]+", "", _normalized_span(name))\n'
        ),
        (
            b'            else:\n'
            b'                finding["supporting_evidence"] = validated_support\n'
            b'        findings[target] = finding\n'
            b'        if (\n'
            b'            attribution_rejection_keys is not None\n'
            b'            and finding["status"] == "UNPROVEN"\n'
            b'            and finding["reason"] in {\n'
            b'                "source quote did not identify the investigated company",\n'
            b'                REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON,\n'
            b'            }\n'
            b'        ):\n'
            b'            attribution_rejection_keys[target] = sorted(\n'
            b'                key for key in (\n'
            b'                    stage_attribution_names if target == "stage" else attribution_names\n'
            b'                ) if len(key) >= 2\n'
            b'            )\n'
            b'        if (\n'
            b'            rejection_details is not None\n'
            b'            and target == "stage"\n'
            b'            and finding["status"] == "UNPROVEN"\n'
            b'            and finding["reason"] == "source quote did not prove current private-equity ownership"\n'
            b'            and ownership_failures\n'
            b'        ):\n'
            b'            # Host-only repair detail from the same predicate that rejected\n'
            b'            # this proof. Do not add grading fields or trust model diagnostics.\n'
            b'            rejection_details[target] = ownership_failures\n'
            b'        if target == "rebrand" and finding["status"] == "VERIFIED":\n'
            b'            acquisition_short_identity_names.update(\n'
            b'                re.sub(r"[^a-z0-9]+", "", _normalized_span(name))\n'
        ),
    ),
    (
        (
            b'                    )\n'
            b'                ):\n'
            b'                    raise ValueError("reasoning_tool_arguments_malformed")\n'
            b'                controller_payload = {\n'
            b'                    # The model assessing capability gaps also selects\n'
            b'                    # the sources and completes the final shared findings.\n'
        ),
        (
            b'                    )\n'
            b'                ):\n'
            b'                    raise ValueError("reasoning_tool_arguments_malformed")\n'
            b'                controller_messages = messages\n'
            b'                if force_submit or required_tool == "submit_findings":\n'
            b'                    last_tool = next(\n'
            b'                        (item for item in reversed(messages) if item.get("role") == "tool"),\n'
            b'                        None,\n'
            b'                    )\n'
            b'                    if last_tool:\n'
            b'                        feedback = json.loads(last_tool["content"])\n'
            b'                        if isinstance(feedback, Mapping) and feedback.get("error") == "deterministic_evidence_validation_failed":\n'
            b'                            controller_messages = list(messages)\n'
            b'                            index = next(index for index in range(len(messages) - 1, -1, -1)\n'
            b'                                         if messages[index].get("role") == "tool")\n'
            b'                            controller_messages[index] = dict(last_tool)\n'
            b'                            controller_messages[index]["content"] = _bounded_message_json(\n'
            b'                                _final_submission_feedback(feedback),\n'
            b'                            )\n'
            b'                controller_payload = {\n'
            b'                    # The model assessing capability gaps also selects\n'
            b'                    # the sources and completes the final shared findings.\n'
        ),
    ),
    (
        (
            b'                    "model": selected_controller_model,\n'
            b'                    "messages": [\n'
            b'                        *_bounded_system_messages(system_prompt),\n'
            b'                        *messages,\n'
            b'                    ],\n'
            b'                    "tools": [\n'
            b'                        {\n'
        ),
        (
            b'                    "model": selected_controller_model,\n'
            b'                    "messages": [\n'
            b'                        *_bounded_system_messages(system_prompt),\n'
            b'                        *controller_messages,\n'
            b'                    ],\n'
            b'                    "tools": [\n'
            b'                        {\n'
        ),
    ),
    (
        (
            b'                                if isinstance(raw, dict) and raw.get("target") == "industry":\n'
            b'                                    raw.update(status="UNPROVEN", evidence_url="",\n'
            b'                                               evidence_quote="", activity_role="unresolved")\n'
            b'                    claims = _validated_findings(\n'
            b'                        arguments,\n'
            b'                        targets=active_submit_targets,\n'
        ),
        (
            b'                                if isinstance(raw, dict) and raw.get("target") == "industry":\n'
            b'                                    raw.update(status="UNPROVEN", evidence_url="",\n'
            b'                                               evidence_quote="", activity_role="unresolved")\n'
            b'                    rejection_details: dict[str, list[str]] = {}\n'
            b'                    attribution_rejection_keys: dict[str, list[str]] = {}\n'
            b'                    claims = _validated_findings(\n'
            b'                        arguments,\n'
            b'                        targets=active_submit_targets,\n'
        ),
    ),
    (
        (
            b'                                bounded_homepage_navigation_locators,\n'
            b'                            )\n'
            b'                        ),\n'
            b'                    )\n'
            b'                    if claims is None:\n'
            b'                        if (\n'
        ),
        (
            b'                                bounded_homepage_navigation_locators,\n'
            b'                            )\n'
            b'                        ),\n'
            b'                        rejection_details=rejection_details,\n'
            b'                        attribution_rejection_keys=attribution_rejection_keys,\n'
            b'                    )\n'
            b'                    if claims is None:\n'
            b'                        if (\n'
        ),
    ),
    (
        (
            b'                        if not reason:\n'
            b'                            continue\n'
            b'                        rejected_finding = {"target": target, "reason": reason}\n'
            b'                        source_url = _safe_https_url(submitted.get("evidence_url"))\n'
            b'                        source_quote = str(submitted.get("evidence_quote") or "")\n'
            b'                        if (\n'
        ),
        (
            b'                        if not reason:\n'
            b'                            continue\n'
            b'                        rejected_finding = {"target": target, "reason": reason}\n'
            b'                        if attribution_rejection_keys.get(target):\n'
            b'                            rejected_finding["company_identity_keys_checked"] = attribution_rejection_keys[target]\n'
            b'                        if target == "stage" and rejection_details.get(target):\n'
            b'                            rejected_finding["ownership_proof_failures"] = rejection_details[target]\n'
            b'                        source_url = _safe_https_url(submitted.get("evidence_url"))\n'
            b'                        source_quote = str(submitted.get("evidence_quote") or "")\n'
            b'                        if (\n'
        ),
    ),
    (
        (
            b'                                    "source_context": _source_context_for_quote(support_quote, support_text),\n'
            b'                                }\n'
            b'                                if support_reason == REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON:\n'
            b'                                    candidate = _company_bound_quote_candidate(\n'
            b'                                        support_quote, support_text, identity_names,\n'
            b'                                    )\n'
        ),
        (
            b'                                    "source_context": _source_context_for_quote(support_quote, support_text),\n'
            b'                                }\n'
            b'                                if support_reason == REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON:\n'
            b'                                    support_rejection["company_identity_keys_checked"] = sorted(\n'
            b'                                        key for key in identity_names if len(key) >= 2\n'
            b'                                    )\n'
            b'                                    candidate = _company_bound_quote_candidate(\n'
            b'                                        support_quote, support_text, identity_names,\n'
            b'                                    )\n'
        ),
    ),
    (
        (
            b'                initial_industry, initial_failure, industry_review_calls = initial_result\n'
            b'                if initial_failure and initial_failure != MALFORMED_RESPONSE_FAILURE_REASON:\n'
            b'                    _record_failure(diagnostic, initial_failure)\n'
            b"                    return {'claims': {}, 'failure_reason': initial_failure}\n"
        ),
        (
            b'                initial_industry, initial_failure, industry_review_calls = initial_result\n'
            b'                if initial_failure and initial_failure != MALFORMED_RESPONSE_FAILURE_REASON:\n'
            b'                    _record_failure(diagnostic, initial_failure)\n'
            b"                    return {'claims': {}, 'failure_reason': initial_failure,\n"
            b'                            PRIVATE_FETCHED_PAGES_KEY: retained_fetched_pages()}\n'
        ),
    ),
    (
        (
            b'                reasoning_turn_limit -= deferred_calls\n'
            b"                input_document['investigation_limits']['reasoning_turns'] = reasoning_turn_limit\n"
            b'                if initial_failure and initial_failure != MALFORMED_RESPONSE_FAILURE_REASON:\n'
            b'                    _record_failure(diagnostic, initial_failure)\n'
            b"                    return {'claims': {}, 'failure_reason': initial_failure}\n"
        ),
        (
            b'                reasoning_turn_limit -= deferred_calls\n'
            b"                input_document['investigation_limits']['reasoning_turns'] = reasoning_turn_limit\n"
            b'                if initial_failure and initial_failure != MALFORMED_RESPONSE_FAILURE_REASON:\n'
            b'                    _record_failure(diagnostic, initial_failure)\n'
            b"                    return {'claims': {}, 'failure_reason': initial_failure,\n"
            b'                            PRIVATE_FETCHED_PAGES_KEY: retained_fetched_pages()}\n'
        ),
    ),
    (
        (
            b'    except (aiohttp.ClientError, RuntimeError, TimeoutError, asyncio.TimeoutError):\n'
            b'        _record_failure(diagnostic, PROVIDER_ERROR_FAILURE_REASON)\n'
            b'        result = {"claims": {}, "failure_reason": PROVIDER_ERROR_FAILURE_REASON}\n'
            b'        if local_tools.pending is not None:\n'
            b'            local_tools.emit(local_tools.pending, result,\n'
            b'                             "executed" if local_tools.pending_executed else "rejected_not_executed", terminal=True)\n'
            b'        return result\n'
        ),
        (
            b'    except (aiohttp.ClientError, RuntimeError, TimeoutError, asyncio.TimeoutError):\n'
            b'        _record_failure(diagnostic, PROVIDER_ERROR_FAILURE_REASON)\n'
            b'        result = {"claims": {}, "failure_reason": PROVIDER_ERROR_FAILURE_REASON}\n'
            b'        if local_tools.pending is not None:\n'
            b'            local_tools.emit(local_tools.pending, result,\n'
            b'                             "executed" if local_tools.pending_executed else "rejected_not_executed", terminal=True)\n'
            b'        # Retain source facts privately without adding them to tool capture.\n'
            b'        return {**result, PRIVATE_FETCHED_PAGES_KEY: retained_fetched_pages()}\n'
        ),
    ),
    (
        (
            b'    except (TypeError, ValueError, KeyError, IndexError):\n'
            b'        _record_failure(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)\n'
            b'        result = {"claims": {}, "failure_reason": MALFORMED_RESPONSE_FAILURE_REASON}\n'
            b'        if local_tools.pending is not None:\n'
            b'            local_tools.emit(local_tools.pending, result,\n'
            b'                             "executed" if local_tools.pending_executed else "rejected_not_executed", terminal=True)\n'
            b'        return result\n'
        ),
        (
            b'    except (TypeError, ValueError, KeyError, IndexError):\n'
            b'        _record_failure(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)\n'
            b'        result = {"claims": {}, "failure_reason": MALFORMED_RESPONSE_FAILURE_REASON}\n'
            b'        if local_tools.pending is not None:\n'
            b'            local_tools.emit(local_tools.pending, result,\n'
            b'                             "executed" if local_tools.pending_executed else "rejected_not_executed", terminal=True)\n'
            b'        return {**result, PRIVATE_FETCHED_PAGES_KEY: retained_fetched_pages()}\n'
        ),
    ),
    (
        (
            b'    except Exception:  # noqa: BLE001\n'
            b'        _record_failure(diagnostic, UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON)\n'
            b'        result = {"claims": {}, "failure_reason": UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON}\n'
            b'        if local_tools.pending is not None:\n'
            b'            local_tools.emit(local_tools.pending, result,\n'
            b'                             "executed" if local_tools.pending_executed else "rejected_not_executed", terminal=True)\n'
            b'        return result\n'
        ),
        (
            b'    except Exception:  # noqa: BLE001\n'
            b'        _record_failure(diagnostic, UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON)\n'
            b'        result = {"claims": {}, "failure_reason": UNEXPECTED_VERIFIER_ERROR_FAILURE_REASON}\n'
            b'        if local_tools.pending is not None:\n'
            b'            local_tools.emit(local_tools.pending, result,\n'
            b'                             "executed" if local_tools.pending_executed else "rejected_not_executed", terminal=True)\n'
            b'        return {**result, PRIVATE_FETCHED_PAGES_KEY: retained_fetched_pages()}\n'
        ),
    ),
    (
        (
            b'    _record_failure(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)\n'
            b'    return {"claims": {}, "failure_reason": MALFORMED_RESPONSE_FAILURE_REASON}\n'
        ),
        (
            b'    _record_failure(diagnostic, MALFORMED_RESPONSE_FAILURE_REASON)\n'
            b'    return {"claims": {}, "failure_reason": MALFORMED_RESPONSE_FAILURE_REASON,\n'
            b'            PRIVATE_FETCHED_PAGES_KEY: retained_fetched_pages()}\n'
        ),
    ),
)


def convert_investigator_feedback_seam(source: bytes) -> bytes:
    """Reject unknown or partial input and verify only the reviewed delta."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise ValueError("unrecognized frozen investigator feedback source")
    result = source
    for old, new in DELTAS:
        if result.count(old) != 1:
            raise ValueError("frozen investigator feedback anchor differs")
        result = result.replace(old, new, 1)
    compile(result, "frozen_investigator_feedback.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise ValueError("investigator feedback postimage differs")
    return result
