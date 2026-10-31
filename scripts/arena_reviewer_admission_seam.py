"""Install the narrow reviewer-admission interface in frozen grading source.

Only the reviewed admission blocks below can change. All prompts, model
configuration, evidence rules, counters, and other source bytes are retained.
Unknown or partial conversion shapes fail closed. A complete runtime hook may
receive only the existing final-correction eligibility binding; otherwise it
is returned byte-for-byte. There is no source-version or image-digest dispatch.
"""

from __future__ import annotations


class AdmissionSeamError(ValueError):
    """The frozen source does not have a reviewed admission shape."""


_CORRECTION_ELIGIBILITY_ARGUMENT = """                    final_correction_eligible=(
                        correction_turn and final_correction_pending
                    ),
"""


# Alternatives cover admission before and after the late-settlement repair.
# They are statement shapes, not scorer versions. Output is the same interface.
_REPLACEMENTS = (
    (
        (
            """import aiohttp

""",
            """import aiohttp

from arena_reviewer_runtime.admission import ReviewerAdmission
from arena_reviewer_runtime.transport import post_json
""",
        ),
        """import aiohttp

from arena_reviewer_runtime.admission import ReviewerAdmission
from arena_reviewer_runtime.transport import post_json
""",
    ),
    (
        (
            """            incomplete_submit_retries = 0
            incomplete_submit_retry_pending = False
""",
            """            incomplete_submit_retries = 0
            incomplete_submit_retry_pending = False
            late_final_submit_pending = bool(
                fetched_pages and (search_calls or fetch_calls)
            )
""",
            """            incomplete_submit_retries = 0
            incomplete_submit_retry_pending = False
            admission = ReviewerAdmission(
                admission_seconds=ADMISSION_DEADLINE_SECONDS,
                judgment_reserve_seconds=JUDGMENT_ADMISSION_RESERVE_SECONDS,
                serialization_grace_seconds=SERIALIZATION_REPAIR_GRACE_SECONDS,
                loaded_research=bool(fetched_pages and (search_calls or fetch_calls)),
            )
""",
        ),
        """            incomplete_submit_retries = 0
            incomplete_submit_retry_pending = False
            admission = ReviewerAdmission(
                admission_seconds=ADMISSION_DEADLINE_SECONDS,
                judgment_reserve_seconds=JUDGMENT_ADMISSION_RESERVE_SECONDS,
                serialization_grace_seconds=SERIALIZATION_REPAIR_GRACE_SECONDS,
                loaded_research=bool(fetched_pages and (search_calls or fetch_calls)),
            )
""",
    ),
    (
        (
            """                # Do not cancel a paid request after admission. Stop admitting
                # the next request when the shared per-company deadline passed,
                # except one pending serialization repair; an admitted request
                # settles under the broker's own bound.
                elapsed = time.monotonic() - started
                late_serialization_retry_pending = bool(
                    positive_semantic_review
                    and incomplete_submit_retry_pending
                    and forced_next_tool == "submit_findings"
                )
                if (
                    elapsed >= ADMISSION_DEADLINE_SECONDS
                    and not late_serialization_retry_pending
                ):
                    return interrupted_result(_turn)
                if (
                    late_serialization_retry_pending
                    and elapsed >= (
                        ADMISSION_DEADLINE_SECONDS
                        + SERIALIZATION_REPAIR_GRACE_SECONDS
                    )
                ):
                    raise ValueError("reasoning_tool_arguments_malformed")
                required_tool = forced_next_tool
                if scoped_correction_active:
                    required_tool = "submit_findings"
                judgment_reserve_active = bool(
                    elapsed >= ADMISSION_DEADLINE_SECONDS
                    - JUDGMENT_ADMISSION_RESERVE_SECONDS
                )
""",
            """                # Let admitted work settle under the broker's bound. Loaded
                # evidence may then receive one submit-only judgment; this does
                # not reopen research or add a reasoning turn or repair retry.
                elapsed = time.monotonic() - started
                late_final_submit = bool(
                    late_final_submit_pending
                    and elapsed >= ADMISSION_DEADLINE_SECONDS
                )
                late_final_submit_pending = False
                late_serialization_retry_pending = bool(
                    positive_semantic_review
                    and incomplete_submit_retry_pending
                    and forced_next_tool == "submit_findings"
                )
                if (
                    elapsed >= ADMISSION_DEADLINE_SECONDS
                    and not late_serialization_retry_pending
                    and not late_final_submit
                ):
                    return interrupted_result(_turn)
                if (
                    late_serialization_retry_pending
                    and elapsed >= (
                        ADMISSION_DEADLINE_SECONDS
                        + SERIALIZATION_REPAIR_GRACE_SECONDS
                    )
                ):
                    raise ValueError("reasoning_tool_arguments_malformed")
                required_tool = forced_next_tool
                if scoped_correction_active:
                    required_tool = "submit_findings"
                judgment_reserve_active = bool(
                    elapsed >= ADMISSION_DEADLINE_SECONDS
                    - JUDGMENT_ADMISSION_RESERVE_SECONDS
                )
""",
            """                elapsed = time.monotonic() - started
                late_serialization_retry_pending = bool(
                    positive_semantic_review
                    and incomplete_submit_retry_pending
                    and forced_next_tool == "submit_findings"
                )
                turn_admission = admission.next_turn(
                    elapsed=elapsed,
                    serialization_retry_pending=late_serialization_retry_pending,
                )
                if turn_admission.mode == "stop":
                    return interrupted_result(_turn)
                if turn_admission.serialization_expired:
                    raise ValueError("reasoning_tool_arguments_malformed")
                required_tool = forced_next_tool
                if scoped_correction_active:
                    required_tool = "submit_findings"
                judgment_reserve_active = turn_admission.mode == "submit_only"
""",
        ),
        """                elapsed = time.monotonic() - started
                late_serialization_retry_pending = bool(
                    positive_semantic_review
                    and incomplete_submit_retry_pending
                    and forced_next_tool == "submit_findings"
                )
                turn_admission = admission.next_turn(
                    elapsed=elapsed,
                    serialization_retry_pending=late_serialization_retry_pending,
                    final_correction_eligible=(
                        correction_turn and final_correction_pending
                    ),
                )
                if turn_admission.mode == "stop":
                    return interrupted_result(_turn)
                if turn_admission.serialization_expired:
                    raise ValueError("reasoning_tool_arguments_malformed")
                required_tool = forced_next_tool
                if scoped_correction_active:
                    required_tool = "submit_findings"
                judgment_reserve_active = turn_admission.mode == "submit_only"
""",
    ),
    (
        (
            """                            for tool in _tools(active_submit_targets)
                            if (
                                not correction_turn
                                and not scoped_correction_active
                            ) or tool["name"] == "submit_findings"
""",
            """                            for tool in _tools(active_submit_targets)
                            if (
                                not correction_turn
                                and not scoped_correction_active
                                and not late_final_submit
                            ) or tool["name"] == "submit_findings"
""",
            """                            for tool in _tools(active_submit_targets)
                            if (
                                not correction_turn
                                and not scoped_correction_active
                                and not turn_admission.restrict_tools
                            ) or tool["name"] == "submit_findings"
""",
        ),
        """                            for tool in _tools(active_submit_targets)
                            if (
                                not correction_turn
                                and not scoped_correction_active
                                and not turn_admission.restrict_tools
                            ) or tool["name"] == "submit_findings"
""",
    ),
    (
        (
            """                    raise ValueError("reasoning_tool_unknown")
""",
            """                    raise ValueError("reasoning_tool_unknown")
                late_final_submit_pending = bool(
                    name in {"search_web", "fetch_page"}
                    and fetched_pages
                )
""",
            """                    raise ValueError("reasoning_tool_unknown")
                admission.tool_settled(
                    name=name, has_loaded_evidence=bool(fetched_pages),
                )
""",
        ),
        """                    raise ValueError("reasoning_tool_unknown")
                admission.tool_settled(
                    name=name, has_loaded_evidence=bool(fetched_pages),
                )
""",
    ),
    (
        (
            """                elif name == "search_web":
                    elapsed = time.monotonic() - started
                    if elapsed >= ADMISSION_DEADLINE_SECONDS:
                        return interrupted_result(_turn + 1)
""",
            """                elif name == "search_web":
                    elapsed = time.monotonic() - started
""",
        ),
        """                elif name == "search_web":
                    elapsed = time.monotonic() - started
""",
    ),
    (
        (
            """                elif name == "fetch_page":
                    elapsed = time.monotonic() - started
                    if elapsed >= ADMISSION_DEADLINE_SECONDS:
                        return interrupted_result(_turn + 1)
""",
            """                elif name == "fetch_page":
                    elapsed = time.monotonic() - started
""",
        ),
        """                elif name == "fetch_page":
                    elapsed = time.monotonic() - started
""",
    ),
    (
        (
            """    async with session.post(url, headers=dict(headers), json=dict(payload)) as response:
        status = response.status
        try:
            body = await response.json()
        except (aiohttp.ContentTypeError, ValueError):
            body = None
    return status, body
""",
            """    return await post_json(session, url, headers=headers, payload=payload)
""",
        ),
        """    return await post_json(session, url, headers=headers, payload=payload)
""",
    ),
)


def convert_admission_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdmissionSeamError("grading source must be UTF-8") from exc
    if "\r" in text:
        raise AdmissionSeamError("grading source must use LF newlines")
    already_converted = (
        "from arena_reviewer_runtime.admission import ReviewerAdmission\n" in text
    )
    original = text
    for alternatives, replacement in _REPLACEMENTS:
        # Longest first: the unconverted prefix may also occur in the
        # converted block. Exactly one full matching block is required.
        matches = [old for old in (*alternatives, replacement) if old in text]
        if not matches:
            raise AdmissionSeamError("unrecognized reviewer admission block")
        old = max(matches, key=len)
        if text.count(old) != 1:
            raise AdmissionSeamError("ambiguous reviewer admission block")
        text = text.replace(old, replacement, 1)
    if (
        already_converted and text != original
        and text.replace(_CORRECTION_ELIGIBILITY_ARGUMENT, "", 1) != original
    ):
        raise AdmissionSeamError("reviewer admission interface is incomplete")
    compile(text, "company_evidence_investigator.py", "exec")
    return text.encode("utf-8")
