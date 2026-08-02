"""Behavioral tests for the fail-closed Sentry event scrubber.

Pure-stdlib tests: they must pass with or without sentry-sdk installed.
"""

from __future__ import annotations

from leadpoet_observability.sentry_scrubbing import (
    MAX_MESSAGE_LENGTH,
    REDACTED,
    REDACTED_PROTECTED,
    TRUNCATION_SUFFIX,
    event_touches_protected_surface,
    scrub_breadcrumb,
    scrub_event,
    scrub_text,
)


def _frame(module, filename="gateway/api/validate.py", **overrides):
    frame = {
        "module": module,
        "filename": filename,
        "abs_path": "/home/ec2-user/leadpoet_repo/" + filename,
        "function": "handler",
        "lineno": 10,
        "context_line": "secret_source_line()",
        "pre_context": ["before"],
        "post_context": ["after"],
        "vars": {"prompt": "SHOULD NEVER SHIP"},
    }
    frame.update(overrides)
    return frame


def _exception_event(module, value="boom", filename="gateway/api/validate.py"):
    return {
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": value,
                    "stacktrace": {"frames": [_frame(module, filename)]},
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Frame hygiene: source context and locals never leave the process
# ---------------------------------------------------------------------------


def test_source_context_and_locals_stripped_from_every_frame():
    event = scrub_event(_exception_event("gateway.api.validate"))
    frame = event["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert "context_line" not in frame
    assert "pre_context" not in frame
    assert "post_context" not in frame
    assert "vars" not in frame
    # Debugging identity survives.
    assert frame["function"] == "handler"
    assert frame["lineno"] == 10


def test_thread_frames_are_stripped_too():
    event = {
        "threads": {
            "values": [{"stacktrace": {"frames": [_frame("gateway.tasks.anchor")]}}]
        }
    }
    scrubbed = scrub_event(event)
    frame = scrubbed["threads"]["values"][0]["stacktrace"]["frames"][0]
    assert "context_line" not in frame and "vars" not in frame


def test_thread_name_mechanism_and_transaction_are_scrubbed():
    hotkey = "5" + "C" * 47
    event = _exception_event("gateway.api.validate")
    event["exception"]["values"][0]["mechanism"] = {"wallet_hotkey": hotkey}
    event["threads"] = {
        "values": [
            {
                "name": f"wallet_hotkey={hotkey}",
                "stacktrace": {"frames": [_frame("gateway.tasks.anchor")]},
            }
        ]
    }
    event["transaction"] = f"candidate/{hotkey}"
    scrubbed = scrub_event(event)
    assert hotkey not in repr(scrubbed)
    assert "transaction" not in scrubbed


# ---------------------------------------------------------------------------
# Protected surfaces: type + stack survive, messages never do
# ---------------------------------------------------------------------------


def test_protected_module_redacts_message_but_keeps_type_and_stack():
    event = scrub_event(
        _exception_event("research_lab.hosted_loop", value="prompt was: SECRET ICP")
    )
    entry = event["exception"]["values"][0]
    assert entry["value"] == REDACTED_PROTECTED
    assert entry["type"] == "ValueError"
    assert entry["stacktrace"]["frames"][0]["module"] == "research_lab.hosted_loop"


def test_protected_filename_matches_when_module_is_missing():
    event = _exception_event(
        None, value="candidate diff contents", filename="gateway/research_lab/code_loop_engine.py"
    )
    assert event_touches_protected_surface(event)
    scrubbed = scrub_event(event)
    assert scrubbed["exception"]["values"][0]["value"] == REDACTED_PROTECTED


def test_one_protected_link_redacts_the_whole_exception_chain():
    event = {
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": "inner trajectory payload",
                    "stacktrace": {"frames": [_frame("research_lab.trajectory_corpus")]},
                },
                {
                    "type": "RuntimeError",
                    "value": "wrapper embedding str(inner): inner trajectory payload",
                    "stacktrace": {"frames": [_frame("gateway.tasks.epoch_lifecycle")]},
                },
            ]
        }
    }
    scrubbed = scrub_event(event)
    values = scrubbed["exception"]["values"]
    assert all(entry["value"] == REDACTED_PROTECTED for entry in values)


def test_protected_thread_frame_marks_whole_event_protected():
    event = _exception_event("gateway.api.validate", value="looks harmless")
    event["threads"] = {
        "values": [{"stacktrace": {"frames": [_frame("miner_models.intent_model")]}}]
    }
    scrubbed = scrub_event(event)
    assert scrubbed["exception"]["values"][0]["value"] == REDACTED_PROTECTED


def test_protected_logger_redacts_log_events_and_drops_params():
    event = {
        "logger": "research_lab.observability.tracing",
        "logentry": {"message": "trace body %s", "params": ["TRAJECTORY DATA"]},
    }
    scrubbed = scrub_event(event)
    assert scrubbed["logentry"]["message"] == REDACTED_PROTECTED
    assert "params" not in scrubbed["logentry"]


def test_langfuse_and_llm_client_modules_are_protected():
    for module in ("langfuse.client", "openai._base_client", "anthropic.resources"):
        event = scrub_event(_exception_event(module, value="request echo"))
        assert event["exception"]["values"][0]["value"] == REDACTED_PROTECTED


def test_host_model_provider_and_lead_surfaces_are_protected():
    for module in (
        "gateway.tee.model_sandbox_v2",
        "gateway.tee.provider_broker_v2",
        "gateway.tee.scoring_executor_v2",
        "Leadpoet.base.validator",
        "Leadpoet.validator.reward",
    ):
        event = scrub_event(_exception_event(module, value="private model or lead body"))
        assert event["exception"]["values"][0]["value"] == REDACTED_PROTECTED


def test_extra_prefixes_widen_protection():
    event = _exception_event("myco.private_engine", value="secret sauce")
    scrubbed = scrub_event(event, extra_prefixes=("myco.private_engine",))
    assert scrubbed["exception"]["values"][0]["value"] == REDACTED_PROTECTED


def test_redact_all_mode_redacts_infra_surfaces_too():
    event = _exception_event("gateway.api.validate", value="ordinary message")
    event["breadcrumbs"] = {"values": [{"category": "gateway.db", "message": "x"}]}
    scrubbed = scrub_event(event, redact_all=True)
    assert scrubbed["exception"]["values"][0]["value"] == REDACTED_PROTECTED
    assert scrubbed["breadcrumbs"]["values"] == []


# ---------------------------------------------------------------------------
# Infra surfaces: messages survive scrubbed
# ---------------------------------------------------------------------------


def test_infra_message_scrubs_email_and_phone_but_stays_informative():
    event = scrub_event(
        _exception_event(
            "gateway.api.validate",
            value="rejected john.doe@acme.com at (415) 555-0100 retrying",
        )
    )
    value = event["exception"]["values"][0]["value"]
    assert "acme.com" not in value and "555-0100" not in value
    assert "[leadpoet-redacted:email]" in value
    assert "[leadpoet-redacted:phone]" in value
    assert "retrying" in value


def test_secret_shaped_values_are_redacted():
    for text in (
        "auth failed Bearer abcdef12345678",
        "key AKIAABCDEFGHIJKLMNOP rejected",
        "key ASIAABCDEFGHIJKLMNOP rejected",
        "supabase sb_secret_abcdefgh12345678 rejected",
        "proxy https://alice:password@example.net:444 failed",
        "jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
    ):
        assert scrub_text(text) != text


def test_wallet_material_is_redacted_from_messages():
    hotkey = "5" + "A" * 47
    scrubbed = scrub_text(f"wallet_hotkey={hotkey} coldkey=operator-vault")
    assert hotkey not in scrubbed
    assert "operator-vault" not in scrubbed


def test_secret_marker_redacts_whole_string():
    assert scrub_text("request with sk-or-v1-abc123 embedded") == REDACTED
    assert scrub_text("dump of judge_prompt follows") == REDACTED


def test_url_query_strings_are_stripped():
    scrubbed = scrub_text("GET https://x.supabase.co/rest/v1/leads?email=eq.a@b.co failed")
    assert "email=eq" not in scrubbed
    assert "?[leadpoet-redacted:query]" in scrubbed


def test_join_keys_survive_verbatim():
    for ref in (
        "execution_trace:0f8fad5b-d9cb-469f-a165-70867728950e",
        "cost_ledger:sha256:" + "ab" * 32,
        "0f8fad5bd9cb469fa16570867728950e",
    ):
        assert scrub_text(ref) == ref


def test_numeric_ranges_are_not_phone_false_positives():
    text = "allocation bytes 6553000-6553360 at block 24151"
    assert scrub_text(text) == text


def test_long_messages_are_truncated():
    scrubbed = scrub_text("x" * (MAX_MESSAGE_LENGTH * 3), MAX_MESSAGE_LENGTH)
    assert scrubbed.endswith(TRUNCATION_SUFFIX)
    assert len(scrubbed) == MAX_MESSAGE_LENGTH + len(TRUNCATION_SUFFIX)


# ---------------------------------------------------------------------------
# Envelope hygiene
# ---------------------------------------------------------------------------


def test_request_user_and_argv_are_dropped():
    event = {
        "request": {"url": "https://gw/x?secret=1", "data": "body"},
        "user": {"ip_address": "1.2.3.4"},
        "extra": {"sys.argv": ["validator.py", "--wallet_name", "w"], "note": "keep"},
        "message": "ok",
    }
    scrubbed = scrub_event(event)
    assert "request" not in scrubbed
    assert "user" not in scrubbed
    assert "sys.argv" not in scrubbed["extra"]
    assert scrubbed["extra"]["note"] == "keep"


def test_unknown_top_level_fields_are_dropped_fail_closed():
    event = {
        "message": "operational failure",
        "sdk": {"packages": [{"name": "private-model", "version": "secret"}]},
        "transaction_info": {"source": "private prompt"},
        "future_payload": "raw customer body",
    }
    scrubbed = scrub_event(event)
    assert scrubbed == {"message": "operational failure"}


def test_extra_key_based_redaction_and_scalar_preservation():
    event = {
        "extra": {
            "openrouter_api_key": "sk-live",
            "lead_email": "a@b.co",
            "page_content": "scraped html",
            "retry_count": 3,
            "epoch": 24151,
            "candidate_sha": "ab" * 20,
            "status_code": 503,
        }
    }
    scrubbed = scrub_event(event)["extra"]
    assert scrubbed["openrouter_api_key"] == REDACTED
    assert scrubbed["lead_email"] == REDACTED
    assert scrubbed["page_content"] == REDACTED
    assert scrubbed["retry_count"] == 3
    assert scrubbed["epoch"] == 24151
    assert scrubbed["candidate_sha"] == "ab" * 20
    assert scrubbed["status_code"] == 503


def test_sensitive_dynamic_mapping_keys_never_leave_event():
    hotkey = "5" + "B" * 47
    event = {
        "extra": {
            "person@example.com": "lookup failed",
            "sk-secretvalue123456789": "provider",
            hotkey: "wallet state",
            "ordinary_field": "safe",
        },
        "fingerprint": ["person@example.com", "raw"],
    }
    scrubbed = scrub_event(event)
    encoded = repr(scrubbed)
    assert "person@example.com" not in encoded
    assert "sk-secretvalue" not in encoded
    assert hotkey not in encoded
    assert "fingerprint" not in scrubbed
    assert scrubbed["extra"]["ordinary_field"] == "safe"


def test_leadpoet_tag_keys_are_not_false_positives():
    event = {"tags": {"leadpoet.component": "gateway", "leadpoet.validator.mode": "coordinator"}}
    scrubbed = scrub_event(event)["tags"]
    assert scrubbed["leadpoet.component"] == "gateway"
    assert scrubbed["leadpoet.validator.mode"] == "coordinator"


def test_arbitrary_objects_in_extra_never_serialize_their_repr():
    class Payload:
        def __repr__(self):
            return "FULL LEAD DUMP"

    scrubbed = scrub_event({"extra": {"note": Payload()}})
    assert scrubbed["extra"]["note"] == REDACTED


def test_non_dict_events_are_dropped():
    assert scrub_event(None) is None
    assert scrub_event("nonsense") is None


# ---------------------------------------------------------------------------
# Breadcrumbs
# ---------------------------------------------------------------------------


def test_breadcrumbs_from_protected_loggers_are_dropped():
    event = {
        "breadcrumbs": {
            "values": [
                {"category": "langfuse.client", "message": "trace payload"},
                {"category": "research_lab.engine_v1", "message": "prompt"},
                {
                    "category": "gateway.db.client",
                    "message": "select ok for a@b.co",
                    "data": {
                        "url": "https://x.co/rest/v1/t?apikey=1",
                        "http.query": "apikey=1",
                        "status_code": 200,
                    },
                },
            ]
        }
    }
    kept = scrub_event(event)["breadcrumbs"]["values"]
    assert len(kept) == 1
    crumb = kept[0]
    assert crumb["category"] == "gateway.db.client"
    assert "a@b.co" not in crumb["message"]
    assert "http.query" not in crumb["data"]
    assert "apikey=1" not in crumb["data"]["url"]
    assert crumb["data"]["status_code"] == 200


def test_scrub_breadcrumb_rejects_non_dicts():
    assert scrub_breadcrumb(None) is None
    assert scrub_breadcrumb("x") is None
