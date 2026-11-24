"""Private recovery of a paid synthetic generator response before grading."""

from __future__ import annotations

import hashlib
import json

from lab_arena.collection_checkpoint import require_private_evidence


CAPTURE_V2 = "collection.workflow-events.v2"
MAX_RUN_TOKENS = 90 * 8192


class OutputLimitReached(ValueError):
    """No output allowance remains for another paid model call."""


class QuarantinedWorkflow(ValueError):
    """Prohibited material was withheld from trace storage."""


def encoded(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def output_charge(usage, maximum):
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return count if type(count) is int and 0 <= count <= maximum else None


def visible_reply(reply):
    out = {"role": reply["role"], "content": reply.get("content") or ""}
    for field in ("tool_calls", "reasoning", "reasoning_content"):
        if reply.get(field):
            out[field] = reply[field]
    return out


class GenerationCapture:
    def __init__(self, run_id, builders, save):
        self.run_id = run_id
        self.builders = builders
        self.save = save
        self.sent = []
        self.calls = []
        self.last_request = None
        self.last_reply = None
        self.messages = []
        self.tools = []
        self.before = None
        self.index = 0
        self.archived_run = False
        self.raw_tool_bytes = 0
        self.tool_outputs = {}
        self.wire_refs = {}
        self.opaque_sidecars = 0
        self.visible_training_safe = True

    def check(self, value, stage):
        try:
            if stage in {"workflow-output", "generator", "event-start"}:
                require_private_evidence(
                    {k: v for k, v in value.items() if k != "messages"}
                )
                require_private_evidence(
                    {
                        "operation_id": "openrouter.chat",
                        "effective_parameters": {"messages": value["messages"]},
                    }
                )
            elif stage == "event-reply":
                require_private_evidence(
                    {k: v for k, v in value.items() if k != "message"}
                )
                require_private_evidence({"choices": [{"message": value["message"]}]})
            else:
                require_private_evidence(value)
            if self.builders.KEYS.search(encoded(value).decode()):
                raise ValueError("credential in generator content")
        except (TypeError, UnicodeError, ValueError):
            if self.save is not None:
                marker = {"run_id": self.run_id, "stage": stage}
                try:
                    raw = encoded(value)
                except (TypeError, UnicodeError, ValueError):
                    marker["encoding_failed"] = True
                else:
                    marker.update(
                        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
                    )
                self.save("quarantine", marker)
            raise QuarantinedWorkflow(
                "sensitive or opaque synthetic content withheld"
            ) from None

    def permitted_response(self, response):
        """Allow returned plaintext while excluding provider-only opaque sidecars."""
        from lab_arena.training_capture import sanitize_training_value

        safe, reasons = sanitize_training_value(response, aggregate_parts=True)
        if set(reasons) == {"hidden_reasoning"} and not self.builders.KEYS.search(
            encoded(response).decode()
        ):
            # Sanitizer still checks the entire response for credential leaks. An
            # opaque field in the *visible* reply is checked by observe() as well.
            return safe, True
        self.check(response, "provider-reply")
        return response, False

    def wire_response(
        self, request_id, request_hash, response_hash, status, cost, generation, body
    ):
        """Write the broker-verified original bytes before any provider normalization."""
        doc = {
            "run_id": self.run_id,
            "request_id": request_id,
            "request_sha256": request_hash,
            "response_sha256": response_hash,
            "http_status": status,
            "miner_cost_microusd": cost,
            "provider_generation_id": generation,
        }
        self.check(doc, "provider-metadata")
        if body is None:
            self.save("quarantine", dict(doc, stage="unverified-provider-response"))
            return
        permitted, withheld = self.permitted_response(json.loads(body))
        if withheld:
            doc.update(opaque_fields_withheld=True, permitted_response=permitted)
        else:
            doc["body_utf8"] = body.decode()
        self.wire_refs[request_id] = self.save(f"wire-{request_id}", doc)

    def returned(self, request_id, request, reply, allowance):
        # Only results returned by the origin/route/receipt-verifying broker get here.
        permitted, withheld = self.permitted_response(reply)
        self.opaque_sidecars += int(withheld)
        self.last_request, self.last_reply = request, permitted
        usage = reply.get("usage")
        choices = reply.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            else {}
        )
        self.calls.append(
            (output_charge(usage, allowance), allowance, choice.get("finish_reason"))
        )
        if self.save is not None:
            msgs = request["messages"]
            start = len(self.sent) if msgs[: len(self.sent)] == self.sent else 0
            packet = {
                "run_id": self.run_id,
                "request_id": request_id,
                "messages_from": start,
                "messages": msgs[start:],
                "request_sha256": hashlib.sha256(encoded(request)).hexdigest(),
                "request_options": {
                    k: v for k, v in request.items() if k != "messages"
                },
            }
            if request_id in self.wire_refs:
                packet["response_ref"] = self.wire_refs[request_id]
            else:
                packet["response"] = permitted
                packet["opaque_fields_withheld"] = withheld
            self.check(
                {
                    "operation_id": "openrouter.chat",
                    "effective_parameters": {
                        **packet["request_options"],
                        "messages": packet["messages"],
                    },
                },
                "provider-request",
            )
            self.save(f"provider-{len(self.calls):04d}", packet)
            self.sent = list(msgs)

    def observe(self, event):
        kind = event["kind"]
        self.index = event["index"] + 1
        if kind == "start":
            self.messages = list(event["messages"])
            self.tools = event["tools"]
        elif kind == "before":
            self.before = event
        elif kind == "reply":
            self.messages.append(visible_reply(event["message"]))
        elif kind == "tool":
            raw = event["output"]
            self.raw_tool_bytes += len(raw.encode())
            self.tool_outputs[event["call_id"]] = raw
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event["call_id"],
                    "content": raw[:20000],
                }
            )

    def partial(self, task, world, model, reason):
        # A truncated response can contain a partial tool call. It is recorded,
        # never executed. The last real checkpoint supplies the observed file state.
        before = self.before or {}
        files = before.get("files", task["files"])
        changed = {p: v for p, v in files.items() if task["files"].get(p) != v}
        messages = self.messages
        tools = self.tools
        if not tools and self.last_request is not None:
            messages = list(self.last_request["messages"])
            tools = self.last_request["tools"]
            choices = self.last_reply.get("choices") or []
            if (
                len(choices) == 1
                and isinstance(choices[0], dict)
                and isinstance(choices[0].get("message"), dict)
                and choices[0]["message"].get("role") == "assistant"
            ):
                messages.append(visible_reply(choices[0]["message"]))
        return {
            "finish": reason,
            "model": model,
            "messages": messages,
            "tools": tools,
            "out": {"files": changed, "state": world.s, "log": world.log, "answer": ""},
        }

    def summary(self, run):
        counts = [n for n, _, _ in self.calls if n is not None]
        msgs = run["messages"]
        return {
            "finish_reason": run["finish"],
            "assistant_turns": sum(m["role"] == "assistant" for m in msgs),
            "final_turn_present": bool(
                run["finish"] == "answered" and msgs and msgs[-1]["role"] == "assistant"
            ),
            "completion_tokens": (
                sum(counts) if len(counts) == len(self.calls) else None
            ),
            "known_completion_tokens": sum(counts),
            "unknown_completion_calls": len(self.calls) - len(counts),
            "response_finish_reasons": [v for _, _, v in self.calls],
            "last_provider_finish_reason": self.calls[-1][2] if self.calls else None,
            "generated_calls": len(self.calls),
            "opaque_sidecars_withheld": self.opaque_sidecars,
            "visible_training_safe": self.visible_training_safe,
            "output_token_upper_bound": sum(
                k if k is not None else cap for k, cap, _ in self.calls
            ),
            "message_bytes": len(encoded(msgs)),
            "reasoning_bytes": sum(
                len((m.get("reasoning") or m.get("reasoning_content") or "").encode())
                for m in msgs
            ),
            "tool_result_bytes": (
                self.raw_tool_bytes
                if self.tools
                else sum(
                    len(m["content"].encode()) for m in msgs if m["role"] == "tool"
                )
            ),
        }

    def archive_run(self, run, task, job):
        if self.archived_run:
            return
        doc = {
            "schema": "collection.private-workflow.v1",
            "run_id": self.run_id,
            "scenario_id": job["scenario_id"],
            "harness_commit": job.get("harness_commit"),
            "builder_commit": getattr(self.builders, "source_revision", None),
            "initial_files": task["files"],
            "initial_state": task.get("state"),
            "generation": self.summary(run),
            **run,
        }
        self.check(doc, "workflow-output")
        if self.save is not None:
            self.save("workflow-output", doc)
        self.archived_run = True
