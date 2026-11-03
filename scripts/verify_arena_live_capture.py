#!/usr/bin/env python3
"""Bounded, local Arena capture probe. Never writes to production.

One exact OpenRouter route, three sequential synthetic turns. The real Broker,
HTTP transport, request capture, response receipt, and training-export decoder
are exercised against disposable PostgreSQL by default. A local file ledger is
available for diagnostics. Optional worker-socket mode verifies delivery through
the existing worker transport. This is not a complete scored ICP or production
round. Candidate route files are admitted only in this process.
"""
from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import replace
from decimal import Decimal, ROUND_UP
import fcntl
import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext
import shutil
import tempfile
from unittest.mock import patch

from lab_arena import broker as br, contracts, lab_arena_checkpoint, operations, reasoning_capture as rc, runner, shim, training_capture, training_observations, training_thinking
from lab_arena.service import LocalObjectStore
from lab_arena.store import ArenaStore, PsycopgTransport
from lab_arena.training_export import _calls
from scripts.probe_full_thinking_routes import INITIAL, TOOLS, SCHEMA, EVIDENCE, assistant_history
from tests.lab_arena.test_lab_arena_broker import FakeLedgerStore, CONTEXT

BUDGET_USD = Decimal("8")
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def scenario_spec(name: str, *, final_format: str = "json_schema"):
    if name == "acme":
        spec = {"initial": copy.deepcopy(INITIAL), "evidence": copy.deepcopy(EVIDENCE),
                "expected": {"company": "Acme Test", "qualified": True,
                             "evidence_ids": ["profile", "headcount"]}}
    elif name == "conflicting_evidence":
        spec = {"initial": [{"role": "user", "content": (
            "Synthetic compatibility test. Decide whether Acme Test is a US B2B software "
            "company with 50-200 current employees. First call lookup_evidence for 'profile', "
            "then for 'headcount', one call per turn. The older profile and the newer official "
            "headcount disagree; use the newer official headcount for the employee criterion. "
            "After both tools return only strict JSON with company, qualified, evidence_ids. "
            "Use both sources and do not infer missing facts.")}],
                "evidence": {
                    "profile": {"evidence_id": "profile", "company": "Acme Test",
                                "country": "US", "business": "B2B software",
                                "employees": 80, "observed_on": "2024-01-01"},
                    "headcount": {"evidence_id": "headcount", "employees": 12,
                                  "observed_on": "2026-10-23", "source": "current official staffing"}},
                "expected": {"company": "Acme Test", "qualified": False,
                             "evidence_ids": ["profile", "headcount"]}}
    else:
        raise ValueError("unknown_probe_scenario")
    if final_format == "text":
        # Add the contract once before the first request. All subsequent
        # requests retain this exact initial message with their tool history.
        spec["initial"][0]["content"] += ("\nAfter both tool results, use this final output contract. "
                                         "Return only JSON conforming to this schema: "
                                         + json.dumps(SCHEMA, sort_keys=True, separators=(",", ":")))
    return spec


def _private_write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


class SharedBudget:
    """Reserve uncertain attempts across separate route processes, fail closed."""

    def __init__(self, path: Path, cap: Decimal):
        self.path = path
        self.cap = cap
        if not path.exists():
            try:
                _private_write(path, _json_bytes({"cap_usd": str(cap), "spent_usd": "0",
                                                  "reserved_usd": "0"}))
            except FileExistsError:
                pass
        if path.stat().st_mode & 0o077:
            raise ValueError("budget_state_permissions_not_private")

    def _change(self, reserve: Decimal = Decimal(0), settle: Decimal = Decimal(0)):
        with self.path.open("r+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            state = json.load(stream)
            if Decimal(state["cap_usd"]) != self.cap:
                raise ValueError("budget_state_cap_mismatch")
            spent, held = Decimal(state["spent_usd"]), Decimal(state["reserved_usd"])
            if reserve:
                if reserve <= 0 or spent + held + reserve > self.cap:
                    raise ValueError("cumulative_budget_exceeded")
                held += reserve
            if settle:
                if settle < 0 or settle > held:
                    raise ValueError("budget_settlement_invalid")
                held -= settle
                spent += settle
            state.update(spent_usd=str(spent), reserved_usd=str(held))
            stream.seek(0)
            json.dump(state, stream, sort_keys=True)
            stream.write("\n")
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return state

    def reserve(self, amount: Decimal):
        return self._change(reserve=amount)

    def settle(self, reserved: Decimal, actual: Decimal):
        if actual < 0 or actual > reserved:
            raise ValueError("provider_cost_outside_reserved_bound")
        with self.path.open("r+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            state = json.load(stream)
            if Decimal(state["cap_usd"]) != self.cap:
                raise ValueError("budget_state_cap_mismatch")
            held = Decimal(state["reserved_usd"])
            if held < reserved:
                raise ValueError("budget_reservation_missing")
            state.update(spent_usd=str(Decimal(state["spent_usd"]) + actual),
                         reserved_usd=str(held - reserved))
            stream.seek(0)
            json.dump(state, stream, sort_keys=True)
            stream.write("\n")
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class AuditStore(FakeLedgerStore):
    """Disposable admission simulation; journal exact broker method arguments/results."""

    def __init__(self, path: Path, *, budget_microusd: int):
        super().__init__(openrouter_capacity=budget_microusd)
        self.path = path
        self.next_id = 1
        _private_write(path, b"")

    def _journal(self, method, arguments, result):
        item = {"entry_id": self.next_id, "method": method,
                "arguments": arguments, "result": result}
        self.next_id += 1
        raw = (json.dumps(item, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode()
        with self.path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return result

    def reserve_call(self, **kwargs):
        return self._journal("reserve_call", kwargs, super().reserve_call(**kwargs))

    def mark_dispatched(self, **kwargs):
        return self._journal("mark_dispatched", kwargs, super().mark_dispatched(**kwargs))

    def settle_call(self, **kwargs):
        return self._journal("settle_call", kwargs, super().settle_call(**kwargs))

    def mark_uncertain(self, **kwargs):
        return self._journal("mark_uncertain", kwargs, super().mark_uncertain(**kwargs))


class DisposablePostgres:
    """Own the existing exact migration fixture and read its real SQL ledger."""

    def __init__(self):
        self.generator = None
        self.transport = None
        self.store = None
        self.context = None
        self.run = None

    def open(self, registry):
        from tests.lab_arena import full_thinking_six_dollar_postgres_test as fixture
        self.generator = fixture.database.__wrapped__()
        database = next(self.generator)
        psycopg, dsn = database
        with psycopg.connect(**dsn) as connection:
            fixture.seed(connection, full=True)
            lease = fixture.prior.claim(connection)
            if lease.get("status") != "leased":
                raise ValueError("disposable_lease_unavailable")
            with connection.cursor() as cursor:
                cursor.execute("SELECT run_id,round_id,assignment_id,icp_position,"
                               "miner_hotkey,submission_id,stage,attempt "
                               "FROM public.lab_arena_runs WHERE run_id=%s", (lease["run_id"],))
                row = cursor.fetchone()
        if row is None:
            raise ValueError("disposable_run_missing")
        self.context = br.RunContext(
            run_id=row[0], round_id=row[1], assignment_id=row[2], icp_position=row[3],
            lease_token_hash="sha256:" + "a" * 64, miner_hotkey=row[4],
            submission_id=row[5], stage=row[6], attempt=row[7],
            reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry)
        self.transport = PsycopgTransport(lambda: psycopg.connect(**dsn), pool_size=1)
        self.store = ArenaStore(self.transport)
        self.run = self.store.get_run(self.context.run_id)
        return self.store, self.context

    def rows(self):
        return self.store.list_ledger(run_id=self.context.run_id, limit=1000)

    def close(self):
        if self.transport is not None:
            self.transport.close()
        if self.generator is not None:
            self.generator.close()


def ledger_rows(path: Path, *, context, operation_id="openrouter.chat") -> list[dict]:
    """Reopen journal and project only observed methods; no receipt synthesis."""
    rows = []
    for line in path.read_bytes().splitlines():
        item = json.loads(line)
        method, arguments, result = item["method"], item["arguments"], item["result"]
        kind = {"reserve_call": "reservation", "mark_dispatched": "dispatch",
                "settle_call": "settlement", "mark_uncertain": "uncertain"}[method]
        if result.get("status") != {"reservation": "reserved", "dispatch": "dispatched",
                                    "settlement": "settled", "uncertain": "uncertain"}[kind]:
            continue
        row = {"entry_id": item["entry_id"], "entry_kind": kind,
               "run_id": arguments["run_id"], "round_id": context.round_id,
               "submission_id": context.submission_id,
               "call_identity": arguments["call_identity"],
               "provider": "openrouter", "operation_id": operation_id,
               "funding_source": "host"}
        if kind == "reservation":
            row.update(amount_microusd=arguments["amount_microusd"],
                       entry_doc=arguments["call_doc"])
        elif kind == "settlement":
            row.update(amount_microusd=arguments["actual_microusd"],
                       terminal_response=arguments["terminal_response"])
        elif kind == "uncertain":
            row.update(amount_microusd=0, entry_doc=arguments["call_doc"])
        else:
            row["amount_microusd"] = 0
        rows.append(row)
    return rows


class WireTransport:
    """Record the exact bounded provider bytes independently of Broker storage."""

    def __init__(self, root: Path, inner=None, *, secret: str = "", operation_id="openrouter.chat",
                 request_budget=None, budget_route=None):
        self.root = root
        self.inner = inner or br.HttpxProviderTransport(max_response_bytes=MAX_RESPONSE_BYTES)
        self.secret = secret.encode()
        self.operation_id = operation_id
        self.attempt = 0
        self.records = []
        self.armed_turn = None
        self.sent_in_turn = False
        self.request_budget = request_budget
        self.budget_route = budget_route
        self.request_reservations = {}
        self.settled_turns = set()

    def arm(self, turn: int):
        self.armed_turn = turn
        self.sent_in_turn = False

    def send(self, **kwargs):
        expected = ("https://openrouter.ai/api/v1/chat/completions" if self.operation_id == "openrouter.chat"
                    else "https://openrouter.ai/api/v1/responses")
        if kwargs.get("url") != expected or self.armed_turn is None or self.sent_in_turn:
            raise br.ProviderTransportError("probe_additional_http_blocked")
        if self.request_budget is not None:
            amount = request_bound(self.budget_route, kwargs["body"])
            self.request_budget.reserve(amount)
            self.request_reservations[self.armed_turn] = amount
        self.sent_in_turn = True
        self.attempt += 1
        response = self.inner.send(**kwargs)
        if self.secret and self.secret in response.body:
            raise br.ProviderTransportError("credential_echo")
        # The transport's redirect/size guards can substitute a generic body.
        # Such a result cannot be called an exact upstream wire receipt.
        if response.internal_provenance:
            self.records.append({"status": response.status, "transport_error": response.internal_provenance})
            return response
        name = f"wire-response-{self.attempt}.bin"
        _private_write(self.root / name, response.body)
        self.records.append({"status": response.status, "path": name,
                             "outbound_body_sha256": contracts.hash_bytes(kwargs["body"]),
                             "sha256": contracts.hash_bytes(response.body),
                             "bytes": len(response.body)})
        return response

    def close(self):
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


class BrokerSocketApi:
    """Adapter at the worker's real provider API boundary, not a fake reply."""

    def __init__(self, broker, context):
        self.broker, self.context = broker, context
        self.results = []

    def provider(self, run_id, lease_token, frame):
        if run_id != self.context.run_id:
            raise ValueError("socket_run_mismatch")
        result = self.broker.execute(
            self.context, operation_id=frame["operation_id"],
            parameters=frame["parameters"], action_sequence=frame["action_sequence"],
            timeout_ms=frame["timeout_ms"])
        self.results.append(result)
        return result.to_document()


def route_registry(route_file: Path | None, model: str):
    working_registry = rc.load_registry()
    if route_file is None:
        route = next((r for r in working_registry["approved_routes"] if r["model_id"] == model), None)
        if route is None:
            raise ValueError("route_not_in_working_registry; supply --candidate-route")
        return working_registry, route, "working_registry"
    candidate = json.loads(route_file.read_text())
    if isinstance(candidate, dict) and "approved_routes" in candidate:
        routes = candidate.get("approved_routes")
        if not isinstance(routes, list) or len(routes) != 1:
            raise ValueError("candidate_registry_must_have_one_route")
        candidate = routes[0]
    if not isinstance(candidate, dict) or candidate.get("model_id") != model:
        raise ValueError("candidate_model_mismatch")
    if candidate.get("api") not in {"openrouter.chat", "openrouter.responses"}:
        raise ValueError("candidate_api_unsupported")
    # The production validator has no probe-only status. This isolated copy
    # permits the exact broker path without modifying the committed whitelist.
    candidate = dict(candidate, status="approved")
    if "evidence" not in candidate:
        candidate["evidence"] = {"documentation_url": "https://openrouter.ai/",
                                 "probe_id": "isolated-candidate-probe",
                                 "verified_at": "2026-10-23T00:00:00Z",
                                 "probe_manifest_sha256": "0" * 64}
    isolated = {"schema_version": 1, "approved_routes": [candidate], "baseline_route": None}
    return rc.validate_registry(isolated), candidate, "isolated_candidate_probe_only"


def conservative_bound(route) -> Decimal:
    """Reserve the whole context plus output at the route ceiling per attempt."""
    price = route["price_ceiling"]
    bound = (Decimal(route["context_tokens"]) * Decimal(price["prompt"])
             + Decimal(route["max_output_tokens"]) *
             (Decimal(price["completion"]) + Decimal(price["internal_reasoning"]))
             + Decimal(price["request"]))
    return bound.quantize(Decimal("0.000001"), rounding=ROUND_UP)


def request_bound(route, body: bytes) -> Decimal:
    """Bound the exact outbound UTF-8 request, using the Broker's byte ceiling.

    The full tested output cap remains reserved. No context metadata is reduced.
    This synthetic probe has no images, hosted search or other priced tools.
    """
    document = json.loads(body)
    output_field = "max_output_tokens" if route["api"] == "openrouter.responses" else "max_tokens"
    if (document.get("model") != route["model_id"]
            or type(document.get(output_field)) is not int
            or document[output_field] > route["max_output_tokens"]
            or document[output_field] <= 0
            or any(item.get("type") != "function" for item in document.get("tools", []))):
        raise ValueError("request_budget_shape_invalid")
    price = route["price_ceiling"]
    amount = (Decimal(br.bounded_input_tokens(document)) * Decimal(price["prompt"])
              + Decimal(document[output_field]) *
              (Decimal(price["completion"]) + Decimal(price["internal_reasoning"]))
              + Decimal(price["request"]))
    return amount.quantize(Decimal("0.000001"), rounding=ROUND_UP)


def _price_table(route):
    return br.validate_price_table({"schema_version": br.PRICE_TABLE_SCHEMA_VERSION,
                                    "fetched_at": "2026-10-23T00:00:00Z",
                                    "source": br.OPENROUTER_MODELS_URL,
                                    "models": {route["model_id"]: {
                                        key: route["price_ceiling"].get(key, "0")
                                        for key in br.PRICED_COMPONENTS}}})


def _diagnostic_host_route(route, *, responses_compat=False):
    """Pin a probe-only route without assigning full-thinking policy evidence."""
    ceiling = route["price_ceiling"]
    price = {key: Decimal(ceiling.get(key, "0")) for key in br.PRICED_COMPONENTS}
    return br.OpenRouterHostRoute(
        provider_policy={"data_collection": "deny", "zdr": True,
                         "only": [route["provider"]], "allow_fallbacks": False,
                         "require_parameters": not responses_compat,
                         "max_price": {
                             "prompt": float(price["prompt"] * br.OPENROUTER_PRICE_PER_MILLION),
                             "completion": float((price["completion"] + price["internal_reasoning"])
                                                 * br.OPENROUTER_PRICE_PER_MILLION),
                             "request": float(price["request"])}},
        reservation_pricing={key: ceiling.get(key, "0") for key in br.PRICED_COMPONENTS})


def _parameters(model, messages, turn, operation_id, mode, route, *,
                diagnostic_tool_choice_auto=False, diagnostic_responses_compat=False,
                tool_choice="named", final_format="json_schema", parallel_tool_calls=None):
    # A full probe explicitly exercises its declared settings. The production
    # capture policy must not inflate this caller's cap or reasoning effort.
    output_limit = int(route["max_output_tokens"]) if mode == "full" else min(2048, int(route["max_output_tokens"]))
    if operation_id == "openrouter.responses":
        value = {"model": model, "input": copy.deepcopy(messages),
                 "max_output_tokens": output_limit}
        if mode == "full":
            value["reasoning"] = copy.deepcopy(route["mode"])
        if mode == "summary_diagnostic":
            value["reasoning"] = {"effort": route["mode"].get("effort", "high"),
                                  "summary": "auto" if diagnostic_responses_compat else "detailed"}
        if turn < 2:
            value.update(tools=[{"type": "function", "name": "lookup_evidence",
                                 "description": "Read synthetic company evidence by ID.",
                                 "parameters": TOOLS[0]["function"]["parameters"]}],
                         tool_choice="required" if tool_choice == "named" else tool_choice,
                         )
            if parallel_tool_calls is not None:
                value["parallel_tool_calls"] = parallel_tool_calls
        else:
            value["text"] = {"format": ({"type": "json_schema", "name": "qualification",
                                         "strict": True, "schema": SCHEMA} if final_format == "json_schema"
                                        else {"type": final_format})}
        return value
    value = {"model": model, "messages": copy.deepcopy(messages),
             "max_tokens": output_limit}
    if mode == "full":
        value["reasoning"] = {**({"max_tokens": route["mode"]["budget_tokens"]}
                                  if "budget_tokens" in route["mode"] else route["mode"]),
                              "exclude": False}
    if mode == "summary_diagnostic":
        value["reasoning"] = {**route["mode"], "exclude": False}
    if turn < 2:
        choice = ("auto" if diagnostic_tool_choice_auto else
                  {"type": "function", "function": {"name": "lookup_evidence"}}
                  if tool_choice == "named" else tool_choice)
        value.update(tools=TOOLS, tool_choice=choice)
        if parallel_tool_calls is not None:
            value["parallel_tool_calls"] = parallel_tool_calls
    elif final_format != "text":
        value["response_format"] = ({"type": "json_schema", "json_schema": {
            "name": "qualification", "strict": True, "schema": SCHEMA}}
            if final_format == "json_schema" else {"type": "json_object"})
    return value


def preflight_probe_controls(*, model, route, mode="full", scenario="acme",
                             tool_choice="named", final_format="json_schema",
                             parallel_tool_calls=None,
                             diagnostic_tool_choice_auto=False,
                             diagnostic_responses_compat=False):
    """Reject unsupported probe API/settings before any provider attempt.

    Initial and final shapes use the same production validator as execution.
    Future provider-returned histories still need their normal per-call checks.
    """
    if (tool_choice not in {"named", "auto", "required"}
            or final_format not in {"json_schema", "json_object", "text"}
            or parallel_tool_calls is not None and type(parallel_tool_calls) is not bool):
        raise ValueError("invalid_probe_controls")
    spec = scenario_spec(scenario, final_format=final_format)
    registry = {"schema_version": 1, "approved_routes": [route], "baseline_route": None}
    for turn in range(3):
        parameters = _parameters(model, spec["initial"], turn, route["api"], mode, route,
                                 tool_choice=tool_choice, final_format=final_format,
                                 parallel_tool_calls=parallel_tool_calls,
                                 diagnostic_tool_choice_auto=diagnostic_tool_choice_auto,
                                 diagnostic_responses_compat=diagnostic_responses_compat)
        if mode == "full":
            br.agent_reasoning_parameters(route["api"], parameters, policy=rc.POLICY,
                                          registry=registry)
        else:
            operations.validate_operation_request(route["api"], parameters, allow_full_thinking=True)


def _next_message(body, turn, operation_id, spec):
    response = json.loads(body)
    if operation_id == "openrouter.responses":
        output = response.get("output") or []
        if turn < 2:
            calls = [item for item in output if item.get("type") == "function_call"]
            if len(calls) != 1:
                raise ValueError("expected_one_tool_call")
            call = calls[0]
            expected = ("profile", "headcount")[turn]
            if call.get("name") != "lookup_evidence" or json.loads(
                    call.get("arguments", "null")) != {"evidence_id": expected}:
                raise ValueError("tool_arguments_invalid")
            return response, [*copy.deepcopy(output), {"type": "function_call_output",
                               "call_id": call["call_id"],
                               "output": json.dumps(spec["evidence"][expected], separators=(",", ":"))}]
        texts = [part.get("text") for item in output if item.get("type") == "message"
                 for part in item.get("content") or [] if part.get("type") == "output_text"]
        final = json.loads(texts[0]) if len(texts) == 1 else None
        if not _valid_final(final, spec["expected"]):
            raise ValueError("strict_final_json_mismatch")
        return response, []
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if turn < 2:
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError("expected_one_tool_call")
        call = calls[0]
        expected = ("profile", "headcount")[turn]
        if call.get("function", {}).get("name") != "lookup_evidence" or json.loads(
                call.get("function", {}).get("arguments", "null")) != {"evidence_id": expected}:
            raise ValueError("tool_arguments_invalid")
        return response, [assistant_history(message), {"role": "tool",
                          "tool_call_id": call["id"],
                          "content": json.dumps(spec["evidence"][expected], separators=(",", ":"))}]
    final = json.loads(message.get("content") or "null")
    if not _valid_final(final, spec["expected"]):
        raise ValueError("strict_final_json_mismatch")
    return response, []


def _legacy_chat_diagnostic_continuation(continuation):
    """Keep tool history while using the policy-off Chat request shape."""
    projected = copy.deepcopy(continuation)
    for message in projected:
        if message.get("role") == "assistant":
            for field in ("reasoning", "reasoning_content", "reasoning_details", "refusal"):
                message.pop(field, None)
            if message.get("content") is None:
                message["content"] = ""
            for call in message.get("tool_calls") or []:
                call.pop("index", None)
    return projected


def _valid_final(value, expected):
    return (isinstance(value, dict) and set(value) == set(expected)
            and value["company"] == expected["company"]
            and type(value["qualified"]) is bool
            and value["qualified"] is expected["qualified"]
            and isinstance(value["evidence_ids"], list)
            and len(value["evidence_ids"]) == 2
            and all(isinstance(item, str) for item in value["evidence_ids"])
            and set(value["evidence_ids"]) == set(expected["evidence_ids"]))


def verify_storage(root: Path, context, registry, wire_records, *, rows=None,
                   operation_id="openrouter.chat", worker_run=None,
                   worker_snapshot=None) -> list[dict]:
    # New instances prove disk readback, and _calls is the real training export
    # ledger/request/response decoder. It cannot prove worker delivery.
    objects = LocalObjectStore(root / "objects")
    rows = rows if rows is not None else ledger_rows(root / "ledger.jsonl", context=context,
                                                     operation_id=operation_id)
    calls, gaps = _calls(rows, {context.run_id}, objects, {context.run_id: 0})
    checks = []
    for index, call in enumerate(calls):
        wire = wire_records[index]
        if "path" not in wire:
            raise ValueError("transport_substituted_response")
        raw = (root / wire["path"]).read_bytes()
        upstream = [item for item in call.get("upstream_response_receipts") or []
                    if item.get("wire_body_sha256") == wire["sha256"]
                    and item.get("wire_body_bytes") == wire["bytes"]
                    and item.get("status") == wire["status"]]
        terminal = next((r for r in rows if r["entry_kind"] == "settlement"
                         and r["call_identity"] == call["call_identity"]), None)
        receipt = terminal.get("terminal_response") or {} if terminal else {}
        from lab_arena.training_export import _response_value
        try:
            sanitized, redactions = _response_value(receipt, "openrouter")
        except Exception:
            sanitized, redactions = None, []
        readable = sanitized is not None and call.get("response") == sanitized
        terminal_exact = False
        if isinstance(receipt.get("body_b64"), str):
            decoded = base64.b64decode(receipt["body_b64"], validate=True)
            terminal_exact = contracts.hash_bytes(decoded) == wire["sha256"]
        exact = (contracts.hash_bytes(raw) == wire["sha256"]
                 and (len(upstream) == 1 or terminal_exact))
        outbound = ((call.get("request") or {}).get("outbound") or {})
        saved_outbound_hash = outbound.get("body_sha256")
        coverage = training_thinking.attempt_coverage(call, "miner", registry)
        checks.append({"call_identity": call["call_identity"],
                       "wire_receipt_exact": exact,
                       "outbound_request_exact": saved_outbound_hash == wire["outbound_body_sha256"],
                       "stored_request_readable": isinstance(call.get("request"), dict),
                       "stored_response_readable": readable,
                       "stored_provider_response_readable": bool(upstream and isinstance(upstream[0].get("response"), dict)),
                       "provider_response_status": wire["status"],
                       "accounting_state": call.get("accounting_state"),
                       "selected_terminal_differs_from_provider": bool(terminal and not terminal_exact),
                       "response_privacy_redactions": redactions,
                       "capture": coverage,
                       "evidence_gaps": call.get("evidence_gaps", [])})
    if len(calls) != len(wire_records):
        raise ValueError("ledger_wire_count_mismatch")
    def complete_readback(row):
        required = ["wire_receipt_exact", "outbound_request_exact", "stored_request_readable"]
        if row["accounting_state"] == "settlement" and row["provider_response_status"] == 200:
            required.append("stored_response_readable")
        if row["accounting_state"] == "uncertain":
            required.append("stored_provider_response_readable")
        return all(row[key] for key in required)

    if any(not complete_readback(row) for row in checks):
        raise ValueError("durable_readback_mismatch")
    worker_gaps = []
    if worker_snapshot is not None:
        from lab_arena.training_export import _link_worker_receipts
        worker_gaps = sorted(_link_worker_receipts(worker_run, calls, worker_snapshot))
        if any(gap != "agent_task_state_unverified" for gap in worker_gaps):
            raise ValueError("worker_receipt_link_invalid:" + ",".join(worker_gaps))
        for check, call in zip(checks, calls):
            provenance = call.get("response_provenance") or {}
            check["worker_socket_written"] = provenance.get("delivery_to_worker") == "socket_written"
            check["agent_exact_response_acknowledged"] = (
                provenance.get("agent_receipt") == "reported_exact_body_received")
            check["worker_selected_response_acknowledged"] = (
                provenance.get("selected_response_acknowledged") is True)
    # This is a clearly incomplete diagnostic export assembled from real
    # _calls rows. It does not assert a worker trajectory, score, or label.
    diagnostic = {"schema_version": "leadpoet.lab_arena.training_example.v1",
                  "run_id": context.run_id, "role": "miner",
                  "configuration": {"reasoning_capture_policy": context.reasoning_capture_policy,
                                    "reasoning_route_registry": context.reasoning_route_registry},
                  "provider_calls": calls,
                  "completeness": {"complete": False,
                                   "incomplete_reasons": (["scoring_label_unavailable", *worker_gaps]
                                                          if worker_snapshot is not None else
                                                          ["worker_receipts_unavailable", "scoring_label_unavailable"])}}
    if worker_snapshot is not None:
        diagnostic["worker_observation_artifacts"] = {context.run_id: worker_snapshot}
    diagnostic["example_hash"] = contracts.document_hash(diagnostic)
    _private_write(root / "diagnostic-export.json", _json_bytes(diagnostic))
    readable = training_thinking.readable_trace(diagnostic)
    _private_write(root / "readable-trace.json", _json_bytes(readable))
    markdown = training_thinking.readable_trace_markdown(
        readable, raw_export_name="diagnostic-export.json")
    _private_write(root / "readable-trace.md", markdown.encode("utf-8"))
    return checks


def run_probe(*, model: str, output: Path, key: str, route_file: Path | None = None,
              transport=None, budget_usd: Decimal = BUDGET_USD,
              ledger_mode: str = "postgres", budget_state: Path | None = None,
              mode: str = "full", scenario: str = "acme", worker_socket: bool = False,
              diagnostic_tool_choice_auto: bool = False,
              diagnostic_responses_compat: bool = False,
              tool_choice: str = "named", final_format: str = "json_schema",
              budget_bound_mode: str = "context", parallel_tool_calls: bool | None = None):
    registry, route, authority = route_registry(route_file, model)
    spec = scenario_spec(scenario, final_format=final_format)
    operation_id = route["api"]
    if (tool_choice not in {"named", "auto", "required"}
            or final_format not in {"json_schema", "json_object", "text"}
            or budget_bound_mode not in {"context", "request"}):
        raise ValueError("invalid_probe_controls")
    if mode not in {"full", "summary_diagnostic"}:
        raise ValueError("invalid_probe_mode")
    if (diagnostic_tool_choice_auto or diagnostic_responses_compat) and mode != "summary_diagnostic":
        raise ValueError("diagnostic_controls_require_summary_mode")
    if diagnostic_tool_choice_auto and operation_id != "openrouter.chat":
        raise ValueError("diagnostic_auto_tool_requires_chat")
    if diagnostic_responses_compat and operation_id != "openrouter.responses":
        raise ValueError("diagnostic_responses_compat_requires_responses")
    if worker_socket and ledger_mode != "postgres":
        raise ValueError("worker_socket_requires_disposable_postgres")
    preflight_probe_controls(model=model, route=route, mode=mode, scenario=scenario,
                             tool_choice=tool_choice, final_format=final_format,
                             parallel_tool_calls=parallel_tool_calls,
                             diagnostic_tool_choice_auto=diagnostic_tool_choice_auto,
                             diagnostic_responses_compat=diagnostic_responses_compat)
    per_call = conservative_bound(route)
    if (per_call <= 0 or budget_usd <= 0 or budget_usd > BUDGET_USD
            or budget_bound_mode == "context" and per_call * 3 > budget_usd):
        raise ValueError("worst_case_three_call_budget_exceeded")
    shared_budget = SharedBudget(budget_state or output.parent / "arena-live-capture-budget.json",
                                 budget_usd)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(output, 0o700)
    objects = LocalObjectStore(output / "objects")
    if ledger_mode == "postgres":
        backend = DisposablePostgres()
        try:
            store, context = backend.open(registry)
        except BaseException:
            backend.close()
            raise
    elif ledger_mode == "local_diagnostic":
        backend = None
        store = AuditStore(output / "ledger.jsonl", budget_microusd=int(budget_usd * 1_000_000))
        context = replace(CONTEXT, run_id="capture-probe-" + hashlib.sha256(str(output).encode()).hexdigest()[:16],
                          round_id="disposable-capture-probe", reasoning_capture_policy=rc.POLICY,
                          reasoning_route_registry=registry)
    else:
        raise ValueError("unknown_ledger_mode")
    if mode == "summary_diagnostic":
        context = replace(context, reasoning_capture_policy=None,
                          reasoning_route_registry=None)
    wire_transport = WireTransport(output, transport, secret=key, operation_id=operation_id,
                                  request_budget=shared_budget if budget_bound_mode == "request" else None,
                                  budget_route=route)
    capture = training_capture.RequestCapture(objects)
    broker = br.Broker(store=store, key_for=lambda provider: key if provider == "openrouter" else "",
                       price_table=_price_table(route), transport=wire_transport,
                       capture_request=capture.prepare,
                       credential_for=lambda _context, provider: key if provider == "openrouter" else "",
                       funding_source_for=lambda _context: "miner_key")
    socket_dir = None
    socket_server = None
    socket_api = None
    socket_state = None
    previous_socket = os.environ.get(shim.WORKER_SOCKET_ENV)
    previous_policy = os.environ.get(shim.REASONING_POLICY_ENV)
    if worker_socket:
        if backend.run is None:
            raise ValueError("disposable_run_missing")
        lease = dict(backend.run)
        lease["kind"] = "execute"
        if context.reasoning_capture_policy is not None:
            lease["reasoning_capture_policy"] = context.reasoning_capture_policy
        socket_state = runner.RunState(lease=lease, lease_token="capture-probe-local-socket")
        socket_api = BrokerSocketApi(broker, context)
        socket_dir = Path(tempfile.mkdtemp(prefix="ar-probe-", dir="/tmp"))
        socket_server = runner.WorkerSocketServer(socket_dir / "w.sock", socket_api, socket_state)
        # The production worker may poll a broker's temporary no-dispatch
        # billing hold. This paid probe has a stricter one-attempt contract.
        # Disable that server-instance poll; all socket framing and receipts
        # still use the real worker implementation.
        socket_server._temporary_hold = lambda document, operation_id, action_sequence: False
        # The production Responses bridge can retry a proved-free 429. The
        # probe's one-attempt and shared-budget contract forbids that retry.
        socket_server._responses_rate_limit_delay = lambda call, retry: None
        socket_server.start()
        os.environ[shim.WORKER_SOCKET_ENV] = str(socket_dir / "w.sock")
        if context.reasoning_capture_policy is not None:
            os.environ[shim.REASONING_POLICY_ENV] = context.reasoning_capture_policy
    messages = copy.deepcopy(spec["initial"])
    spent = Decimal(0)
    reserved_uncertain = Decimal(0)
    outcomes = []
    protocol_complete = False
    failure = None
    failure_turn = None
    failure_phase = None
    try:
        for turn in range(3):
            if budget_bound_mode == "context":
                if spent + reserved_uncertain + per_call > budget_usd:
                    raise ValueError("cumulative_budget_exceeded")
                shared_budget.reserve(per_call)
            # A dispatch may bill even when the response is lost. Reserve the
            # full bound until the provider reports a smaller definite cost.
                reserved_uncertain += per_call
            wire_transport.arm(turn)
            route_scope = (patch.object(br, "_openrouter_host_route",
                                        return_value=_diagnostic_host_route(
                                            route, responses_compat=diagnostic_responses_compat))
                           if mode == "summary_diagnostic" else nullcontext())
            original_build = operations.build_outbound_request
            original_validate = operations.validate_operation_request

            def diagnostic_validate(selected_operation, parameters, **kwargs):
                # Keep new continuation shapes local to this isolated probe.
                # Production rounds with no capture policy retain main's schema.
                if selected_operation == operation_id:
                    kwargs["allow_full_thinking"] = True
                return original_validate(selected_operation, parameters, **kwargs)

            def diagnostic_build(selected_operation, parameters, **kwargs):
                # The operation validator only accepts a single exact pin when
                # allow_full_thinking is set. This probe-only adapter grants
                # that request shape, without adding policy/approval annotation.
                if selected_operation == operation_id:
                    kwargs["allow_full_thinking"] = True
                    if diagnostic_responses_compat:
                        strict_policy = dict(kwargs["openrouter_provider_policy"])
                        strict_policy["require_parameters"] = True
                        kwargs["openrouter_provider_policy"] = strict_policy
                outbound = original_build(selected_operation, parameters, **kwargs)
                if selected_operation == operation_id and diagnostic_responses_compat:
                    # Preserve the production validator and change only this
                    # isolated diagnostic outbound after exact-pin validation.
                    document = json.loads(outbound.body)
                    if document["provider"].get("require_parameters") is not True:
                        raise ValueError("diagnostic_pin_invalid")
                    document["provider"]["require_parameters"] = False
                    outbound = replace(outbound, body=json.dumps(
                        document, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False).encode("utf-8"))
                return outbound

            build_scope = (patch.object(operations, "build_outbound_request",
                                        side_effect=diagnostic_build)
                           if mode == "summary_diagnostic" else nullcontext())
            validation_scope = (patch.object(operations, "validate_operation_request",
                                             side_effect=diagnostic_validate)
                                if mode == "summary_diagnostic" else nullcontext())
            with route_scope, build_scope, validation_scope:
                parameters = _parameters(
                    model, messages, turn, operation_id, mode, route,
                    diagnostic_tool_choice_auto=diagnostic_tool_choice_auto,
                    diagnostic_responses_compat=diagnostic_responses_compat,
                    tool_choice=tool_choice, final_format=final_format,
                    parallel_tool_calls=parallel_tool_calls)
                if worker_socket:
                    status, headers, delivered_body = shim.dispatch(operation_id, parameters, 120000)
                    result = socket_api.results[-1]
                    if status != result.status or delivered_body != result.body:
                        raise ValueError("socket_delivery_mismatch")
                    identity = headers.get(shim.CALL_IDENTITY_HEADER)
                    if identity != result.call.get("call_identity"):
                        raise ValueError("socket_identity_mismatch")
                    if lab_arena_checkpoint.record_response_received(
                            call_identity=identity, status=status, body=delivered_body) is None:
                        raise ValueError("socket_agent_ack_failed")
                else:
                    result = broker.execute(context, operation_id=operation_id,
                                            parameters=parameters,
                                            action_sequence=turn, timeout_ms=120000)
            outcomes.append({"turn": turn, "status": result.status,
                             "outcome": result.call.get("outcome"),
                             "call_identity": result.call.get("call_identity")})
            if budget_bound_mode == "request":
                per_call = wire_transport.request_reservations.get(turn, Decimal(0))
                reserved_uncertain += per_call
            if result.call.get("outcome") != "settled" or result.status != 200:
                break
            # Account for a definite provider charge even when the model
            # violates the requested tool/final format below.
            response = json.loads(result.body)
            observed_cost = response.get("usage", {}).get("cost")
            if observed_cost is None:
                # A settled ledger can lack a provider cost. Keep the worst
                # case reserved for the entire probe; do not infer free usage.
                pass
            else:
                cost = Decimal(str(observed_cost))
                if cost < 0 or cost > per_call:
                    raise ValueError("provider_cost_outside_reserved_bound")
                shared_budget.settle(per_call, cost)
                wire_transport.settled_turns.add(turn)
                reserved_uncertain -= per_call
                spent += cost
            _, continuation = _next_message(result.body, turn, operation_id, spec)
            if mode == "summary_diagnostic" and operation_id == "openrouter.chat":
                continuation = _legacy_chat_diagnostic_continuation(continuation)
            messages.extend(continuation)
        protocol_complete = len(outcomes) == 3 and all(
            row["status"] == 200 and row["outcome"] == "settled" for row in outcomes)
    except Exception as exc:
        failure = type(exc).__name__
        failure_turn = turn
        failure_phase = ("after_provider_http_send" if wire_transport.armed_turn == turn
                         and wire_transport.sent_in_turn else
                         "before_provider_http_send")
    finally:
        if budget_bound_mode == "request":
            reserved_uncertain = sum((amount for selected_turn, amount in
                wire_transport.request_reservations.items()
                if selected_turn not in wire_transport.settled_turns), Decimal(0))
        wire_transport.close()
        worker_snapshot = None
        worker_failure = None
        if socket_server is not None:
            socket_server.stop()
            if previous_socket is None:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            else:
                os.environ[shim.WORKER_SOCKET_ENV] = previous_socket
            if previous_policy is None:
                os.environ.pop(shim.REASONING_POLICY_ENV, None)
            else:
                os.environ[shim.REASONING_POLICY_ENV] = previous_policy
            shutil.rmtree(socket_dir)
            try:
                payload = socket_state.worker_observations.document()
                run = backend.run
                snapshot, compressed = training_observations.build_snapshot(
                    context.run_id, "execute", payload,
                    lease_generation=run["lease_generation"],
                    claim_request_hash=run["claim_request_hash"],
                    secrets=(socket_state.lease_token, key))
                ref = training_observations.object_ref(
                    context.run_id, run["lease_generation"], run["claim_request_hash"])
                objects.put(ref, compressed)
                worker_snapshot = training_observations.read_snapshot(
                    LocalObjectStore(output / "objects"), context.run_id, "execute",
                    run["lease_generation"], run["claim_request_hash"])
                if worker_snapshot["payload_sha256"] != snapshot["payload_sha256"]:
                    raise ValueError("worker_snapshot_readback_mismatch")
            except Exception as exc:
                worker_failure = type(exc).__name__
        try:
            rows = backend.rows() if backend is not None else None
            if rows is not None:
                # The fixture database is disposable. Keep its actual SQL
                # readback before a later export or comparison can fail.
                _private_write(output / "ledger-readback.json", _json_bytes({
                    "schema_version": "leadpoet.lab_arena.disposable_ledger_readback.v1",
                    "run_id": context.run_id, "rows": rows}))
            readback = verify_storage(output, context, registry, wire_transport.records,
                                      rows=rows, operation_id=operation_id,
                                      worker_run=backend.run if backend is not None else None,
                                      worker_snapshot=worker_snapshot) if wire_transport.records else []
            readback_failure = None
        except Exception as exc:
            readback, readback_failure = [], type(exc).__name__
        returned = [row["capture"]["returned"] for row in readback]
        worker_receipts_proved = bool(readback) and all(
            row.get("worker_socket_written") and row.get("agent_exact_response_acknowledged")
            and row.get("worker_selected_response_acknowledged") for row in readback)
        classification = ("full_thinking_observed" if protocol_complete and len(returned) == 3
                          and "full_thinking" in returned
                          and all(value in {"full_thinking", "requested_empty"} for value in returned) else
                          "summary_only_diagnostic" if returned and all(value == "summary" for value in returned) else
                          "incomplete_or_unproved")
        report = {"model": model, "provider": route["provider"], "route_authority": authority,
                  "mode": mode,
                  "diagnostic_tool_choice_auto": diagnostic_tool_choice_auto,
                  "diagnostic_responses_compat": diagnostic_responses_compat,
                  "scenario": scenario,
                  "tool_choice": tool_choice, "final_format": final_format,
                  "parallel_tool_calls": parallel_tool_calls,
                  "native_final_constraint": ({"json_schema": "strict_json_schema",
                                                "json_object": "json_object_only",
                                                "text": "unconstrained_text"}[final_format]),
                  "final_value_validation": "exact_local_qualification_schema_and_expected_values",
                  "budget_bound_mode": budget_bound_mode,
                  "request_budget_bounds_usd": {str(k): str(v) for k, v in
                                                wire_transport.request_reservations.items()},
                  "ledger_mode": ledger_mode,
                  "worker_socket_enabled": worker_socket,
                  "worker_temporary_hold_retry_disabled": worker_socket,
                  "worker_responses_429_retry_disabled": worker_socket,
                  "worker_receipts_proved": worker_receipts_proved,
                  "worker_snapshot_failure": worker_failure,
                  "worker_snapshot_payload_sha256": worker_snapshot.get("payload_sha256") if worker_snapshot else None,
                  "operation_id": operation_id, "budget_usd": str(budget_usd),
                  "shared_budget_state": str(shared_budget.path),
                  "worst_case_per_call_usd": str(max(wire_transport.request_reservations.values(),
                                                      default=Decimal(0))
                                                    if budget_bound_mode == "request" else per_call),
                  "budget_bound_note": ("Exact outbound canonical UTF-8 bytes plus Broker token overhead, "
                                        "full requested output cap and route price ceilings; checked before each HTTP send. "
                                        "A later continuation can be refused when the shared cap is exhausted."
                                        if budget_bound_mode == "request" else
                                        "Full advertised context and tested output cap reserved per turn."),
                  "provider_reported_spend_usd": str(spent),
                  "still_reserved_or_unpriced_usd": str(reserved_uncertain),
                  "turns": outcomes, "readback": readback,
                  "probe_failure": failure, "probe_failure_turn": failure_turn,
                  "probe_failure_phase": failure_phase,
                  "readback_failure": readback_failure,
                  "budget_hold_note": "Unpriced or failed attempts retain their full reserved bound; no charge is inferred.",
                  "classification": classification,
                  "all_turns_full_thinking": len(returned) == 3 and all(
                      value == "full_thinking" for value in returned),
                  "tool_continuations_and_final_json": protocol_complete,
                  "registry_listed": authority == "working_registry",
                  "production_activation": False,
                  "route_probe_passed": mode == "full" and authority == "working_registry"
                                            and classification == "full_thinking_observed",
                  "transport_body_note": "Exact bytes returned by HTTP transport after HTTP content decoding; distinct from privacy-sanitized export.",
                  "proof_limit": ("Disposable PostgreSQL ArenaStore RPC and SQL ledger readback; " if backend else
                                  "Local diagnostic file ledger; ") +
                                 ("real local worker socket write and SDK read acknowledgement; " if worker_receipts_proved else
                                  "no proven worker receipt or agent acknowledgement; ") +
                                 "no complete training trajectory or production activation."}
        try:
            _private_write(output / "report.json", _json_bytes(report))
        finally:
            if backend is not None:
                backend.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-route", type=Path)
    parser.add_argument("--mode", choices=("full", "summary_diagnostic"), default="full")
    parser.add_argument("--scenario", choices=("acme", "conflicting_evidence"), default="acme")
    parser.add_argument("--worker-socket", action="store_true",
                        help="exercise real local worker socket and SDK acknowledgement")
    parser.add_argument("--tool-choice", choices=("named", "auto", "required"), default="named")
    parser.add_argument("--final-format", choices=("json_schema", "json_object", "text"), default="json_schema")
    parser.add_argument("--serial-tools", action="store_true",
                        help="explicit test control: send parallel_tool_calls=false on tool turns")
    parser.add_argument("--budget-bound-mode", choices=("context", "request"), default="context",
                        help="request: reserve exact serialized outbound input plus full output before HTTP")
    parser.add_argument("--diagnostic-tool-choice-auto", action="store_true",
                        help="summary-only Chat probe: permit an automatic tool decision")
    parser.add_argument("--diagnostic-responses-compat", action="store_true",
                        help="summary-only Responses probe: summary auto, require_parameters false")
    parser.add_argument("--budget-usd", type=Decimal, default=BUDGET_USD)
    parser.add_argument("--budget-state", type=Path,
                        help="shared private budget file for all route probes; defaults to output parent")
    parser.add_argument("--ledger-mode", choices=("postgres", "local_diagnostic"),
                        default="postgres")
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY must be set in the environment")
    result = run_probe(model=args.model, output=args.output, key=key,
                       route_file=args.candidate_route, budget_usd=args.budget_usd,
                       ledger_mode=args.ledger_mode, budget_state=args.budget_state,
                       mode=args.mode, scenario=args.scenario,
                       worker_socket=args.worker_socket,
                       tool_choice=args.tool_choice, final_format=args.final_format,
                       parallel_tool_calls=False if args.serial_tools else None,
                       budget_bound_mode=args.budget_bound_mode,
                       diagnostic_tool_choice_auto=args.diagnostic_tool_choice_auto,
                       diagnostic_responses_compat=args.diagnostic_responses_compat)
    print(json.dumps({key: result[key] for key in ("model", "provider", "route_authority",
                                               "provider_reported_spend_usd",
                                               "tool_continuations_and_final_json", "classification")}))


if __name__ == "__main__":
    main()
