#!/usr/bin/env python3
"""Bounded, local Arena capture probe. Never writes to production.

One exact OpenRouter route, three sequential synthetic turns. The real Broker,
HTTP transport, request capture, response receipt, and training-export decoder
are exercised. The ledger is a durable local diagnostic adapter, not PostgreSQL.
No worker socket, agent acknowledgement, complete trajectory, or production
round is claimed. Candidate route files are admitted only in this process.
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
from unittest.mock import patch

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc, training_capture, training_thinking
from lab_arena.service import LocalObjectStore
from lab_arena.store import ArenaStore, PsycopgTransport
from lab_arena.training_export import _calls
from scripts.probe_full_thinking_routes import INITIAL, TOOLS, SCHEMA, EVIDENCE, assistant_history
from tests.lab_arena.test_lab_arena_broker import FakeLedgerStore, CONTEXT

BUDGET_USD = Decimal("8")
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


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

    def __init__(self, root: Path, inner=None, *, secret: str = "", operation_id="openrouter.chat"):
        self.root = root
        self.inner = inner or br.HttpxProviderTransport(max_response_bytes=MAX_RESPONSE_BYTES)
        self.secret = secret.encode()
        self.operation_id = operation_id
        self.attempt = 0
        self.records = []
        self.armed_turn = None
        self.sent_in_turn = False

    def arm(self, turn: int):
        self.armed_turn = turn
        self.sent_in_turn = False

    def send(self, **kwargs):
        expected = ("https://openrouter.ai/api/v1/chat/completions" if self.operation_id == "openrouter.chat"
                    else "https://openrouter.ai/api/v1/responses")
        if kwargs.get("url") != expected or self.armed_turn is None or self.sent_in_turn:
            raise br.ProviderTransportError("probe_additional_http_blocked")
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


def route_registry(route_file: Path | None, model: str):
    committed = rc.load_registry()
    if route_file is None:
        route = next((r for r in committed["approved_routes"] if r["model_id"] == model), None)
        if route is None:
            raise ValueError("route_not_in_committed_registry; supply --candidate-route")
        return committed, route, "committed"
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
             max(Decimal(price["completion"]), Decimal(price["internal_reasoning"]))
             + Decimal(price["request"]))
    return bound.quantize(Decimal("0.000001"), rounding=ROUND_UP)


def _price_table(route):
    return br.validate_price_table({"schema_version": br.PRICE_TABLE_SCHEMA_VERSION,
                                    "fetched_at": "2026-10-23T00:00:00Z",
                                    "source": br.OPENROUTER_MODELS_URL,
                                    "models": {route["model_id"]: {
                                        key: route["price_ceiling"].get(key, "0")
                                        for key in br.PRICED_COMPONENTS}}})


def _diagnostic_host_route(route):
    """Pin a probe-only route without assigning full-thinking policy evidence."""
    ceiling = route["price_ceiling"]
    price = {key: Decimal(ceiling.get(key, "0")) for key in br.PRICED_COMPONENTS}
    return br.OpenRouterHostRoute(
        provider_policy={"data_collection": "deny", "zdr": True,
                         "only": [route["provider"]], "allow_fallbacks": False,
                         "require_parameters": True,
                         "max_price": {
                             "prompt": float(price["prompt"] * br.OPENROUTER_PRICE_PER_MILLION),
                             "completion": float((price["completion"] + price["internal_reasoning"])
                                                 * br.OPENROUTER_PRICE_PER_MILLION),
                             "request": float(price["request"])}},
        reservation_pricing={key: ceiling.get(key, "0") for key in br.PRICED_COMPONENTS})


def _parameters(model, messages, turn, operation_id, mode, route):
    if operation_id == "openrouter.responses":
        value = {"model": model, "input": copy.deepcopy(messages),
                 "max_output_tokens": 2048}
        if mode == "summary_diagnostic":
            value["reasoning"] = {"effort": route["mode"].get("effort", "high"),
                                  "summary": "detailed"}
        if turn < 2:
            value.update(tools=[{"type": "function", "name": "lookup_evidence",
                                 "description": "Read synthetic company evidence by ID.",
                                 "parameters": TOOLS[0]["function"]["parameters"]}],
                         tool_choice="required", parallel_tool_calls=False,
                         )
        else:
            value["text"] = {"format": {"type": "json_schema", "name": "qualification",
                                         "strict": True, "schema": SCHEMA}}
        return value
    value = {"model": model, "messages": copy.deepcopy(messages),
             "max_tokens": 2048}
    if mode == "summary_diagnostic":
        value["reasoning"] = {**route["mode"], "exclude": False}
    if turn < 2:
        value.update(tools=TOOLS, tool_choice={"type": "function",
                                                "function": {"name": "lookup_evidence"}})
    else:
        value["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "qualification", "strict": True, "schema": SCHEMA}}
    return value


def _next_message(body, turn, operation_id):
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
                               "output": json.dumps(EVIDENCE[expected], separators=(",", ":"))}]
        texts = [part.get("text") for item in output if item.get("type") == "message"
                 for part in item.get("content") or [] if part.get("type") == "output_text"]
        final = json.loads(texts[0]) if len(texts) == 1 else None
        if final != {"company": "Acme Test", "qualified": True,
                     "evidence_ids": ["profile", "headcount"]}:
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
                          "content": json.dumps(EVIDENCE[expected], separators=(",", ":"))}]
    final = json.loads(message.get("content") or "null")
    if final != {"company": "Acme Test", "qualified": True,
                 "evidence_ids": ["profile", "headcount"]}:
        raise ValueError("strict_final_json_mismatch")
    return response, []


def verify_storage(root: Path, context, registry, wire_records, *, rows=None,
                   operation_id="openrouter.chat") -> list[dict]:
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
        terminal = next(r for r in rows if r["entry_kind"] == "settlement"
                        and r["call_identity"] == call["call_identity"])
        receipt = terminal["terminal_response"]
        decoded = base64.b64decode(receipt["body_b64"], validate=True)
        exact = (contracts.hash_bytes(raw) == wire["sha256"]
                 and contracts.hash_bytes(decoded) == wire["sha256"])
        # Response privacy can alter the exported JSON. Compare the sanitized
        # value with the real export, and report exactness separately.
        from lab_arena.training_export import _response_value
        sanitized, redactions = _response_value(receipt, "openrouter")
        readable = call.get("response") == sanitized
        outbound = ((call.get("request") or {}).get("outbound") or {})
        saved_outbound_hash = outbound.get("body_sha256")
        coverage = training_thinking.attempt_coverage(call, "miner", registry)
        checks.append({"call_identity": call["call_identity"],
                       "wire_receipt_exact": exact,
                       "outbound_request_exact": saved_outbound_hash == wire["outbound_body_sha256"],
                       "stored_request_readable": isinstance(call.get("request"), dict),
                       "stored_response_readable": readable,
                       "response_privacy_redactions": redactions,
                       "capture": coverage,
                       "evidence_gaps": call.get("evidence_gaps", [])})
    if len(calls) != len(wire_records):
        raise ValueError("ledger_wire_count_mismatch")
    if any(not all(row[key] for key in ("wire_receipt_exact", "outbound_request_exact", "stored_request_readable",
                                       "stored_response_readable")) for row in checks):
        raise ValueError("durable_readback_mismatch")
    # This is a clearly incomplete diagnostic export assembled from real
    # _calls rows. It does not assert a worker trajectory, score, or label.
    diagnostic = {"schema_version": "leadpoet.lab_arena.training_example.v1",
                  "run_id": context.run_id, "role": "miner",
                  "configuration": {"reasoning_capture_policy": context.reasoning_capture_policy,
                                    "reasoning_route_registry": context.reasoning_route_registry},
                  "provider_calls": calls,
                  "completeness": {"complete": False,
                                   "incomplete_reasons": ["worker_receipts_unavailable",
                                                          "scoring_label_unavailable"]}}
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
              mode: str = "full"):
    registry, route, authority = route_registry(route_file, model)
    operation_id = route["api"]
    if mode not in {"full", "summary_diagnostic"}:
        raise ValueError("invalid_probe_mode")
    per_call = conservative_bound(route)
    if per_call <= 0 or per_call * 3 > budget_usd or budget_usd > BUDGET_USD:
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
    wire_transport = WireTransport(output, transport, secret=key, operation_id=operation_id)
    capture = training_capture.RequestCapture(objects)
    broker = br.Broker(store=store, key_for=lambda provider: key if provider == "openrouter" else "",
                       price_table=_price_table(route), transport=wire_transport,
                       capture_request=capture.prepare,
                       credential_for=lambda _context, provider: key if provider == "openrouter" else "",
                       funding_source_for=lambda _context: "miner_key")
    messages = copy.deepcopy(INITIAL)
    spent = Decimal(0)
    reserved_uncertain = Decimal(0)
    outcomes = []
    protocol_complete = False
    failure = None
    try:
        for turn in range(3):
            if spent + reserved_uncertain + per_call > budget_usd:
                raise ValueError("cumulative_budget_exceeded")
            shared_budget.reserve(per_call)
            # A dispatch may bill even when the response is lost. Reserve the
            # full bound until the provider reports a smaller definite cost.
            reserved_uncertain += per_call
            wire_transport.arm(turn)
            route_scope = (patch.object(br, "_openrouter_host_route",
                                        return_value=_diagnostic_host_route(route))
                           if mode == "summary_diagnostic" else nullcontext())
            original_build = operations.build_outbound_request

            def diagnostic_build(selected_operation, parameters, **kwargs):
                # The operation validator only accepts a single exact pin when
                # allow_full_thinking is set. This probe-only adapter grants
                # that request shape, without adding policy/approval annotation.
                if selected_operation == operation_id:
                    kwargs["allow_full_thinking"] = True
                return original_build(selected_operation, parameters, **kwargs)

            build_scope = (patch.object(operations, "build_outbound_request",
                                        side_effect=diagnostic_build)
                           if mode == "summary_diagnostic" else nullcontext())
            with route_scope, build_scope:
                result = broker.execute(context, operation_id=operation_id,
                                        parameters=_parameters(model, messages, turn,
                                                               operation_id, mode, route),
                                        action_sequence=turn, timeout_ms=120000)
            outcomes.append({"turn": turn, "status": result.status,
                             "outcome": result.call.get("outcome"),
                             "call_identity": result.call.get("call_identity")})
            if result.call.get("outcome") != "settled" or result.status != 200:
                break
            response, continuation = _next_message(result.body, turn, operation_id)
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
                reserved_uncertain -= per_call
                spent += cost
            messages.extend(continuation)
        protocol_complete = len(outcomes) == 3 and all(
            row["status"] == 200 and row["outcome"] == "settled" for row in outcomes)
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        wire_transport.close()
        try:
            rows = backend.rows() if backend is not None else None
            readback = verify_storage(output, context, registry, wire_transport.records,
                                      rows=rows, operation_id=operation_id) if wire_transport.records else []
            readback_failure = None
        except Exception as exc:
            readback, readback_failure = [], type(exc).__name__
        returned = [row["capture"]["returned"] for row in readback]
        classification = ("full_thinking_observed" if protocol_complete and len(returned) == 3
                          and "full_thinking" in returned
                          and all(value in {"full_thinking", "empty"} for value in returned) else
                          "summary_only_diagnostic" if returned and all(value == "summary" for value in returned) else
                          "incomplete_or_unproved")
        report = {"model": model, "provider": route["provider"], "route_authority": authority,
                  "mode": mode,
                  "ledger_mode": ledger_mode,
                  "operation_id": operation_id, "budget_usd": str(budget_usd),
                  "shared_budget_state": str(shared_budget.path),
                  "worst_case_per_call_usd": str(per_call),
                  "provider_reported_spend_usd": str(spent),
                  "still_reserved_or_unpriced_usd": str(reserved_uncertain),
                  "turns": outcomes, "readback": readback,
                  "probe_failure": failure, "readback_failure": readback_failure,
                  "classification": classification,
                  "all_turns_full_thinking": len(returned) == 3 and all(
                      value == "full_thinking" for value in returned),
                  "tool_continuations_and_final_json": protocol_complete,
                  "production_whitelist_approved": authority == "committed",
                  "full_thinking_eligible": mode == "full" and authority == "committed"
                                            and classification == "full_thinking_observed",
                  "transport_body_note": "Exact bytes returned by HTTP transport after HTTP content decoding; distinct from privacy-sanitized export.",
                  "proof_limit": ("Disposable PostgreSQL ArenaStore RPC and SQL ledger readback; " if backend else
                                  "Local diagnostic file ledger; ") +
                                 "no worker receipt, agent acknowledgement, complete training trajectory, or production activation."}
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
                       mode=args.mode)
    print(json.dumps({key: result[key] for key in ("model", "provider", "route_authority",
                                               "provider_reported_spend_usd",
                                               "tool_continuations_and_final_json", "classification")}))


if __name__ == "__main__":
    main()
