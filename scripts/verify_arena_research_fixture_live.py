#!/usr/bin/env python3
"""Run exact reviewed synthetic source through a real disposable Arena Broker.

The fixture's injected model callable dispatches through the local worker socket,
Broker, and disposable PostgreSQL. This checks actual fixture tool control and
saved full-thinking receipts. It does not admit a miner submission, score an ICP,
or prove a complete production training trajectory. Raw wire and SQL evidence
is private. No paid request is sent by import or offline preparation.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import broker as br, contracts, lab_arena_checkpoint, runner, shim
from lab_arena import training_capture, training_observations
from lab_arena.service import LocalObjectStore
from scripts.verify_arena_live_capture import (
    BUDGET_USD, BrokerSocketApi, DisposablePostgres, SharedBudget, WireTransport,
    _json_bytes, _price_table, _private_write, route_registry, verify_storage,
)
from scripts.verify_arena_research_review import prepared_case

MODEL_ID = "deepseek/deepseek-v4.1-flash-1017"
SCENARIOS = (("adaptive_json", "sufficient"), ("adaptive_json", "continue"),
             ("fixed_extractor", "sufficient"),
             ("posthoc_explanation", "sufficient"))
COMPANIES = [{"company_name": "Example", "company_website": "https://example.com"}]
ICP = {"industry": "software"}


def _fixture(case: str):
    path = ROOT / "tests" / "fixtures" / "research_review" / case / "harness.py"
    if case not in {name for name, _ in SCENARIOS} or not path.is_file():
        raise ValueError("fixture_not_approved")
    spec = importlib.util.spec_from_file_location("live_review_fixture_" + case, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _system_prompt(case: str) -> str:
    if case == "adaptive_json":
        return (
            "You control a synthetic research agent. Return exactly one JSON object. "
            "Read the current observations in the user JSON. Choose a useful next action "
            "from search, fetch, or finish. Search for the ICP when no observations exist. "
            "If the latest observation says finding=sufficient, finish with companies "
            "[{\"company_name\":\"Example\",\"company_website\":\"https://example.com\"}]. "
            "If the search observation is not sufficient, fetch its url. After a fetch, "
            "finish. For search/fetch use exactly action and arguments; for finish use "
            "action and companies. Do not include commentary outside JSON."
        )
    if case == "fixed_extractor":
        return (
            "Extract one synthetic company from the supplied evidence. Return exactly "
            "JSON object {\"companies\":[{\"company_name\":\"Example\","
            "\"company_website\":\"https://example.com\"}]}."
        )
    return "Explain the supplied synthetic research evidence in one JSON object."


class LocalTools:
    def __init__(self, finding: str):
        self.finding = finding
        self.calls = []

    def __call__(self, name, arguments):
        if name not in {"search", "fetch"} or not isinstance(arguments, dict):
            raise ValueError("fixture_tool_call_invalid")
        self.calls.append({"name": name, "arguments": arguments})
        return {"url": "https://example.com/evidence", "finding": self.finding,
                "company_name": "Example", "company_website": "https://example.com"}


def run_fixture(*, case: str, finding: str, key: str, output: Path,
                budget_state: Path, transport=None, model_id: str = MODEL_ID) -> dict:
    """One fixture with a caller-supplied test key and no production writes."""
    if (case, finding) not in SCENARIOS or not isinstance(key, str) or not key:
        raise ValueError("fixture_or_key_invalid")
    registry, route, authority = route_registry(None, model_id)
    if route["api"] != "openrouter.chat" or route["provider"] != "fireworks":
        raise ValueError("fixture_route_not_fireworks_chat")
    prepared = prepared_case(case)
    shared_budget = SharedBudget(budget_state, BUDGET_USD)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(output, 0o700)
    objects = LocalObjectStore(output / "objects")
    backend = DisposablePostgres()
    try:
        store, context = backend.open(registry)
    except BaseException:
        backend.close()
        raise
    wire = WireTransport(output, transport, secret=key, operation_id="openrouter.chat")
    capture = training_capture.RequestCapture(objects)
    broker = br.Broker(
        store=store, key_for=lambda provider: key if provider == "openrouter" else "",
        price_table=_price_table(route), transport=wire,
        capture_request=capture.prepare,
        credential_for=lambda _context, provider: key if provider == "openrouter" else "",
        funding_source_for=lambda _context: "miner_key",
    )
    lease = dict(backend.run, kind="execute")
    socket_state = runner.RunState(lease=lease, lease_token="research-fixture-local-socket")
    socket_api = BrokerSocketApi(broker, context)
    socket_dir = Path(tempfile.mkdtemp(prefix="ar-research-fixture-", dir="/tmp"))
    socket_server = runner.WorkerSocketServer(socket_dir / "w.sock", socket_api, socket_state)
    socket_server._temporary_hold = lambda document, operation_id, action_sequence: False
    socket_server._responses_rate_limit_delay = lambda call, retry: None
    previous_socket = os.environ.get(shim.WORKER_SOCKET_ENV)
    socket_server.start()
    os.environ[shim.WORKER_SOCKET_ENV] = str(socket_dir / "w.sock")
    model_calls = []
    tools = LocalTools(finding)
    failed = None
    output_value = None
    fixture_started = False

    def model(request):
        turn = len(model_calls)
        if turn >= 4:
            raise ValueError("fixture_model_call_limit")
        parameters = {"model": model_id,
                      "messages": [{"role": "system", "content": _system_prompt(case)},
                                   {"role": "user", "content": contracts.canonical_json(request)}],
                      "max_tokens": min(2048, int(route["max_output_tokens"])),
                      "response_format": {"type": "json_object"}}
        # Bound the exact normalized request, including the frozen route mode.
        normalized = br.normalized_request(
            "openrouter.chat", parameters,
            reasoning_capture_policy=context.reasoning_capture_policy,
            reasoning_route_registry=context.reasoning_route_registry,
        )
        reservation = br.max_openrouter_cost_microusd(
            _price_table(route), model_id, normalized,
            max_output_tokens=normalized["max_tokens"],
        )
        held = Decimal(reservation) / Decimal(1_000_000)
        shared_budget.reserve(held)
        wire.arm(turn)
        status, headers, body = shim.dispatch("openrouter.chat", parameters, 120000)
        result = socket_api.results[-1]
        if result.status != status or result.body != body:
            raise ValueError("fixture_socket_delivery_mismatch")
        identity = headers.get(shim.CALL_IDENTITY_HEADER)
        if identity != result.call.get("call_identity"):
            raise ValueError("fixture_socket_identity_mismatch")
        if lab_arena_checkpoint.record_response_received(
                call_identity=identity, status=status, body=body) is None:
            raise ValueError("fixture_worker_ack_failed")
        model_calls.append({"turn": turn, "status": status,
                            "outcome": result.call.get("outcome"),
                            "reservation_microusd": reservation,
                            "call_identity": identity})
        if result.call.get("outcome") != "settled" or status != 200:
            raise ValueError("fixture_provider_call_incomplete")
        response = json.loads(body)
        actual = response.get("usage", {}).get("cost")
        if actual is not None:
            cost = Decimal(str(actual))
            if cost < 0 or cost > held:
                raise ValueError("fixture_provider_cost_outside_bound")
            shared_budget.settle(held, cost)
            model_calls[-1]["cost_usd"] = str(cost)
            model_calls[-1]["cost_status"] = "settled"
        else:
            model_calls[-1]["cost_status"] = "held"
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("fixture_model_choice_invalid")
        message = choices[0].get("message") or {}
        if choices[0].get("finish_reason") != "stop" or not isinstance(message.get("content"), str):
            raise ValueError("fixture_model_finish_invalid")
        reply = json.loads(message["content"])
        if not isinstance(reply, dict):
            raise ValueError("fixture_model_json_invalid")
        return reply

    try:
        fixture = _fixture(case)
        fixture.configure(model, tools)
        fixture_started = True
        output_value = fixture.run_icp(ICP)
    except Exception as exc:
        failed = type(exc).__name__
    finally:
        wire.close()
        socket_server.stop()
        if previous_socket is None:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        else:
            os.environ[shim.WORKER_SOCKET_ENV] = previous_socket
        shutil.rmtree(socket_dir)
        worker_snapshot = None
        worker_failure = None
        try:
            payload = socket_state.worker_observations.document()
            snapshot, compressed = training_observations.build_snapshot(
                context.run_id, "execute", payload,
                lease_generation=backend.run["lease_generation"],
                claim_request_hash=backend.run["claim_request_hash"],
                secrets=(socket_state.lease_token, key),
            )
            ref = training_observations.object_ref(
                context.run_id, backend.run["lease_generation"],
                backend.run["claim_request_hash"])
            objects.put(ref, compressed)
            worker_snapshot = training_observations.read_snapshot(
                LocalObjectStore(output / "objects"), context.run_id, "execute",
                backend.run["lease_generation"], backend.run["claim_request_hash"])
            if worker_snapshot["payload_sha256"] != snapshot["payload_sha256"]:
                raise ValueError("fixture_worker_snapshot_mismatch")
        except Exception as exc:
            worker_failure = type(exc).__name__
        try:
            rows = backend.rows()
            _private_write(output / "ledger-readback.json", _json_bytes({
                "schema_version": "leadpoet.lab_arena.disposable_ledger_readback.v1",
                "run_id": context.run_id, "rows": rows,
            }))
            readback = verify_storage(
                output, context, registry, wire.records, rows=rows,
                operation_id="openrouter.chat", worker_run=backend.run,
                worker_snapshot=worker_snapshot,
            ) if wire.records else []
            readback_failure = None
        except Exception as exc:
            readback, readback_failure = [], type(exc).__name__
        finally:
            backend.close()
    names = [item["name"] for item in tools.calls]
    expected_names = (["search"] if case == "adaptive_json" and finding == "sufficient"
                      else ["search", "fetch"])
    full_thinking = any(item["capture"]["returned"] == "full_thinking" for item in readback)
    receipts = bool(readback) and all(
        item.get("worker_socket_written") and item.get("agent_exact_response_acknowledged")
        and item.get("worker_selected_response_acknowledged") for item in readback)
    report = {
        "scope": "synthetic_injected_runtime_not_admitted_submission",
        "test_key_origin": "caller_supplied_gateway_key",
        "synthetic_ledger_funding_label": "miner_key",
        "source_reviewer_invoked": False,
        "model_behavior_scope": "conditional_test_prompt_not_general_agent_quality",
        "case": case, "finding": finding,
        "source_archive_sha256": prepared.source_archive_sha256,
        "model": model_id, "provider": route["provider"],
        "route_authority": authority, "shared_budget_state": str(shared_budget.path),
        "model_calls": model_calls, "tool_calls": tools.calls,
        "output": output_value, "fixture_failure": failed,
        "worker_snapshot_failure": worker_failure,
        "readback_failure": readback_failure,
        "readback_count": len(readback), "worker_receipts_proved": receipts,
        "full_thinking_observed": full_thinking,
        "actual_actions_match_observation": names == expected_names,
        "source_fixture_executed": fixture_started,
        "pass": failed is None and readback_failure is None and worker_failure is None
                and receipts and full_thinking and names == expected_names
                and len(readback) == len(model_calls),
    }
    _private_write(output / "report.json", _json_bytes(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget-state", type=Path, required=True)
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args(argv)
    key = os.environ.get(args.key_env)
    if not key:
        parser.error("live reviewer key unavailable in selected environment variable")
    args.output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    results = []
    for case, finding in SCENARIOS:
        result = run_fixture(case=case, finding=finding, key=key,
                             output=args.output_root / f"{case}-{finding}",
                             budget_state=args.budget_state)
        results.append(result)
        print(json.dumps({"case": case, "finding": finding,
                          "pass": result["pass"],
                          "model_calls": len(result["model_calls"]),
                          "tool_calls": len(result["tool_calls"]),
                          "full_thinking_observed": result["full_thinking_observed"]},
                         sort_keys=True), flush=True)
    return 0 if all(result["pass"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
