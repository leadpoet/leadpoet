"""The real Research Lab judge inside the Arena scorer contract.

The validator's judge sandbox runs the pinned scorer image, which is this
repository's ``scorer_entrypoint`` around the Research Lab evaluator. Every
provider request the judge makes crosses the shim, so it must match one closed
operation and use only the models the scorer policy permits.

This test runs the real evaluator in a subprocess against a permissive fake
provider that answers every frame with a shape-valid reply and records what the
judge asked for.
"""

from __future__ import annotations

import json
import os
import shutil
import site
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest

from lab_arena import contracts, output, scoring, shim
from lab_arena import broker as br
from lab_arena import operations as ops
from lab_arena import runtime

pytest.importorskip("qualification.scoring.lead_scorer")
pytest.importorskip("aiohttp")

from tests.lab_arena.icp_fixtures import raw_icp  # noqa: E402

EVALUATION_DATE = "2026-09-02"
COMPANY = "Acme Robotics"
SITE = "https://www.acmerobotics.io"
NEWS_URL = SITE + "/news/series-b"
SENTENCE = (
    "Acme Robotics announced a Series B funding round of $40 million in July 2026 "
    "to expand its West Coast engineering team."
)
PAGE_TEXT = (
    "Acme Robotics raises Series B. " + SENTENCE + " Acme Robotics is a software company "
    "headquartered in San Francisco, California with 120 employees, founded in 2019. "
    "The company sells a robotics fleet-management platform to logistics operators."
)
PAGE_HTML = "<html><head><title>Acme Robotics raises Series B</title></head><body><h1>%s</h1><p>%s</p></body></html>" % (COMPANY, PAGE_TEXT)


def arena_icp() -> Dict[str, Any]:
    """One ordinary ICP from the same stored daily-set shape as production."""

    return raw_icp("Software", 1)


def companies_for(icp: Mapping[str, Any]) -> List[Dict[str, Any]]:
    bucket = icp["employee_count"][0]
    rows = []
    for index, name in enumerate([COMPANY, "Beta Logistics Software", "Gamma Fleet Systems"]):
        site = SITE if index == 0 else "https://www.%s.com" % name.lower().replace(" ", "")
        rows.append({
            "company_name": name,
            "company_website": site,
            "company_linkedin": "https://www.linkedin.com/company/%s" % name.lower().replace(" ", "-"),
            "industry": icp["industry"],
            "employee_count": bucket,
            "company_stage": str(icp.get("company_stage") or ""),
            "country": icp.get("country") or "United States",
            "state": "",
            "fit_summary": "The company matches the ICP.",
            "fit_evidence_urls": [site + "/about"],
            "intent_signals": [{
                "description": "Announced a Series B funding round in July 2026",
                "url": NEWS_URL if index == 0 else site + "/news",
                "date": "2026-07-15",
                "why_now": "The funding makes outreach timely.",
                "snippet": SENTENCE,
                "matched_icp_signal": 0,
            }],
        })
    # The same contract the Arena applies to every model output.
    return output.validate_companies(rows)


def exa_results() -> Dict[str, Any]:
    return {
        "requestId": "req-1",
        "results": [{
            "id": NEWS_URL,
            "url": NEWS_URL,
            "title": "Acme Robotics raises Series B",
            "publishedDate": "2026-07-15T00:00:00.000Z",
            "author": "",
            "text": PAGE_TEXT,
            "highlights": [SENTENCE],
            "highlightScores": [0.9],
            "score": 0.9,
        }],
        "searchTime": 12.0,
    }


def chat_completion(model: str, content: str) -> Dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480},
    }


COMPANIES: List[Dict[str, Any]] = []  # the companies of the work item being judged, for company-aware replies


def company_for_prompt(prompt: str) -> Dict[str, Any]:
    lowered = prompt.lower()
    for company in COMPANIES:
        domain = company["company_website"].split("//", 1)[-1].replace("www.", "").rstrip("/")
        if domain in lowered or company["company_name"].lower() in lowered:
            return company
    return COMPANIES[0] if COMPANIES else {"company_name": COMPANY, "company_website": SITE, "company_linkedin": ""}


def judge_reply(normalized: Mapping[str, Any]) -> str:
    """A JSON verdict shaped for whichever judge prompt asked, bound to the company it asks about."""

    prompt = " ".join(str(m.get("content") or "") for m in normalized.get("messages") or [])
    company = company_for_prompt(prompt)
    evidence = {"url": NEWS_URL, "quote": SENTENCE}
    verdict = {
        # intent-signal judge
        "verdict": "supported",
        "company_named_in_page": True,
        "quote_supporting_claim": SENTENCE,
        "signal_date": "2026-07-15",
        "reason": "The page names the company and quotes the announcement.",
        # verification helpers
        "verified": True,
        "confidence": 92,
        # company-fit web reverification
        "observed_company_name": company["company_name"],
        "observed_company_website": company["company_website"],
        "observed_company_linkedin": company.get("company_linkedin") or "",
        "observed_employee_count": str(company.get("employee_count") or "51-200"),
        "employee_size_matches": True,
        "observed_industry": "Software",
        "observed_subindustry": "Robotics software",
        "industry_matches": True,
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "observed_company_stage": "Series A",
        "stage_matches": True,
        "attribute_satisfied": True,
        "attribute_evidence": SENTENCE,
        "dimension_evidence": {name: dict(evidence) for name in ("identity", "employee_size", "industry", "geography", "stage", "required_attribute")},
        "decision": "match",
        "evidence_type": "press_release",
    }
    if int(normalized.get("max_tokens") or 0) <= 64 and "json" not in prompt.lower():
        return "press_release"
    return json.dumps(verdict)


def respond(operation_id: str, normalized: Mapping[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
    json_headers = {"content-type": "application/json"}
    if operation_id == "openrouter.chat":
        return 200, json_headers, json.dumps(chat_completion(str(normalized.get("model")), judge_reply(normalized))).encode()
    if operation_id in ("exa.search", "exa.contents"):
        return 200, json_headers, json.dumps(exa_results()).encode()
    if operation_id == "deepline.execute":
        envelope = {"job_id": "job-1", "status": "completed", "result": {"data": exa_results()}, "billing": {"credits_charged": 1, "cost_usd": 0.01}}
        return 200, json_headers, json.dumps(envelope).encode()
    if operation_id.startswith("scrapingdog.scrape"):
        return 200, {"content-type": "text/html; charset=utf-8"}, PAGE_HTML.encode()
    if operation_id.startswith("scrapingdog."):
        return 200, json_headers, json.dumps({"name": COMPANY, "website": SITE, "description": PAGE_TEXT, "employees": "120", "posts": [], "jobs": []}).encode()
    return 200, json_headers, b"{}"


def _recv_exact(connection, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise OSError("connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


class JudgeProviderServer:
    """Answers every judge frame with a shape-valid reply and records what the judge asked."""

    def __init__(self, socket_path: Path) -> None:
        self._path = socket_path
        self._server = None
        self._thread = None
        self.lock = threading.Lock()
        self.frames: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []

    def handle_frame(self, raw: bytes) -> bytes:
        try:
            operation_id, parameters, _timeout = shim.decode_operation_frame(raw)
            normalized = br.normalized_request(operation_id, parameters)
        except (shim.OperationFrameError, ops.OperationError, br.BrokerError) as exc:
            with self.lock:
                self.rejected.append({"error": "%s: %s" % (type(exc).__name__, str(exc)[:160])})
            return shim.encode_worker_error("invalid_request")
        with self.lock:
            self.frames.append({"operation_id": operation_id, "model": normalized.get("model"), "request_hash": contracts.document_hash(normalized)})
        status, headers, body = respond(operation_id, normalized)
        return shim.encode_worker_response(status, headers, body)

    def start(self) -> None:
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = self.request
                connection.settimeout(120)
                try:
                    size = int.from_bytes(_recv_exact(connection, 4), "big")
                    if size < 2 or size > shim.MAX_FRAME_BYTES:
                        payload = shim.encode_worker_error("frame_too_large")
                    else:
                        payload = owner.handle_frame(_recv_exact(connection, size))
                    connection.sendall(len(payload).to_bytes(4, "big") + payload)
                except OSError:
                    return

        if self._path.exists():
            self._path.unlink()
        self._server = socketserver.ThreadingUnixStreamServer(str(self._path), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._path.exists():
            self._path.unlink()


def run_real_judge(work_dir: Path, input_document: Mapping[str, Any], *, timeout_seconds: int = 600) -> Dict[str, Any]:
    """Run the scorer entrypoint on the real evaluator against the permissive provider."""

    COMPANIES[:] = [dict(item) for item in input_document.get("companies") or []]
    run_dir = Path(tempfile.mkdtemp(prefix="judge-", dir=str(work_dir)))
    socket_dir = Path(tempfile.mkdtemp(prefix="lj", dir="/tmp"))
    socket_path = socket_dir / runtime.SANDBOX_SOCKET_NAME
    server = JudgeProviderServer(socket_path)
    try:
        (run_dir / "sitecustomize.py").write_text("from lab_arena import shim\nshim.install()\n", encoding="utf-8")
        input_path = run_dir / "input.json"
        output_path = run_dir / "output.json"
        input_path.write_text(json.dumps(dict(input_document), sort_keys=True), encoding="utf-8")
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(run_dir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join((str(run_dir), repo_root, site.getusersitepackages())),
            "LAB_ARENA_INPUT_PATH": str(input_path),
            "LAB_ARENA_OUTPUT_PATH": str(output_path),
            shim.WORKER_SOCKET_ENV: str(socket_path),
            shim.TRUSTED_SCORER_ENV: "1",
            "LAB_ARENA_EVALUATION_DATE": str(input_document.get("evaluation_date") or ""),
            shim.TRACE_PATH_ENV: str(run_dir / "shim-trace.jsonl"),
        }
        environment.update(runtime.PROVIDER_BASE_URLS)
        server.start()
        started = time.monotonic()
        completed = subprocess.run([sys.executable, "-m", "lab_arena.scorer_entrypoint"], cwd=str(run_dir), env=environment, capture_output=True, timeout=timeout_seconds, check=False)
        seconds = time.monotonic() - started
        output = scoring.scoring_output_from_bytes(output_path.read_bytes()) if output_path.exists() else None
        trace_path = run_dir / "shim-trace.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()] if trace_path.exists() else []
        return {
            "trace": trace,
            "exit_code": completed.returncode,
            "output": output,
            "stderr": completed.stderr.decode("utf-8", "replace")[-6000:],
            "stdout": completed.stdout.decode("utf-8", "replace")[-2000:],
            "frames": list(server.frames),
            "rejected": list(server.rejected),
            "seconds": seconds,
        }
    finally:
        server.stop()
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(socket_dir, ignore_errors=True)


def scoring_input() -> Dict[str, Any]:
    icp = arena_icp()
    return scoring.build_scoring_input(scored_run_id="real-judge", icp=icp, companies=companies_for(icp), policy=scoring.build_scorer_policy(), evaluation_date=EVALUATION_DATE)


def describe(result: Mapping[str, Any]) -> str:
    operations_used = sorted({frame["operation_id"] for frame in result["frames"]})
    models_used = sorted({str(frame["model"]) for frame in result["frames"] if frame.get("model")})
    lines = [
        "exit_code=%s seconds=%.1f frames=%d rejected=%d" % (result["exit_code"], result["seconds"], len(result["frames"]), len(result["rejected"])),
        "operations=%s" % operations_used,
        "models=%s" % models_used,
        "rejected=%s" % result["rejected"][:5],
        "trace_no_match=%s" % [t for t in result["trace"] if t.get("event") == "no_match"][:8],
        "trace_counts=%s" % {event: sum(1 for t in result["trace"] if t.get("event") == event) for event in ("matched", "page_fetch", "no_match")},
        "output=%s" % (json.dumps(result["output"])[:1500] if result["output"] is not None else None),
        "stderr_tail=%s" % result["stderr"][-2500:],
    ]
    return "\n".join(lines)


def test_real_judge_requests_match_closed_operations_and_signed_models(tmp_path):
    """Every judge request matches one operation and uses only the pinned judge models."""

    document = scoring_input()
    result = run_real_judge(tmp_path, document)
    summary = describe(result)
    assert result["exit_code"] == 0 and result["output"] is not None, summary
    assert not result["rejected"], summary
    assert "failure" not in result["output"], summary
    policy_models = set(document["scorer_policy"]["judge_models"].values())
    models_used = {str(frame["model"]) for frame in result["frames"] if frame.get("model")}
    assert models_used <= policy_models, (models_used, policy_models, summary)
    # The shim refuses a request it cannot map client-side; the judge reports that inside its failure reasons.
    assert "no_matching_operation" not in result["stderr"] and "no_matching_operation" not in json.dumps(result["output"]), summary
    breakdowns = scoring.validate_breakdowns_for_item(result["output"]["breakdowns"], icp=document["icp"], companies=document["companies"])
    assert len(breakdowns) == len(document["companies"]), summary
