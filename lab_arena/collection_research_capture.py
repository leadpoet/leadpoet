"""Rights-scoped, private research journal for the existing authenticated worker relay."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from lab_arena import (
    contracts,
    deepline_catalog,
    training_capture,
    training_observations,
)
from lab_arena.web_egress import WebEgressError, WebEgressServer

SCOPE = "licensed_research"
REQUIRED_APPROVALS = {
    "delegated_funding",
    "provider_zero_retention",
    "source_reuse",
    "personal_data",
}
PASSIVE = {"firecrawl_scrape", "exa_search", "contextdev_post_web_search"}


def _host(url):
    deepline_catalog._public_url(url)
    return urlsplit(url).hostname


class ResearchWebEgress(WebEgressServer):
    """Keep the canonical DNS/SNI/public destination checks for each licensed HTTPS host."""

    def __init__(self, *args, allowed_hosts, expires_at, **kwargs):
        self._allowed_hosts = frozenset(allowed_hosts)
        self._expires_at = expires_at
        super().__init__(*args, **kwargs)

    def _resolved_ip(self, host, port):
        if (
            host not in self._allowed_hosts
            or port != 443
            or time.time() >= self._expires_at
        ):
            raise WebEgressError("destination outside licensed research scope")
        return super()._resolved_ip(host, port)


class ResearchGrant:
    def __init__(self, document, pinned_sha256):
        self.sha256 = contracts.document_hash(document)
        if (
            self.sha256 != pinned_sha256
            or set(document)
            != {
                "schema",
                "approvals",
                "batch_id",
                "registration_id",
                "miner_hotkey",
                "owner_coldkey",
                "terms_sha256",
                "scenario_id",
                "icp_sha256",
                "model",
                "expires_at",
                "hosts",
                "deepline_requests",
            }
            or document["schema"] != "collection.research-grant.v1"
        ):
            raise ValueError("research approval differs from deployment pin")
        for field in ("owner_coldkey", "miner_hotkey"):
            contracts.require_hotkey(document[field], field)
        if (
            str(UUID(document["registration_id"])) != document["registration_id"]
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", document["batch_id"])
            or not re.fullmatch(
                r"research-[A-Za-z0-9_-]{1,90}", document["scenario_id"]
            )
        ):
            raise ValueError("invalid licensed research identity")
        approvals = document["approvals"]
        if (
            set(approvals) != REQUIRED_APPROVALS
            or not all(contracts.SHA256_RE.fullmatch(v) for v in approvals.values())
            or not all(
                contracts.SHA256_RE.fullmatch(document[k])
                for k in ("terms_sha256", "icp_sha256")
            )
            or type(document["expires_at"]) is not int
        ):
            raise ValueError("missing delegated funding, retention or source approval")
        if (
            not isinstance(document["hosts"], list)
            or not 1 <= len(document["hosts"]) <= 256
        ):
            raise ValueError("invalid research host list")
        if any(
            not isinstance(h, str) or _host("https://" + h + "/") != h
            for h in document["hosts"]
        ):
            raise ValueError("invalid licensed research host")
        if (
            not isinstance(document["deepline_requests"], list)
            or len(document["deepline_requests"]) > 1000
            or not all(
                contracts.SHA256_RE.fullmatch(s) for s in document["deepline_requests"]
            )
        ):
            raise ValueError("invalid licensed provider request list")
        self.document = json.loads(contracts.canonical_json(document))

    @classmethod
    def deployed(cls):
        from scripts.collection_review import read_file

        path = os.environ.get("LAB_ARENA_COLLECTION_RESEARCH_GRANT_FILE")
        pin = os.environ.get("LAB_ARENA_COLLECTION_RESEARCH_GRANT_SHA256")
        if not path or not pin:
            raise ValueError("licensed research requires a deployment-approved grant")
        return cls(json.loads(read_file(Path(path), maximum=128 * 1024)), pin)

    def active(self):
        if time.time() >= self.document["expires_at"]:
            raise ValueError("research approval expired")

    def bind(self, job, case, objects):
        self.active()
        if (
            job.get("task_scope") != SCOPE
            or not job.get("deepline_key_id")
            or any(
                job.get(k) != self.document[k]
                for k in (
                    "batch_id",
                    "registration_id",
                    "miner_hotkey",
                    "owner_coldkey",
                    "terms_sha256",
                    "scenario_id",
                    "model",
                )
            )
            or contracts.document_hash(case["icp"]) != self.document["icp_sha256"]
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job["batch_id"])
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job["job_id"])
        ):
            raise ValueError(
                "research grant does not cover this funding owner and input"
            )
        return ResearchJournal(self, job, objects)

    def provider_request(self, op, params):
        self.active()
        if op == "deepline.execute":
            if (
                params.get("tool") not in PASSIVE
                or contracts.document_hash(params)
                not in self.document["deepline_requests"]
            ):
                raise ValueError("unapproved research tool request")
            if params["tool"] == "firecrawl_scrape":
                payload = params["payload"]
                if (
                    set(payload) - {"url", "formats", "onlyMainContent"}
                    or _host(payload["url"]) not in self.document["hosts"]
                ):
                    raise ValueError("unlicensed scrape destination")
        if op.startswith("openrouter.") and any(
            t.get("type") != "function" for t in params.get("tools", [])
        ):
            raise ValueError("hosted search is outside this grant")


class ResearchJournal:
    def __init__(self, grant, job, objects):
        self.grant, self.job, self.objects = grant, job, objects
        self.hashes = []
        self.complete = True
        self.last_agent_id = -1
        self.bytes = 0

    def append(self, kind, sequence, data):
        self.grant.active()
        clean, redactions = training_capture.sanitize_training_value(data)
        if redactions:
            self.complete = False
            raise ValueError("sensitive research evidence refused")
        event = {
            "schema": "collection.research-observation.v1",
            "grant_sha256": self.grant.sha256,
            "batch_id": self.job["batch_id"],
            "job_id": self.job["job_id"],
            "scenario_id": self.job["scenario_id"],
            "authority": kind,
            "sequence": sequence,
            "index": len(self.hashes),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "data": clean,
        }
        raw = contracts.canonical_json(event).encode()
        if self.bytes + len(raw) > 128 * 1024 * 1024 or len(self.hashes) >= 8192:
            raise ValueError("research journal capacity reached")
        key = f"collection/{self.job['batch_id']}/{self.job['job_id']}/research-events/{len(self.hashes):05d}.json"
        self.objects.put(key, raw)
        if self.objects.get_bounded(key, len(raw)) != raw:
            raise ValueError("research event upload did not verify")
        self.hashes.append(contracts.hash_bytes(raw))
        self.bytes += len(raw)

    def observation(self, row):
        row = training_observations.clean_agent_observation(row)
        if row["kind"] not in {
            "web_fetch",
            "web_fetch_binary",
            "http_response_metadata",
            "capture_failure",
        }:
            raise ValueError("unexpected research observation")
        if row["id"] <= self.last_agent_id or row["privacy"]["redacted"]:
            raise ValueError("redacted or duplicate research observation")
        data = row["data"]
        if row["kind"] == "capture_failure":
            self.complete = False
        else:
            urls = [data["url"]] + [
                data[k] for k in ("final_url", "redirect_url") if data.get(k)
            ]
            if any(_host(url) not in self.grant.document["hosts"] for url in urls):
                raise ValueError("unlicensed research source")
            if row["kind"] == "http_response_metadata" or not (
                data.get("capture") or {}
            ).get("complete"):
                self.complete = False
            if (
                row["kind"] in {"web_fetch", "web_fetch_binary"}
                and row["after_action_sequence"] is None
            ):
                self.complete = False
        self.append("sandbox_http", row["after_action_sequence"], row)
        self.last_agent_id = row["id"]


class ResearchObservations(training_observations.Buffer):
    """Relay the existing sanitized host-side observation at the point of capture."""

    def __init__(self, publish):
        super().__init__()
        self.publish = publish

    def append_agent_observation(self, value, secrets=()):
        result = super().append_agent_observation(value, secrets)
        if result is None or value["kind"] not in {
            "web_fetch",
            "web_fetch_binary",
            "http_response_metadata",
            "capture_failure",
        }:
            return result
        # Keep the previous body reachable even if the process stops before the
        # terminal result. The generic buffer interns identical bodies by id.
        row = self.agent_observations[result]
        if row["data"].get("body_ref") is not None:
            original = self.agent_observations[row["data"]["body_ref"]]
            field = "body" if row["kind"] == "web_fetch" else "body_b64"
            row = {
                **row,
                "data": {
                    **row["data"],
                    field: original["data"][field],
                    "body_ref": None,
                },
            }
        try:
            self.publish(row)
        except Exception:
            self.failures.add("capture_incomplete")
            raise
        return result
