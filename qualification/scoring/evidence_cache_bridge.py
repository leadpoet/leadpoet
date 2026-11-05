"""Connect prospective scorer evidence checks to existing Arena authority."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
from typing import Any, Mapping

from lab_arena.contracts import document_hash
from qualification.scoring.signal_judgment_cache import (
    SignalJudgmentCacheError, invoke_exchange, reusable_terminal_judgment,
)

POLICY = "evidence_judgment_v2"


def enabled() -> bool:
    return os.environ.get("ARENA_EVIDENCE_JUDGMENT_POLICY", "") == POLICY


@contextmanager
def install_company_fit_exchange(exchange):
    """The surrounding scorer exchange supplies the original company index."""
    from qualification.scoring.company_fit_evidence_reuse import install_fit_evidence_reuse
    async def fit_exchange(request):
        payload = dict(request)
        payload.update(evidence_judgment_policy=POLICY,
                       checkpoint_kind="company_fit_evidence_v2", signal_index=-1)
        manifest = payload.pop("source_manifest", None)
        snapshot_hash = payload.pop("source_snapshot_hash", None)
        if manifest is not None:
            payload["evidence_input"] = manifest
            payload["evidence_hash"] = snapshot_hash
        return await invoke_exchange(exchange, payload)
    with install_fit_evidence_reuse(fit_exchange):
        yield


class SignalEvidenceExchange:
    """One signal's fresh-source lookup and final terminal-result admission."""

    def __init__(self, exchange, request, *, expected_urls, expected_index,
                 expected_evidence_type, allowed_context_urls=()):
        self.exchange = exchange
        self.base = deepcopy(request)
        self.base.update(evidence_judgment_policy=POLICY,
                         judgment_kind="signal_evidence_v2")
        self.expected = {"expected_urls": expected_urls, "expected_index": expected_index,
                         "expected_evidence_type": expected_evidence_type,
                         "allowed_context_urls": allowed_context_urls}
        self.request = None
        self.reference = None
        self.hit = None
        self.pending = None

    def _envelope(self, value):
        if (not isinstance(value, Mapping)
                or value.get("schema_version") != "leadpoet.lab_arena.signal_evidence.v2"
                or not isinstance(value.get("verifier_result"), Mapping)
                or not isinstance(value.get("claim"), Mapping)
                or value.get("evidence_input") != self.request["evidence_input"]
                or not reusable_terminal_judgment(value.get("raw_judgment"), **self.expected)):
            raise SignalJudgmentCacheError("evidence signal authority is invalid")
        return deepcopy(dict(value))

    async def __call__(self, callback_request):
        if callback_request.get("operation") == "complete":
            if self.request is not None and callback_request.get("evidence_input") == self.request["evidence_input"]:
                self.pending = deepcopy(callback_request.get("judgment"))
            return {"status": "captured"}
        if callback_request.get("operation") != "lookup" or self.request is not None:
            raise SignalJudgmentCacheError("evidence signal lookup lifecycle is invalid")
        evidence, claim = callback_request["evidence_input"], callback_request["claim"]
        self.request = {**self.base, "evidence_input": deepcopy(evidence),
                        "evidence_hash": document_hash(evidence), "claim": deepcopy(claim)}
        candidates = await invoke_exchange(self.exchange, {**self.request, "operation": "candidates"})
        rows = candidates.get("candidates") if isinstance(candidates, Mapping) else None
        if not isinstance(rows, list) or len(rows) > 8:
            raise SignalJudgmentCacheError("evidence signal candidates are invalid")
        from qualification.scoring.claim_equivalence import verify_claim_equivalence
        model = evidence.get("model")
        for candidate in rows:
            envelope = self._envelope(candidate.get("judgment"))
            proof = None
            if envelope["claim"] != claim:
                proof = await verify_claim_equivalence(envelope["claim"], claim, model=model)
                if proof is None:
                    continue
            request = {**self.request, "operation": "claim",
                       "candidate_reference": candidate.get("reference")}
            if proof is not None:
                request["equivalence_receipt"] = proof
            result = await invoke_exchange(self.exchange, request)
            if result.get("status") == "hit":
                self.reference = result.get("reference")
                self.hit = self._envelope(result.get("judgment"))
                return {"status": "hit", "judgment": deepcopy(self.hit["verifier_result"])}
            if result.get("status") not in {"miss", "unavailable"}:
                raise SignalJudgmentCacheError("evidence candidate claim is invalid")
        # Exact variants share the existing reservation. An in-flight owner
        # may have completed while the candidate list was read.
        result = await invoke_exchange(self.exchange, {**self.request, "operation": "claim"})
        self.reference = result.get("reference")
        if not isinstance(self.reference, Mapping):
            raise SignalJudgmentCacheError("evidence signal reference is missing")
        if result.get("status") == "hit":
            self.hit = self._envelope(result.get("judgment"))
            if self.hit["claim"] != claim:
                raise SignalJudgmentCacheError("exact evidence variant claim differs")
            return {"status": "hit", "judgment": deepcopy(self.hit["verifier_result"])}
        if result.get("status") != "miss":
            raise SignalJudgmentCacheError("evidence signal claim is invalid")
        return {"status": "miss"}

    async def finish(self, raw_judgment):
        if self.reference is None:
            return None  # Unsupported source path kept the native judge.
        if self.hit is not None:
            if raw_judgment != self.hit["raw_judgment"]:
                raise SignalJudgmentCacheError("reused verifier changed the terminal signal")
            return deepcopy(self.reference)
        terminal = self.pending is not None and reusable_terminal_judgment(raw_judgment, **self.expected)
        envelope = ({"schema_version": "leadpoet.lab_arena.signal_evidence.v2",
                     "raw_judgment": deepcopy(raw_judgment), "verifier_result": self.pending,
                     "evidence_input": self.request["evidence_input"], "claim": self.request["claim"]}
                    if terminal else None)
        result = await invoke_exchange(self.exchange, {**self.request, "operation": "complete",
                                      "reference": self.reference, "judgment": envelope})
        if (result.get("status") not in ({"stored", "existing"} if terminal else {"released"})
                or result.get("reference") != self.reference):
            raise SignalJudgmentCacheError("evidence signal completion is invalid")
        return deepcopy(self.reference) if terminal else None

    async def release(self):
        if self.reference is not None and self.hit is None:
            result = await invoke_exchange(self.exchange, {**self.request, "operation": "complete",
                                          "reference": self.reference, "judgment": None})
            if result.get("status") != "released":
                raise SignalJudgmentCacheError("evidence signal release is invalid")
