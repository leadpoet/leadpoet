"""Immutable per-turn records shared by synthetic collection runners."""

import hashlib
import json
import re

from lab_arena import collection_judge as judge

MAX_EVENTS = 4096
MAX_EVENT_BYTES = 128 * 1024 * 1024


class WorkflowJournal:
    def __init__(self, objects, batch, run_id):
        if not all(
            isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", s)
            for s in (batch, run_id)
        ):
            raise ValueError("invalid collection journal coordinates")
        self.batch, self.run_id = batch, run_id
        self.objects = objects
        self.path = f"collection/{batch}/{run_id}/events/"
        self.hashes = []

    def save(self, kind, value):
        """Publish a private recovery pointer before later grading or billing steps."""
        from lab_arena.collection_workflow import store_artifact

        ref = store_artifact(self.objects, self.batch, self.run_id, kind, value)
        data = judge._json(ref).encode()
        key = f"collection/{self.batch}/{self.run_id}/saved/{kind}.json"
        self.objects.put(key, data)
        if self.objects.get_bounded(key, len(data)) != data:
            raise ValueError("private recovery pointer failed readback")
        return ref

    def append(self, event):
        index = len(self.hashes)
        if index != event["index"] or index >= MAX_EVENTS:
            raise ValueError("nonsequential collection event")
        data = judge._json(event).encode()
        if len(data) > MAX_EVENT_BYTES:
            raise ValueError("collection event too large")
        ref = self.path + f"{index:05d}.json"
        self.objects.put(ref, data)
        if self.objects.get_bounded(ref, len(data)) != data:
            raise ValueError("stored collection event failed readback")
        self.hashes.append("sha256:" + hashlib.sha256(data).hexdigest())

    def read(self, expected=None):
        """Stream verified event bytes; a hard stop may have no terminal index."""
        if expected is not None and (
            not isinstance(expected, list) or len(expected) > MAX_EVENTS
        ):
            raise ValueError("invalid collection event index")
        for i in range(MAX_EVENTS if expected is None else len(expected)):
            raw = self.objects.get_optional_bounded(
                self.path + f"{i:05d}.json", MAX_EVENT_BYTES
            )
            if raw is None and expected is None:
                break
            if raw is None or json.loads(raw)["index"] != i:
                raise ValueError("run journal is missing or changed")
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            if expected is not None and digest != expected[i]:
                raise ValueError("run journal is missing or changed")
            yield raw, digest

    def finish(self, status):
        """Anchor a normal return or trapped exception; a process kill leaves only its prefix."""
        data = judge._json({"status": status, "events": self.hashes}).encode()
        ref = self.path + "index.json"
        self.objects.put(ref, data)
        if self.objects.get_bounded(ref, len(data)) != data:
            raise ValueError("collection journal index failed readback")

    def comparison_index(
        self, *, complete, model=None, manifest_events=None, generator=None, verify_terminal=False, checkpoint=None
    ):
        """Read the original private journal before signing its manifest or failed prefix."""
        raw = self.objects.get_optional_bounded(self.path + "index.json", 512 * 1024)
        index = json.loads(raw) if raw is not None else None
        if index is not None and (
            set(index) != {"events", "status"}
            or index["status"] not in {"completed", "interrupted"}
            or not isinstance(index["events"], list)
        ):
            raise ValueError("invalid comparison journal index")
        if complete and (
            index is None
            or index["status"] != "completed"
            or manifest_events != index["events"]
        ):
            raise ValueError(
                "comparison manifest differs from its complete event index"
            )
        expected = None if index is None else index["events"]
        hashes = self.transcript_hashes(
            expected, model=model, generator=generator if complete else None,
            verify_terminal=verify_terminal, checkpoint=checkpoint,
        )
        return {
            "status": index["status"] if index is not None else "unsealed",
            "events": hashes,
        }

    def transcript_hashes(self, expected, *, model, generator=None, verify_terminal=False, checkpoint=None):
        """Check private events against the original assistant-visible transcript."""
        first = last = None
        hashes = []
        transcript = []
        restored = False
        for data, digest in self.read(expected):
            event = json.loads(data)
            if not hashes:
                first = event
            if generator is not None:
                kind = event["kind"]
                if kind == "start":
                    if checkpoint is not None and any(event[key] != checkpoint[key] for key in ("messages","tools")):
                        raise ValueError("continuation start differs from the verified student transcript")
                    transcript.extend(event["messages"])
                elif kind == "before" and checkpoint is not None and not restored:
                    if any(event[key] != checkpoint[key] for key in ("files","state","log")):
                        raise ValueError("continuation world differs from the verified student checkpoint")
                    restored = True
                elif kind == "reply":
                    reply = event["message"]
                    msg = {"role": reply["role"], "content": reply.get("content") or ""}
                    msg.update(
                        {
                            k: reply[k]
                            for k in ("tool_calls", "reasoning", "reasoning_content")
                            if reply.get(k)
                        }
                    )
                    transcript.append(msg)
                elif kind == "tool":
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_call_id": event["call_id"],
                            "content": event["output"][:20000],
                        }
                    )
            last = event["kind"]
            hashes.append(digest)
        if hashes and (first["kind"] != "start" or first["model"] != model):
            raise ValueError("comparison generator identity differs from its journal")
        if generator is not None and (
            not hashes
            or last != "finish"
            or generator is None
            or transcript != generator["messages"]
            or first["tools"] != generator["tools"]
        ):
            raise ValueError(
                "complete comparison journal differs from generator evidence"
            )
        if verify_terminal:
            evidence = generator["evidence"]
            # The original finish is stored before scoring. Bind the attested
            # terminal state and the pinned code check embedded in that journal.
            if (last != "finish" or (checkpoint is not None and not restored)
                    or event["finish"] != evidence["finish"] or event["code_checks"] != evidence["grade"]
                    or any(event[k] != evidence["out"][k] for k in ("log","state","answer"))
                    or event["files"] != {**evidence["initial_files"], **evidence["out"]["files"]}):
                raise ValueError("terminal evidence differs from the original completed journal")
        return hashes
