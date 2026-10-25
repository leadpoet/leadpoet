"""Declared model responses for protocol tests; not semantic truth labels."""
import json


def coverage(criterion, url, quote, *, nodes=None, unproven=()):
    nodes = nodes or [{"id": "p0", "span": criterion, "op": "leaf", "children": []}]
    return json.dumps({"nodes": nodes, "claims": [
        {"id": n["id"], "status": "UNPROVEN" if n["id"] in unproven else "VERIFIED",
         "activity_role": "supplier_operator",
         "evidence": [] if n["id"] in unproven else [{"url": url, "quote": quote}]}
        for n in nodes if n["op"] == "leaf"
    ]}, separators=(",", ":"))
