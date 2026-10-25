"""Native declared model receipts for offline protocol tests, not truth labels."""
import json

from qualification.scoring import product_coverage as pc
from qualification.scoring import company_evidence_investigator as inv


def trusted_plan(criterion, nodes=None):
    nodes = nodes or [{"id": "p0", "span": criterion, "op": "leaf", "children": []}]
    by_id = {n["id"]: n for n in nodes}
    converted = []

    def visit(key):
        node = by_id[key]
        if node["op"] == "example":
            return None
        children = [child for c in node["children"] if (child := visit(c)) is not None]
        if node["op"] != "leaf" and len(children) == 1:
            return children[0]
        converted.append({"id": key, "text": node.get("text", node.get("span", "")),
                          "op": node["op"], "children": children})
        return key

    root = visit("p0")
    plan = pc.validate_semantic_plan({"root": root, "nodes": converted},
                                    criterion=criterion, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL)
    pc._store_trusted_plan(plan)
    return plan


def coverage(criterion, url, quote, *, nodes=None, unproven=()):
    plan = trusted_plan(criterion, nodes)
    unproven_texts = {n.get("text", n.get("span")) for n in (nodes or []) if n["id"] in unproven}
    if nodes is None and "p0" in unproven:
        unproven_texts.add(criterion)
    return json.dumps({"plan_id": plan.plan_id, "claims": [
        {"id": n["id"], "status": "UNPROVEN" if n["text"] in unproven_texts else "VERIFIED",
         "activity_role": "supplier_operator",
         "evidence": [] if n["text"] in unproven_texts else [{"url": url, "quote": quote}]}
        for n in plan.nodes if n["op"] == "leaf"
    ]}, separators=(",", ":"))
