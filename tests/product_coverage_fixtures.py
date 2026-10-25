"""Native declared model receipts for offline protocol tests, not truth labels."""
from copy import deepcopy
import json

from qualification.scoring import product_coverage as pc
from qualification.scoring import company_evidence_investigator as inv


def native_product_post(post):
    """Supply offline model wire receipts; run the real host plan and proof checks.

    The investigator mock still decides each finding's status and citation. This
    adapter only gives that decision the two native protocol envelopes required
    by product review. It cannot grant authority to an unbound source quote.
    """
    async def wrapped(session, url, *, headers, payload):
        tool_names = {
            tool.get("function", {}).get("name")
            for tool in payload.get("tools", [])
        }
        if "submit_product_plan" in tool_names:
            criterion = json.loads(payload["messages"][1]["content"])[
                "requested_product_service"
            ]
            plan_json = json.dumps({"root": "p0", "nodes": [{
                "id": "p0", "op": "leaf", "text": criterion, "children": [],
            }]})
            return 200, {"choices": [{"message": {"tool_calls": [{
                "id": "offline-product-plan", "type": "function",
                "function": {"name": "submit_product_plan", "arguments": json.dumps({
                    "plan_json": plan_json,
                })},
            }]}}]}

        status, body = await post(session, url, headers=headers, payload=payload)
        if status != 200 or not isinstance(body, dict):
            return status, body
        document = None
        for message in payload.get("messages", []):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                candidate = json.loads(content.split("\n", 1)[-1])
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict) and "server_frozen_product_plan" in candidate:
                document = candidate
                break
        if document is None:
            return status, body
        frozen = document["server_frozen_product_plan"]
        copied = deepcopy(body)
        for choice in copied.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls", []):
                function = call.get("function", {})
                if function.get("name") != "submit_findings":
                    continue
                try:
                    arguments = json.loads(function["arguments"])
                except (KeyError, TypeError, ValueError):
                    continue
                for finding in arguments.get("findings", []):
                    if (finding.get("target") != "industry"
                            or finding.get("product_coverage") is not None):
                        continue
                    finding_status = finding.get("status", "UNPROVEN")
                    evidence = ([{"url": finding.get("evidence_url", ""),
                                  "quote": finding.get("evidence_quote", "")}]
                                if finding_status in {"VERIFIED", "CONTRADICTED"}
                                else [])
                    finding["product_coverage"] = json.dumps({
                        "plan_id": frozen["plan_id"],
                        "claims": [{
                            "id": node["id"], "status": finding_status,
                            "activity_role": finding.get("activity_role", "unresolved"),
                            "evidence": evidence,
                        } for node in frozen["nodes"] if node["op"] == "leaf"],
                    })
                function["arguments"] = json.dumps(arguments)
        return status, copied

    return wrapped


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


def validated_industry_finding(
    finding, criterion, *, fetched_pages, first_party_domains, identity_names,
    fetched_final_urls=None, identity_anchor=None,
):
    """Validate a mocked product judgment with the production host checks."""
    plan = trusted_plan(criterion)
    status = finding["status"]
    evidence = ([{"url": finding.get("evidence_url", ""),
                  "quote": finding.get("evidence_quote", "")}]
                if status in {"VERIFIED", "CONTRADICTED"} else [])
    claim = dict(finding)
    claim["product_coverage"] = json.dumps({
        "plan_id": plan.plan_id,
        "claims": [{
            "id": node["id"], "status": status,
            "activity_role": finding.get("activity_role", "unresolved"),
            "evidence": evidence,
        } for node in plan.nodes if node["op"] == "leaf"],
    })
    validated = inv._validated_findings(
        {"findings": [claim]}, targets=("industry",),
        fetched_pages=fetched_pages,
        fetched_final_urls=fetched_final_urls or {url: url for url in fetched_pages},
        first_party_domains=first_party_domains,
        identity_names=identity_names, identity_anchor=identity_anchor or {},
        requested_product_service=criterion, product_plan=plan,
    )
    assert validated is not None
    return validated["industry"]
