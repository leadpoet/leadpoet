"""Bounded private relay between the collection broker and the gVisor worker.

The trusted operator opens the two ends over authenticated SSH. The validator
sees the scenario and provider results. Only the gateway holds funding keys.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import threading

from lab_arena import collection_intake, contracts, deepline_catalog, lab_arena_checkpoint, operations, output, store
from lab_arena.collection_workflow import CHAMPION_REV

MAX_FRAME = 20 * 1024 * 1024
MAX_CALLS = 1000
MAX_TRACE_BYTES = 128 * 1024 * 1024
WIRE = "collection.research.v1"


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def send(stream, message):
    payload = encode(dict(message, wire=WIRE))
    if len(payload) > MAX_FRAME:
        raise ValueError("research frame exceeds limit")
    stream.write(payload + b"\n")
    stream.flush()


def receive(stream):
    line = stream.readline(MAX_FRAME + 2)
    if len(line) > MAX_FRAME + 1 or not line.endswith(b"\n"):
        raise ValueError("missing or oversized research frame")
    value = json.loads(line)
    if type(value) is not dict or value.pop("wire", None) != WIRE:
        raise ValueError("invalid research protocol")
    return value


class PipeAPI:
    """Credential-free API surface used by the existing validator socket."""
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.lock = threading.Lock()

    def _exchange(self, request, expected):
        with self.lock:
            send(self.writer, request)
            answer = receive(self.reader)
        if set(answer) != {"type", "document"} or answer["type"] != expected:
            raise ValueError("research broker refused request")
        return answer["document"]

    def provider(self, run_id, lease_token, frame):
        return self._exchange({"type": "provider", "frame": frame}, "provider_result")

    def quota_usage(self, run_id, lease_token, *, include_sourcing_cost=False):
        value = self._exchange({"type": "quota"}, "quota_result")
        if include_sourcing_cost:
            return value
        return {"schema_version": lab_arena_checkpoint.QUOTA_SNAPSHOT_SCHEMA_VERSION, "providers": value["providers"]}


def _usage(broker):
    j = broker.job
    usage = broker.usage()
    providers = {}
    for name, cap in contracts.DEEPLINE_BUDGET_ONLY_CALL_QUOTAS.items():
        usage_row = usage["providers"].get(name, {})
        used = usage_row.get("used", 0)
        # The count ceiling is advisory. Monetary admission always occurs first
        # at the gateway. Keep the shared v2 quota snapshot internally valid.
        limit = max(cap, used) if cap else 0
        providers[name] = dict(used=used, limit=limit, remaining=limit-used if limit else None,
                               inflight=usage_row.get("inflight", 0))
    return lab_arena_checkpoint.validate_quota_cost_snapshot({
        "schema_version": lab_arena_checkpoint.QUOTA_COST_SNAPSHOT_SCHEMA_VERSION,
        "providers": providers, "sourcing_cost": {
            "successful_microusd": usage.get("successful_microusd", 0), "settled_microusd": usage["settled_microusd"],
            "reserved_or_uncertain_microusd": usage["reserved_or_uncertain_microusd"],
            "success_unresolved_microusd": usage["reserved_or_uncertain_microusd"],
            "inflight_calls": usage["inflight_calls"], "success_unresolved_calls": usage["inflight_calls"],
            "admission_cap_microusd":j["budget_microusd"], "per_qualified_pair_cap_microusd": j["budget_microusd"],
        },
    })


def _worker_reply(job_id, sequence, op, body, *, code=200, actual=None):
    provider = op.split(".",1)[0]
    call = {"operation_id":op,"provider":provider,"action_sequence":sequence,"status":code,
            "outcome":"settled" if code==200 else "refused"}
    if actual is not None:
        call.update(actual_microusd=actual, funding_source="miner_key",
                    call_identity=contracts.document_hash([job_id,sequence,op]))
    return {"headers":{"content-type":"application/json"}, "status":code,
            "body_b64":base64.b64encode(encode(body)).decode(),"call":call}


def _catalog(snapshot, op, params):
    if op == "deepline.tools.get":
        return deepline_catalog.public_tool_definition(deepline_catalog.tool_entry(snapshot,params["tool"]))
    rows = [deepline_catalog.public_tool_definition(t, compact=True) for t in snapshot["tools"]]
    if op == "deepline.tools.search":
        terms = set(re.findall(r"[a-z0-9]+", params.get("query","").lower()))
        rows = sorted(rows,key=lambda t:-len(terms & set(re.findall(r"[a-z0-9]+",encode(t).decode().lower()))))
    start = params.get("offset",0)
    return {"tools":rows[start:start+params.get("limit",50)], "total":len(rows)}


def _public_call(frame, broker, trace):
    if not isinstance(frame,dict) or set(frame)!={"operation_id","parameters","action_sequence","timeout_ms"}:
        raise ValueError("malformed research provider frame")
    op, sequence = frame["operation_id"],frame["action_sequence"]
    if type(sequence) is not int or not 0<=sequence<MAX_CALLS or sequence in trace:
        raise ValueError("duplicate or out-of-range research action")
    j = broker.job
    if op not in {"openrouter.responses","openrouter.chat","deepline.execute",*operations.CATALOG_OPERATIONS}:
        return _worker_reply(j["job_id"],sequence,op,{"error":"provider_unavailable"},code=403)
    params = operations.validate_operation_request(op,frame["parameters"],deepline_catalog=broker.catalog,
                                                   allow_full_thinking=True)
    request_id=f"generate-research-{sequence}"
    if op in operations.CATALOG_OPERATIONS:
        body = _catalog(broker.catalog,op,params)
    elif op == "deepline.execute":
        body = broker.deepline(request_id,params)
    else:
        if params.get("model")!=j["model"]:
            raise ValueError("unassigned generator")
        body = broker.model_call(request_id,params,api=op.split(".")[1])
    # Preserve every exact provider-visible turn privately. Never send model
    # identifiers, prices, evaluator prompts or credentials to the judge.
    trace[sequence] = {"op":op,"request":params,"response":body}
    from lab_arena.provider_costs import openrouter_cost, deepline_cost
    cost = (openrouter_cost(body) if op.startswith("openrouter.") else
            deepline_cost(body) if op=="deepline.execute" else None)
    return _worker_reply(j["job_id"],sequence,op,body,actual=cost.microusd if cost else None)


def _content(parts):
    if isinstance(parts,str):
        return parts
    if not isinstance(parts,list) or any(not isinstance(p,dict) or not isinstance(p.get("text"),str) for p in parts):
        raise ValueError("unrepresentable research message")
    return "\n\n".join(p["text"] for p in parts)


def _response_messages(items):
    msgs=[]
    for it in items:
        kind = it.get("type","message")
        if kind=="function_call_output":
            msgs.append({"role":"tool","tool_call_id":it["call_id"],"content":_content(it["output"])})
        elif kind=="message" and it.get("role")!="assistant":
            msgs.append({"role":"system" if it["role"]=="developer" else it["role"],"content":_content(it["content"])})
        else:
            if not msgs or msgs[-1]["role"]!="assistant":
                msgs.append({"role":"assistant","content":""})
            turn=msgs[-1]
            if kind=="message" and it["role"]=="assistant":
                text=_content(it["content"])
                turn["content"] += ("\n\n" if turn["content"] else "")+text
            elif kind=="function_call":
                json.loads(it["arguments"])
                turn.setdefault("tool_calls",[]).append({"id":it["call_id"],"type":"function",
                    "function":{"name":it["name"],"arguments":it["arguments"]}})
            elif kind=="reasoning":
                exposed=_content(it.get("content") or [])
                if exposed:turn["reasoning_content"]=exposed
            else:
                raise ValueError("unsupported research action")
    return msgs


def decision_row(entry):
    """Keep one verified action under its original context, independent of sibling sessions."""
    req,resp=entry["request"],entry["response"]
    if entry["op"]=="openrouter.responses":
        if resp.get("status")!="completed":
            raise ValueError("research model turn incomplete")
        tools=[]
        for t in req.get("tools",[]):
            if t.get("type")!="function":
                raise ValueError("unsupported tool shape for training")
            tools.append({"type":"function","function":{k:v for k,v in t.items() if k!="type"}})
        messages=_response_messages(req["input"])
        if req.get("instructions"):
            messages.insert(0,{"role":"system","content":req["instructions"]})
        target,=_response_messages(resp["output"])
    else:
        choice,=resp["choices"]
        if choice.get("finish_reason") not in {"stop","tool_calls"}:
            raise ValueError("research model turn incomplete")
        messages=list(req["messages"])
        tools=req.get("tools",[])
        target={k:v for k,v in choice["message"].items() if k in {"content","role","tool_calls","reasoning_content","reasoning"}}
    if target.get("role")!="assistant" or not (target.get("content") or target.get("tool_calls")):
        raise ValueError("research action missing")
    last=len(messages)
    return {"messages":messages+[target],"tools":tools,"label_turns":[last],"no_loss_turns":[],
            "steps":[{"seq":1,"message":last}]}


class RemoteResearch:
    def __init__(self, open_channel, load_case, grade):
        self.open_channel,self.load_case,self.grade=open_channel,load_case,grade

    def __call__(self, job, broker):
        case=self.load_case(job)
        if job["harness_commit"]!=CHAMPION_REV or job["model"] not in collection_intake.MODELS:
            raise ValueError("research source or model differs from the frozen assignment")
        broker.catalog = case["deepline_catalog"]
        trace={}; trace_bytes=0
        with self.open_channel(job) as (reader, writer):
            send(writer,{"type":"research_start",**{k:job[k] for k in ("job_id","scenario_id","model","harness_commit")},
                         "icp":case["icp"],"evaluation_date":case["evaluation_date"],"deepline_catalog":broker.catalog,
                         "scorer_image_digest":case["reference"]["scorer_image_digest"]})
            for _ in range(4*MAX_CALLS):
                event=receive(reader)
                if event.get("type")=="quota":
                    send(writer,{"type":"quota_result","document":_usage(broker)})
                elif event.get("type")=="provider":
                    document=_public_call(event["frame"],broker,trace)
                    trace_bytes+=len(encode(event))+len(document["body_b64"])
                    if trace_bytes>MAX_TRACE_BYTES:
                        raise ValueError("research trace size limit")
                    send(writer,{"type":"provider_result","document":document})
                elif set(event)=={"type","job_id","companies"} and event["type"]=="research_result" and event["job_id"]==job["job_id"]:
                    companies=output.validate_companies(event["companies"],schema_version=case["icp"]["output_schema_version"])
                    break
                else:
                    raise ValueError("research worker did not deliver a result")
            else:
                raise ValueError("research event limit")
        rows=[]; steps=[]
        for seq,entry in sorted(trace.items()):
            if not entry["op"].startswith("openrouter."):
                continue
            try:
                r=decision_row(entry)
            except (ValueError,KeyError,TypeError,IndexError):
                rows.append(None)
                steps.append({"step":len(rows),"action":"Unsupported or incomplete provider turn.","evidence":""})
                continue
            target=r["messages"][-1]
            rows.append(dict(r,run=job["job_id"],source="locked_research",teacher=job["model"],
                             task=job["scenario_id"],action_sequence=seq))
            action=encode(target.get("tool_calls") or target["content"]).decode()
            evidence="\n".join(_content(m["content"]) for m in r["messages"][:-1] if m["role"]=="tool")
            steps.append({"step":len(rows),"action":action[:8192],"evidence":evidence[-8192:],
                          **({"reasoning":target["reasoning_content"][:8192]} if target.get("reasoning_content") else {})})
        if not rows:
            raise ValueError("research has no verified generator decisions")
        checked=self.grade(job,case,companies)
        if (checked["icp_sha256"]!=contracts.document_hash(case["icp"])
            or checked["output_sha256"]!=contracts.document_hash(companies)
            or not checked["checks"] or not all(type(v) is bool for v in checked["checks"].values())):
            raise ValueError("research judgment is not bound to the submitted artifacts")
        from lab_arena import training_capture
        checked["checks"].update(answered=True, shape=all(r is not None for r in rows),
                                 training_split=case.get("split")=="train", privacy=True)
        packet={"run_id":job["job_id"],"scenario_id":job["scenario_id"],"task":encode(case["icp"]).decode(),
                "steps":steps,"final_output":encode(companies).decode(),
                **{k:checked[k] for k in ("facts","reference_answer","grading_notes","checks")}}
        wire, redactions=training_capture.sanitize_training_value(
            {"provider_calls":[{"operation_id":x["op"],"request":{"outbound":{"body":x["request"]}},
                                "response":x["response"]} for _,x in sorted(trace.items())]}, aggregate_parts=True)
        result={"run":job["job_id"],"teacher":job["model"],"research_rows":rows, "capture":wire,
                "native_review_required":not checked["checks"].get("native_coverage",True)}
        # Apply the same privacy boundary to the derived judge packet, including
        # native verifier evidence. Retain no raw secret in the operator store.
        clean_packet, packet_redactions=training_capture.sanitize_training_value(packet)
        if redactions or packet_redactions:
            # Never retain unfiltered sensitive/opaque fields in a derived copy.
            result["research_rows"]=[]
            packet={**clean_packet,"checks":{**checked["checks"],"privacy":False},"steps":[],"facts":"",
                    "final_output":"","reference_answer":""}
        return result,packet


def export_research_rows(row, assessment, mode):
    if mode not in {"decisions","reasoning","hard_steps"} or row["run"]!=assessment["run_id"]:
        raise ValueError("invalid research export binding")
    if assessment["status"] not in {"accepted","salvaged"}:
        return []
    keep=assessment["keep_steps"]
    if type(keep) is not int or not 1<=keep<=len(row["research_rows"]):
        raise ValueError("invalid research export prefix")
    result=[]
    for n,segment in enumerate(row["research_rows"][:keep],1):
        messages=[]
        for i,message in enumerate(segment["messages"]):
            m=dict(message)
            if mode=="decisions" or mode=="hard_steps" and (i!=segment["label_turns"][0] or n not in assessment["hard_steps"]):
                m.pop("reasoning",None);m.pop("reasoning_content",None)
            messages.append(m)
        result.append(dict(segment,messages=messages))
    return result


class FrozenResearchCases:
    """Verified real ICP cases. Read private canonical source/label authorities once.

    An unknown new company is an uncovered claim. A repeated company name
    without identical effective evidence never supplies a native verdict.
    """
    def __init__(self, arena_store, objects, index):
        from lab_arena import training_reader
        self.fetch = lambda rid: training_reader.export_run(arena_store, objects, rid)
        if (type(index) is not dict or not 1<=len(index)<=1024
                or any(not re.fullmatch(r"research-[A-Za-z0-9_-]{1,90}",k) for k in index)):
            raise ValueError("invalid frozen research case index")
        for ref in index.values():
            if (type(ref) is not dict or set(ref)!={"run_id","example_hash","scorer_image_digest","scorer_policy_hash","split"}
                    or ref.get("split")!="train" or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", ref["run_id"])
                    or not all(re.fullmatch(r"sha256:[0-9a-f]{64}",ref[k]) for k in ("example_hash","scorer_image_digest","scorer_policy_hash"))):
                raise ValueError("unbound research reference")
        self.index = json.loads(encode(index))

    def load(self,job):
        from lab_arena import training_dataset
        ref=self.index[job["scenario_id"]]
        raw=self.fetch(ref["run_id"])
        rules=training_dataset.Rules(scorer_image_digest=ref["scorer_image_digest"],scorer_policy_hash=ref["scorer_policy_hash"])
        if (raw.get("run_id")!=ref["run_id"] or raw.get("example_hash")!=ref["example_hash"]
                or training_dataset.example_exclusion(raw,rules) or not raw.get("score_run_id")):
            raise ValueError("frozen research scorer or artifact identity differs")
        icp,source=training_dataset.agent_icp(raw)
        if source!="input_snapshot" or icp.get("output_schema_version")!="leadpoet.lab_arena.output.v6":
            raise ValueError("exact company-only research input is missing")
        catalog=(raw.get("configuration") or {}).get("deepline_catalog")
        if not isinstance(catalog,dict):
            raise ValueError("original research catalog unavailable")
        labels={}
        for row in training_dataset.records_from_example(raw):
            if row.get("source_evidence_error"):
                continue
            c=row["input"]["company"]
            if row["input"].get("source_evidence") and row["meta"]["score_run_id"]==raw["score_run_id"]:
                labels[contracts.document_hash(c)]={"company":c.get("company_name"),
                    "class":row["label"]["class"],"source_evidence":row["input"]["source_evidence"]}
        return {"icp":icp,"split":ref["split"],"evaluation_date":raw["evaluation_date"],"labels":labels,"reference":ref,
                "scored_run_id":raw["score_run_id"],"deepline_catalog":deepline_catalog.validate_catalog(catalog)}

    def grade(self,job,case,companies):
        matches=[case["labels"].get(contracts.document_hash(c)) for c in companies]
        uncovered=any(m is None for m in matches)
        rows=[m for m in matches if m is not None]
        evidence=encode([r["source_evidence"] for r in rows]).decode()
        if len(evidence)>65536:
            # A clipped quote is never proof. Preserve the row for review.
            evidence="Retained native evidence exceeds the judge packet limit."
            uncovered=True
        return {"icp_sha256":contracts.document_hash(case["icp"]),"output_sha256":contracts.document_hash(companies),
                "facts":evidence,"reference_answer":encode([{k:r[k] for k in ("company","class")} for r in rows]).decode(),
                "grading_notes":"Existing same-ICP/date native company judgments. Unmatched or unavailable evidence needs a fresh native judgment.",
                "checks":{"nonempty_output":bool(companies),"native_coverage":not uncovered,
                          "distinct_companies":len({str(c.get("company_website","")).lower() for c in companies})==len(companies),
                          "native_qualified":bool(rows) and all(r["class"]=="qualified" for r in rows)}}
