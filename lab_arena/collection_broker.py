"""Meter calls from the pinned collection worker using the job owner's credentials."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from lab_arena import broker, contracts, operations, provider_costs, store
from lab_arena.lead_orders.broker import deepline_cost_microusd

store.FUNCTION_SIGNATURES.update({
    "collection_job_usage_v1": (("p_job","uuid"), ("p_lease","text")),
    "collection_reserve_call_v1": (("p_job","uuid"),("p_lease","text"),("p_request","text"),
        ("p_provider","text"),("p_hash","text"),("p_amount","bigint")),
    "collection_record_call_v1": (("p_job","uuid"),("p_lease","text"),("p_request","text"),
        ("p_hash","text"),("p_provider_id","text"),("p_cost","bigint"),("p_status","integer")),
})


class CollectionCallError(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CollectionBroker:
    def __init__(self, intake, database, transport, job, pricing, *, routes, catalog=None, judge_key=None):
        self.intake = intake
        self.database = database
        self.transport = transport
        self.job = job
        self.pricing = pricing
        self.routes = routes
        self.catalog = catalog
        self.judge_key = judge_key

    def _rpc(self, name, request_id, **params):
        return self.database.rpc(name, {"p_job": self.job["job_id"],
            "p_lease": store.hash_lease_token(self.job["lease_token"]),
            "p_request": request_id, **params})

    def usage(self):
        return self.database.rpc("collection_job_usage_v1", {"p_job":self.job["job_id"],
            "p_lease":store.hash_lease_token(self.job["lease_token"])})

    def _call(self, request_id, provider, url, content, reserve, *, model=None, timeout=120):
        secret = (self.judge_key if provider == "judge" else
                  self.intake.credential(self.job["job_id"], self.job["lease_token"], provider))
        if not secret or secret.encode() in content:
            raise CollectionCallError("invalid provider credential binding")
        admitted = self._rpc("collection_reserve_call_v1", request_id, p_provider=provider,
                             p_amount=max(1, reserve), p_hash=digest(content))
        if admitted["status"] != "reserved":
            raise CollectionCallError(admitted["status"])
        try:
            response = self.transport.send(method="POST", url=url,
                headers={"authorization":"Bearer " + secret, "content-type":"application/json"},
                body=content, timeout_seconds=timeout)
        except broker.ProviderTransportError:
            self._rpc("collection_record_call_v1", request_id, p_cost=None, p_status=0,
                      p_provider_id=None, p_hash=digest(b""))
            raise CollectionCallError("provider transport failed; reservation retained") from None
        try:
            body = json.loads(response.body)
            if not isinstance(body, dict):
                body = {}
        except (ValueError, UnicodeError):
            body = {}
        if provider in {"judge", "openrouter"}:
            cost = provider_costs.openrouter_cost(body)
            cost = None if cost is None else cost.microusd
            upstream_id = body.get("id")
        else:
            cost = deepline_cost_microusd(response.body)
            upstream_id = (body.get("request_id") or response.headers.get("x-request-id"))
        if not isinstance(upstream_id, str) or len(upstream_id) > 256 or secret in upstream_id:
            upstream_id = None
        self._rpc("collection_record_call_v1", request_id, p_hash=digest(response.body),
                  p_cost=cost, p_status=response.status, p_provider_id=upstream_id)
        if response.status != 200 or cost is None or broker._response_contains_credential(response, secret):
            raise CollectionCallError("provider reply unavailable or unverified")
        if model and body.get("model") != model:
            raise CollectionCallError("assigned provider model did not serve the response")
        return body

    def model_call(self, request_id, parameters, *, judge=None, api="chat"):
        model = self.job["model"] if judge is None else judge
        if judge is not None and judge not in {"anthropic/claude-fable-5.2","openai/gpt-6.2-astra"}:
            raise CollectionCallError("unknown collection judge")
        if parameters.get("model") != model or api not in {"chat","responses"}:
            raise CollectionCallError("wrong collection assignment")
        token_field = "max_tokens" if api == "chat" else "max_output_tokens"
        tokens = parameters.get(token_field)
        if type(tokens) is not int or not 1 <= tokens <= (400 if judge else 32768):
            raise CollectionCallError("invalid output allowance")
        allowed = {"model",token_field,"reasoning","tools","tool_choice","parallel_tool_calls"}
        allowed |= {"messages","response_format"} if api == "chat" else {"input","instructions","text","include"}
        if set(parameters) - allowed:
            raise CollectionCallError("unsupported model request fields")
        if any(t.get("type") != "function" for t in parameters.get("tools", [])):
            raise CollectionCallError("unbudgeted hosted tool")
        route = self.routes[model]
        price = self.pricing["models"][model]
        policy = {"only":[route],"zdr":True,"data_collection":"deny","allow_fallbacks":False,
                  "max_price":{k:float(Decimal(price[k])*(1 if k=="request" else 1_000_000))
                               for k in ("prompt","completion","request")}}
        if not judge:
            policy["enforce_distillable_text"] = True
        body = dict(parameters, provider=policy)
        if api == "chat":
            body["usage"] = {"include":True}
        reserve = broker.max_openrouter_cost_microusd(self.pricing, model, body, max_output_tokens=tokens)
        path = "chat/completions" if api == "chat" else "responses"
        return self._call(request_id,"judge" if judge else "openrouter","https://openrouter.ai/api/v1/"+path,
                          contracts.canonical_json(body).encode(),reserve,model=model)

    def judge(self, *, request_id, model, messages, max_tokens):
        response = self.model_call(request_id,{"model":model,"messages":messages,"max_tokens":max_tokens,
                "response_format":{"type":"json_object"}},judge=model)
        choice, = response["choices"]
        if choice.get("finish_reason") != "stop":
            return ""  # Truncation goes through the single invalid-verdict retry.
        return choice["message"]["content"]

    def deepline(self, request_id, parameters):
        if self.catalog is None:
            raise CollectionCallError("missing frozen Deepline catalog")
        from lab_arena import deepline_catalog
        if deepline_catalog.is_direct_inference_tool(self.catalog, parameters["tool"]):
            raise CollectionCallError("alternate inference is ineligible")
        outbound = operations.build_outbound_request("deepline.execute",parameters,deepline_catalog=self.catalog)
        price = provider_costs.deepline_reservation_cost(parameters, catalog_entry=deepline_catalog.tool_entry(
            self.catalog, parameters["tool"])) or provider_costs.deepline_reservation_cost(parameters)
        if price is None:
            raise CollectionCallError("unpriced Deepline operation")
        return self._call(request_id,"deepline",outbound.url,outbound.body,price.microusd,timeout=240)
