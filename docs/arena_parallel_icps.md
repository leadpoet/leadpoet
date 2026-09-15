# Arena parallel ICP execution

Arena can run the twenty ICPs for one model in parallel. The validator uses
one native host route and one route for each verified Webshare proxy. The
change affects execution speed and public web egress. It does not change the
model output contract, scoring, paid provider broker, provider budgets,
validator identity, or wallet use.

## Validator configuration

Use indexed settings in the validator environment:

```text
LAB_ARENA_WEBSHARE_PROXY_1=https://user:password@proxy.example.com:443
LAB_ARENA_WEBSHARE_PROXY_2=https://user:password@proxy.example.net:443
```

Indices must be positive. Gaps are permitted. Each configured URL must be
valid and each route must pass an authenticated TLS CONNECT check. Startup
also checks the public exit IP for every proxy and the native host. All exit
IPs must be different. One invalid route or duplicate exit IP stops startup;
the validator does not reduce its capacity silently.

`LAB_ARENA_PROXY_ENV_FILE` can name an absolute, private, owned validator env
file. This permits reuse of existing indexed
`QUALIFICATION_WEBSHARE_PROXY_N` entries. The loader reads recognized indexed
proxy settings only. It does not import wallet keys, provider keys, or other
secrets from that file. Direct `LAB_ARENA_WEBSHARE_PROXY_N` settings are the
standard configuration for new installations.

The canonical validator restart resolves this optional file while it still
runs as the operator that owns it. Before cutover, it creates a mode `0600`,
root-owned service environment with compact
`LAB_ARENA_WEBSHARE_PROXY_N` entries. The service file does not retain the
secondary path or legacy proxy aliases. Each restart regenerates this snapshot,
so changes to the existing validator proxy inventory take effect through the
same canonical restart and readiness sequence.

`LAB_ARENA_MAX_PARALLEL_RUNS` is retired. Configured and verified proxy routes
determine local capacity. Each validator owns its own proxy profile. External
validators use the same signed runner and model contract, but do not share
proxy URLs, wallet processes, or local slot state.

The slot formula is:

```text
local execution slots = min(verified Webshare proxies + 1 native slot, 20)
```

The native coordinator is slot 0. Webshare routes are slots 1 through N. All
slots use the same validator hotkey. They are sandbox execution slots, not
extra Bittensor validators or wallet workers.

Nine proxies give ten slots, so one model's twenty ICPs run as two groups of
ten. Nineteen proxies give twenty slots, so all twenty can run together. A
validator can configure more proxies, but one dataset cannot exceed the
round's frozen limit of twenty. With validators of different sizes, each one
declares its own smaller local capacity. A larger eligible validator can claim
a later group when it becomes available, but it cannot raise the frozen round
limit.

## Model web access

The sandbox still has no direct external network route. For new parallel
rounds, it receives a credential-free loopback proxy URL in `HTTP_PROXY`,
`HTTPS_PROXY`, `http_proxy`, and `https_proxy`. Webshare credentials stay in
the validator host process.

These environment variables work with libraries that implement the usual
proxy environment convention. That convention is not universal. The sandbox
also receives `LAB_ARENA_WEB_PROXY_URL` for libraries that need an explicit
proxy argument. For example, a model that uses DDGS must pass this URL through
the proxy option supported by its installed DDGS version if that version does
not use `HTTP_PROXY` and `HTTPS_PROXY` automatically.

One live attempt keeps one exclusive slot and one exit IP until it finishes.
An independently scheduled retry can receive a different slot. Proxy
geography and public-site responses can differ, so parallel and sequential
live research are not expected to return byte-identical evidence. The output
schema and scorer remain identical.

If transport cleanup cannot prove that all connections and threads stopped,
the process quarantines that exit and lowers its usable capacity. Other
attempts continue. A new pool in the same process cannot reuse that exit;
recovery requires a process restart through the normal restart procedure.

The paid Scrapingdog, Deepline, and OpenRouter paths remain on the existing
broker. More exit IPs can reduce IP-based public-site throttling. They do not
increase account quotas, provider budgets, or paid API allowances.

## Round and host safeguards

Only a round frozen with `parallel_twenty_icp_execution=true` uses the new
twenty-ICP scheduling and web bridge. Existing and frozen historical rounds
keep their prior stage behavior. A validator restart cannot rewrite that
round-level decision.

The readiness guard reserves 2 GiB for each active slot, 2 GiB for the host,
and another 128 MiB for each slot. Twenty slots therefore require 44.5 GiB;
plan for approximately 45 GiB. The readiness check stops scoring startup when
the host cannot support the derived slot count.

Deploy the gateway with the canonical `gw_restart.sh` controller and the
validator with the exact-SHA `validator_restart.sh` procedure in the
[normal validator deployment section](arena_normal_validator_weights.md#deployment).
Use the normal gateway and validator readiness checks. Do not use an ad hoc
Docker, process-kill, or service restart command for this change.

Focused local tests cover proxy inventory, N+1 leasing, shared transport,
gateway compatibility, web egress, and the Arena import boundary. Production
restart, live proxy exit verification, host-memory readiness, and complete
round evidence remain required before the feature is declared live.
