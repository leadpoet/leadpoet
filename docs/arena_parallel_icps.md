# Arena parallel ICP execution

Arena can run the configured ICPs for one model in parallel. The validator uses
one native host route and one route for each verified Webshare proxy. The
change affects execution speed and public web egress. It does not change the
model output contract, scoring, paid provider broker, provider budgets,
validator identity, or wallet use.

## Daily benchmark settings

New rounds default to 10 ICPs and a champion promotion margin of 0.5 points.
The gateway accepts `LAB_ARENA_BENCHMARK_ICP_COUNT` (2 to 100) and
`LAB_ARENA_PROMOTION_MARGIN` (0 to 100). Restart through the normal deployment
workflow after changing these settings. They apply to newly created rounds.
Each round stores its own count and margin; existing rounds keep their values.
The generator also reads an open round's frozen count when its bank is missing.

Baseline and miners use the same committed bank. The final score remains the
mean of the configured ICP scores, with the existing per-ICP qualification and
cost rules. Promotion requires a miner score at least baseline plus the stored
margin. Submission cutoff, ICP disclosure, promotion, and reward timing stay
unchanged. Historical rounds without a stored margin retain the 1.0 threshold.

Generation allocates the bank across the existing 20-industry catalog. A
10-ICP bank uses ten different industries; a larger bank covers all industries
before repeating them. Exact count, industry allocation, and duplicate checks
run before storage. Stage, geography, and intent diversity also remain in the
generation prompt; the international-share check reports drift as a warning.

## Validator configuration

Use indexed settings in the validator environment:

```text
LAB_ARENA_WEBSHARE_PROXY_1=http://USER:PASSWORD@PROXY_IP_1:PORT_1
LAB_ARENA_WEBSHARE_PROXY_2=http://USER:PASSWORD@PROXY_IP_2:PORT_2
```

### Webshare setup for external validators

Use your own [Webshare Proxy Server](https://www.webshare.io/proxy-server)
datacenter proxies. Select **Username/Password** authentication and **Direct
Connection** on the Proxy List page. Use the address, port, username and
password shown on each row. Start with ten different US proxies; rotating
residential proxies are not needed for this setup. Plan prices and bandwidth
allowances can change, so check them before purchase.

Use `http://` for Webshare's standard direct endpoints, including when the
destination is HTTPS. The host opens a CONNECT tunnel and verifies TLS to the
destination. `https://` instead requires TLS on the proxy endpoint itself;
do not change the scheme merely because the destination uses HTTPS. This
matches [Webshare's direct connection example](https://apidocs.webshare.io/proxy-connection#direct-connection).

Create a private file outside the repository, for example
`$HOME/.config/leadpoet/validator-proxies.env`, containing the indexed settings
above, with your own values. Continue through `LAB_ARENA_WEBSHARE_PROXY_10` for
ten proxies. Quote each URL with single quotes. URL-encode special characters
in the username or password, such as `@` as `%40`. Do not paste credentials
into Discord, Git, shell command arguments, or screenshots.

```bash
chmod 600 "$HOME/.config/leadpoet/validator-proxies.env"
export LAB_ARENA_PROXY_ENV_FILE="$HOME/.config/leadpoet/validator-proxies.env"
```

Set that export in the environment used to start your validator. The normal
command does not source an arbitrary `.env` file. The private proxy file is
parsed as data; only the indexed proxy settings are imported. Existing direct
environment values must not conflict with values in the file.

After the one-time [host setup](arena_normal_validator_weights.md#run-a-validator),
check the exact Python environment and proxy settings before starting:

```bash
python neurons/validator.py --check-scoring-only
python neurons/validator.py \
  --netuid 71 --subtensor.network finney \
  --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY \
  --wallet.path /absolute/path/to/YOUR_WALLETS_DIRECTORY
```

The check uses the same existing sudo permission and scoring setup as normal
startup. It verifies the host, proxy connections, distinct exit IPs, and memory
capacity. It does not load a wallet, claim a job, or submit weights. Host rights
and gVisor still need to be installed by the operator once. Keep existing
wallet, journal and runner paths, and run only one process for each hotkey.

### Configuration rules

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
local execution slots = min(verified Webshare proxies + 1 native slot,
                            memory-supported slots, 20)
```

The native coordinator is slot 0. Webshare routes are slots 1 through N. All
slots use the same validator hotkey. They are sandbox execution slots, not
extra Bittensor validators or wallet workers.

With sufficient available memory, nine proxies give ten slots, so a ten-ICP
benchmark fits in one group. A twenty-ICP benchmark needs two groups with
those same slots. Nineteen proxies give twenty slots. A larger bank uses
additional groups under the same slot limit. The round freezes its benchmark
count separately from its concurrency limit. With validators of different sizes, each one
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

Paid calls also retain the broker's existing cost-admission rules. A Deepline
call without an authoritative maximum price reserves the submission's remaining
execution budget until its charge settles. Such calls serialize across that
submission's ICPs and can cause other calls to return `budget_busy`. Proxy IPs
cannot remove this limit. Observed charges and provider price estimates are
not safe substitutes for an enforced maximum price.

Two groups of ten reduce sandbox execution time when model work can proceed
independently. They do not guarantee a complete benchmark in two single-ICP
runtimes. Shared paid-provider admission, account quotas, model deadlines,
scoring, settlement, and retries can add time. Live output quality must be
checked as well as completed attempt counts.

## Round and host safeguards

Only a round frozen with `parallel_twenty_icp_execution=true` uses the new
twenty-ICP scheduling and web bridge. Existing and frozen historical rounds
keep their prior stage behavior. A validator restart cannot rewrite that
round-level decision.

The readiness guard reserves 2 GiB for each active slot, 2 GiB for the host,
and another 128 MiB for each slot. Twenty slots therefore require 44.5 GiB;
plan for approximately 45 GiB of available memory. Readiness derives the safe
slot count from available host and cgroup memory. It reduces concurrent slots
when memory is limited, retains all verified proxy profiles for slot rotation,
and stops scoring startup if even one slot cannot fit. The weight loop remains
independent.

Deploy the gateway with the canonical `gw_restart.sh` controller and the
validator with the exact-SHA `validator_restart.sh` procedure in the
[normal validator deployment section](arena_normal_validator_weights.md#deployment).
Use the normal gateway and validator readiness checks. Do not use an ad hoc
Docker, process-kill, or service restart command for this change.

Focused local tests cover proxy inventory, N+1 leasing, shared transport,
gateway compatibility, web egress, and the Arena import boundary. Production
restart, live proxy exit verification, host-memory readiness, and complete
round evidence remain required before the feature is declared live.
