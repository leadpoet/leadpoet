# Codex Runtime in Arena

## Scope

Run Codex-based harnesses through Arena's existing worker and provider broker.
Arena supplies the executable and a local Responses bridge; each harness
supplies `harness.run_icp(icp)`, local tools, instructions and company output.
There is no new service, database table, scheduler or agent framework.

Acceptance: the packaged Codex CLI must call a local tool, continue with the
tool result, and complete through the real worker/broker path with settled
costs. The Linux probe must additionally exercise the actual gVisor sandbox.
Scoring, output validation, deadlines, provider budgets, credentials and
accepted results retain their existing contracts.

## Harness Interface

New parallel-execution sandboxes mount `lab_arena_codex` beside
`lab_arena_checkpoint`. The runtime uses their existing private loopback
permission; gVisor still runs with `--network=none`. Historical executions and
scorers do not receive this helper.

```python
import lab_arena_codex

def run_icp(icp):
    # Prepare the harness's writable workspace under /tmp or /output.
    # Supply an OpenRouter model included in this round's price table.
    final_message = lab_arena_codex.run(
        "Run the research task using the workspace's tools and instructions.",
        model="openai/gpt-4o-mini",
        cwd="/tmp/research-workspace",
        reasoning_effort="medium",
        max_output_tokens=16384,
        timeout_seconds=2700,
    )
    # The harness validates and loads its saved company records, then returns
    # the list through the existing Arena output contract.
```

`run` returns Codex's final message; that message is not accepted as company
evidence. The harness owns output and checkpoints. The outer Arena deadline
remains authoritative, including when a harness asks for a longer timeout.
The model above illustrates the configuration shape; select and test the
actual model/effort separately. There is no TYCHE-specific model selection,
model catalog override or replacement agent framework in this runtime.

For an existing CLI or SDK launcher,
`session(model=..., reasoning_effort=..., max_output_tokens=...)`
is a context manager yielding a dict-compatible child environment. Use it only
while the context is open. It supplies an isolated `CODEX_HOME`, local provider
config and an attempt-local bridge token. It does not inherit a personal Codex
login or provider key. Use `/usr/local/bin/codex`; the pinned native package is
`0.154.0`. The CLI path avoids an additional Node/SDK dependency.

The yielded environment also provides `wait_idle(timeout_seconds)`. A model
supervisor can call it before starting another Codex process when an interrupted
process may have left its last Responses request in progress. The call only
waits on the attempt-local bridge semaphore. It returns `True` after the prior
request settles and `False` at the timeout; it does not send, queue, replay or
cancel a provider request. The finite timeout must be from 0 through 2,700
seconds. The model still decides when to start finalization or write a
checkpoint, and the capability becomes invalid when the session closes.

`session` also accepts an optional zero-argument `request_guard`. The bridge
calls it after validating an otherwise admissible Responses request and while
holding the attempt-local request gate, immediately before worker dispatch.
Only the exact value `True` permits dispatch. `False`, another return type, or
an exception produces the same generic local HTTP 429 response without a
worker frame, provider call, ledger entry or queue. With the default `None`,
the bridge does not call a guard and retains its existing behavior.

The guard is one local admission check per bridge HTTP request. It does not
reserve quota: existing Codex, worker and credential retry paths can create
more than one ledger identity around a model turn. A caller must obtain any
authoritative state through a separate passive interface and own its phase and
fallback decisions. The bridge does not infer quotas, deadlines or model
finalization, and it does not change the request body.

## Request Path

Codex posts to a private `127.0.0.1` Responses listener. The helper sends an
`openrouter.responses` frame to the existing worker Unix socket. The gateway
checks the closed request schema, reserves the existing budget, injects the
submission's credential and posts to OpenRouter `/api/v1/responses`.

The upstream request is non-streaming and stateless (`stream=false`,
`store=false`). Once billing is settled, the helper replays the native output
items as SSE events for Codex. This deliberately buffers each model response;
there is no token-by-token delivery from the provider. Tool call identities,
arguments and opaque reasoning history are preserved, not translated to Chat
Completions. Model, input, output and response-size limits remain bounded.

The same path accepts standard Responses and Codex Responses Lite: local
function/custom tools, namespaces, `additional_tools`, reasoning context and
string or typed text tool results. Native model metadata decides which wire
shape Codex uses. Arena preserves the shape through compaction and subsequent
tool calls. Tools inside namespaces are subject to the same closed allowlist.
The Responses parameter depth allowance is 24; other operations retain their
existing limits. Request bytes, text lengths and tool counts remain bounded.

`run` and `session` request 16,384 output tokens per model call by default,
including reasoning. Callers can choose 1–32,768; the bridge refuses requests
above the session allowance. With migration 321, sourcing admission counts only
confirmed settlement costs; output estimates and pending bills do not reserve
money or block another request. Current-policy judge admission also uses confirmed
charges, under its separate existing cap. Direct Responses
calls still default to 4,096; Chat and judge caps remain 4,096. Raising the
per-call allowance does not raise the round's budget or guarantee that a model
can finish a turn within it.

Supported reasoning effort settings are `none`, `minimal`, `low`, `medium`,
`high`, `xhigh` and `max`, subject to the selected model's support. The helper
starts one Codex agent and disables hosted search, image generation and
automatic subagents. Any model must be available through OpenRouter's
Responses endpoint and admitted by the round's price table. This is a shared
text/local-tool runtime, not a claim that every provider model or hosted tool
is available.

Only text, local function/custom tools and their returned history are accepted.
Hosted tools, remote images/files, server-side conversation IDs, background
work and caller-selected routing are rejected. Research-provider calls still
use the harness's Arena broker adapter. Codex's built-in hosted web search is
disabled.
Codex HTTP and streaming retries are disabled. A transport failure or HTTP 5xx
can have an unknown bill, so it must not create an automatic duplicate paid POST.
The original call remains tracked for exact billing recovery. The runner may retry a Responses request at most twice only after
the broker proves the preceding attempt settled at zero cost with provider
status 429. Each runner retry also uses a new normal action sequence, consumes
the existing call quota, honors a valid bounded `Retry-After`, and stays inside
the worker request deadline. Stopping the worker cancels a pending backoff.
Incomplete model responses retain actual billing and do not become successful
sourcing calls.

A harness can inspect its current per-run call counters through the already
mounted checkpoint helper:

```python
from lab_arena_checkpoint import quota_usage

snapshot = quota_usage()
openrouter_remaining = snapshot["providers"]["openrouter"]["remaining"]
```

Each fixed provider reports `limit`, `used`, `remaining`, and `inflight`.
Reservations and dispatches are used and in flight; settlements and uncertain
outcomes are used; refusals and recovered reservations are not used. Retries
with separate call identities count separately. The read uses the active lease,
changes no provider, run, ledger, lease, action-sequence, or refusal state, and
is cached by the worker for one second. At most 256 reads are accepted during a
run. Stale leases, malformed responses, transport failures, and the read cap
raise the same generic `QuotaUnavailable("quota unavailable")` error.

The result is a point-in-time observation. It does not reserve a call or money,
renew the lease, change admission, or promise that a later request will succeed.

On a host with migration 319, a model can opt into version 2 with
`quota_usage(include_sourcing_cost=True)`. The same lease-bound endpoint and
worker socket return the existing counters plus `sourcing_cost`. It includes
all execute attempts for this ICP, including Codex/OpenRouter usage, and
excludes judge calls. `successful_microusd` is settled successful sourcing;
`success_unresolved_microusd` is its unresolved exposure. The separate
`settled_microusd` includes billed failures. `reserved_or_uncertain_microusd`
retains historical holds for audit; it is not spend and migration 321 excludes it
from sourcing admission. New execute lifecycle reservations carry amount zero.
The response includes in-flight/unresolved call counts, the admission cap,
and the allowance per qualified company. Qualification is known
only after judging. Models own their stopping decisions; this read changes
neither admission nor scoring. Default version 1 responses remain unchanged.

The gateway shares a conservative Responses reliability gate across rounds by
OpenRouter credential. It admits two concurrent requests per credential by
default; this is a local protection, not a claimed upstream account limit. A
gateway service can set `LAB_ARENA_OPENROUTER_MAX_CONCURRENCY` to an integer
from 1 through 10. The default needs no secret or environment update. Other
provider operations and different OpenRouter credentials remain independent.
For a 120-second provider operation, queueing can use about 90 seconds while
retaining 20 seconds for database admission, at least 30 seconds for provider
work, 30 seconds for billing, and 15 seconds of API grace. A disconnect cancels
the queue wait and is checked once more before reservation. After reservation
commits, the existing dispatch, transport, and settlement sequence completes so
the ledger cannot retain an abandoned reservation. The setting does not change
model, round, quota, cost, or schema policy.

Codex disables its inner OS sandbox because the process is already inside
gVisor, with a read-only image/source, an unprivileged UID, bounded writable
directories and no external network interface. This helper is not a general
host-side launcher and must not be used to run untrusted code outside Arena.

## Build and Verify

The existing judge image also supplies the execute rootfs. Its Dockerfile
installs the native Codex package through `scripts/install_arena_codex.py`,
checking a fixed SHA-512 before extracting regular files. Publish and select
the image through the existing release process; a source checkout alone does
not install the runtime in production. Apply
`264-lab-arena-codex-cost-reconciliation.sql` before enabling the new runtime:
it extends the two existing exact-generation recovery functions to Responses,
preserving operation identity, permissions and ledger history. No round rerun
is needed.

```sh
pytest -q tests/lab_arena/codex_runtime_test.py \
  tests/lab_arena/codex_protocol_validation_test.py \
  tests/lab_arena/codex_runtime_limits_test.py
ARENA_TEST_CODEX_BINARY=/path/to/codex pytest -q \
  tests/lab_arena/codex_runtime_test.py -k "real_codex or pinned"
LAB_ARENA_PG_LOCAL=1 pytest -q tests/lab_arena/codex_cost_reconciliation_postgres_test.py
# On a disposable Linux x86_64 root host with the built image exported:
ARENA_TEST_CODEX_ROOTFS=/path/to/rootfs ARENA_TEST_RUNSC=/usr/local/bin/runsc \
  pytest -q tests/lab_arena/codex_runtime_test.py -k real_gvisor
```

The native CLI test uses scripted provider responses through the actual worker
and broker. It proves wire compatibility, a real shell command, continuation
and exact ledger settlement without paid calls. It does not prove live model
quality, OpenRouter model availability or gVisor execution on macOS.

## Verification status

The PR's original flat-tool smoke test passed on Codex 0.154.0. A subsequent
source and native-wire audit identified rejected newer-model fields, depth
limits and a trusted-helper staging omission; the updated tests cover those
regressions. The Deploy Checks workflow now installs the same checksum-pinned
Linux package and runs native protocol checks before the full suite. Scripted
responses exercise the worker, broker, cost ledger and local tool execution;
they do not make paid model or research-provider calls. The Docker smoke job
also builds the Arena judge/execute image, including its pinned Codex installer
and build-time version/import checks, without publishing it.

The integrated runtime has local protocol, budget, broker, sandbox and
PostgreSQL recovery coverage. The new database tests check both Chat and
Responses settlement, credential ownership, idempotency and service-only
permissions. Deployment still requires the built Linux image's gVisor probe,
an authorized live call with the chosen admitted model, and a harness run whose
saved company output is accepted by Arena. Scripted protocol tests do not prove
live provider availability or lead quality.

References: [Codex provider configuration](https://learn.chatgpt.com/docs/config-file/config-reference),
[OpenRouter Responses schema](https://openrouter.ai/openapi.json),
[Codex 0.154.0 source](https://github.com/openai/codex/tree/rust-v0.154.0).
