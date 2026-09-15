# Codex Runtime in Arena

## Scope

Keep Tyche's Codex research loop. Arena supplies the executable and a local
Responses bridge; Tyche supplies `harness.run_icp(icp)`, research tools and
company output. There is no new service, database table, scheduler or agent
framework.

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
    # Prepare Tyche's writable research workspace under /tmp or /output.
    # Supply an OpenRouter model included in this round's price table.
    final_message = lab_arena_codex.run(
        "Run the research task using the workspace's tools and instructions.",
        model="openai/gpt-4o-mini",
        cwd="/tmp/tyche-workspace",
        reasoning_effort="medium",
        timeout_seconds=2700,
    )
    # Tyche validates and loads its saved company records here, then returns
    # the list through the existing Arena output contract.
```

`run` returns Codex's final message; that message is not accepted as company
evidence. The harness owns output and checkpoints. The outer Arena deadline
remains authoritative, including when a harness asks for a longer timeout.
The model above illustrates the configuration shape; select and test the
actual Tyche model/effort separately.

For an existing CLI or SDK launcher, `session(model=..., reasoning_effort=...)`
is a context manager yielding the child environment. Use it only while the
context is open. It supplies an isolated `CODEX_HOME`, local provider config
and an attempt-local bridge token. It does not inherit a personal Codex login
or provider key. Use `/usr/local/bin/codex`; the pinned native package is
`0.154.0`. The CLI path avoids an additional Node/SDK dependency.

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

Only text, local function/custom tools and their returned history are accepted.
Hosted tools, remote images/files, server-side conversation IDs, background
work and caller-selected routing are rejected. Research-provider calls still
use Tyche's Arena broker adapter. Codex's built-in hosted web search is disabled.
Codex transport retries are disabled; the existing broker owns retry and cost
recovery. Incomplete model responses retain actual billing and do not become
successful sourcing calls.

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
`258-lab-arena-codex-cost-reconciliation.sql` before enabling the new runtime:
it extends the two existing exact-generation recovery functions to Responses,
preserving operation identity, permissions and ledger history. No round rerun
is needed.

```sh
pytest -q tests/lab_arena/codex_runtime_test.py
ARENA_TEST_CODEX_BINARY=/path/to/codex pytest -q \
  tests/lab_arena/codex_runtime_test.py -k real_codex
LAB_ARENA_PG_LOCAL=1 pytest -q tests/lab_arena/codex_cost_reconciliation_postgres_test.py
# On a disposable Linux x86_64 root host with the built image exported:
ARENA_TEST_CODEX_ROOTFS=/path/to/rootfs ARENA_TEST_RUNSC=/usr/local/bin/runsc \
  pytest -q tests/lab_arena/codex_runtime_test.py -k real_gvisor
```

The native CLI test uses scripted provider responses through the actual worker
and broker. It proves wire compatibility, a real shell command, continuation
and exact ledger settlement without paid calls. It does not prove live model
quality, OpenRouter model availability or gVisor execution on macOS.

Verified locally on 2026-09-15: Codex 0.154.0 executed a Python tool from a
writable workspace, imported the mounted runtime helper, and continued with
the tool result through the actual HTTP listener, worker socket and broker.
Both calls settled their scripted costs. The test also sets web proxy
variables to confirm that the private Responses listener is reached directly.
The focused operation/broker/runtime/runner/shim regression suite passed
1,077 tests. Disposable Postgres tests verified both operation types, exact
generation and credential matching, one settlement on replay, migration
reapplication and unchanged service-only permissions.

Not verified here: execution of the built Linux image inside gVisor, paid
OpenRouter calls with Tyche's selected model/effort, and the separate Tyche
harness/output adapter. This macOS host has no running Docker daemon. The
Linux test above is the required next runtime check before deployment.

References: [Codex provider configuration](https://learn.chatgpt.com/docs/config-file/config-reference),
[OpenRouter Responses API](https://openrouter.ai/docs/api_reference/responses/overview).
