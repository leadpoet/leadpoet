# Lab Arena V1 operator runbook

This runbook covers the Arena service, floor runners, migration, and the
go-live gates from `labarena.md`. Everything here is repository-owned; the
Arena never touches enclave, validator, or weight code paths.

## 1. Database

1. Apply `scripts/178-lab-arena-v1.sql` once, with an owner connection, to
   the PostgreSQL the Arena will use. It is idempotent and creates the
   `lab_arena_owner` and `lab_arena_service` roles, six `lab_arena_*` tables,
   append-only and write-once triggers, RLS, and the SECURITY DEFINER
   service functions. Existing tables and roles are never altered.
   Give the Arena its own Supabase project (or database) for the live
   trial: nothing in `lab_arena` reads a gateway table, and a round at the
   challenger cap writes on the order of half a million ledger, event, and
   run rows inside its stage windows. If the Arena must share the gateway's
   instance, the migration bounds the service role (`statement_timeout` 30 s,
   `lock_timeout` 5 s, `idle_in_transaction_session_timeout` 60 s, applied
   by PostgREST to the impersonated role on every request), keep
   `LAB_ARENA_SCORING_WORKERS` and the runner slot ceiling at the sized
   values, and watch the gateway's database-route latency during the first
   rounds; a rise there while a no-database route stays flat is contention
   on the shared instance.
2. Mint a PostgREST JWT for role `lab_arena_service` (the role is NOLOGIN;
   the JWT carries `role=lab_arena_service`). Store it only in the Arena
   host's secret store. Runners never receive it.
3. Confirm through the API-side check that the token resolves to the
   service role: `python3 scripts/run_lab_arena_service.py --check-only`
   fails closed when `lab_arena_whoami()` returns any other role, when a
   table or service function is missing, or when the signing key or bucket
   probe fails.

## 2. AWS

- Signing key: one KMS asymmetric key, `ECC_NIST_P256`, usage `SIGN_VERIFY`.
  Its id goes in `LAB_ARENA_SIGNING_KEY_ID`. The public key is published on
  every round configuration and is what the standalone verifier trusts.
- Credential key: one KMS RSA key for decrypting miner OpenRouter runtime
  keys, id in `LAB_ARENA_OPENROUTER_KMS_KEY_ID`. Its public key is served at
  `GET /arena/v1/recipient`. Miners encrypt one key per provider
  (Scrapingdog, Deepline, OpenRouter) to it; the Arena holds no provider
  key of its own for miner runs and there is no TAO deposit.
- Object store: one bucket in `LAB_ARENA_BUCKET`. The `arena/<round>/public/`
  prefix is the only public prefix; packages, price tables, timing, and
  score bundles stay private.

## 3. Service environment

Required: `LAB_ARENA_SUPABASE_URL`, `LAB_ARENA_SUPABASE_ANON_KEY`,
`LAB_ARENA_SERVICE_JWT`, `LAB_ARENA_SIGNING_KEY_ID`, `LAB_ARENA_BUCKET`,
`LAB_ARENA_CHAIN_ENDPOINT`, `LAB_ARENA_GENERATION_OPENROUTER_API_KEY`,
`LAB_ARENA_OPENROUTER_KMS_KEY_ID`, `LAB_ARENA_SCORING_CACHE_DIR`, and one
scorer credential per name:
`LAB_ARENA_SCORING_OPENROUTER_API_KEY`, `LAB_ARENA_SCORING_QUALIFICATION_OPENROUTER_API_KEY`, `LAB_ARENA_SCORING_SCRAPINGDOG_API_KEY`, `LAB_ARENA_SCORING_EXA_API_KEY`.

Optional: `LAB_ARENA_NETUID` (71), `LAB_ARENA_NETWORK` (finney),
`LAB_ARENA_CHAIN_TIMEOUT_SECONDS`, `LAB_ARENA_FLOOR_RUNNER_HOTKEYS`
(comma-separated), `LAB_ARENA_OPENROUTER_ALLOWED_MODELS`,
`LAB_ARENA_BASE_IMAGE_DIGEST`, `LAB_ARENA_REPOSITORY_COMMIT`,
`LAB_ARENA_SCORING_WORKERS` (4), `LAB_ARENA_BANNED_HOTKEYS_PATH` (JSON list),
`LAB_ARENA_MAX_CHALLENGERS` (256, the admitted challengers per round; lower
it only while capacity is being commissioned), `AWS_REGION`. Required for validator scoring: `LAB_ARENA_SCORER_IMAGE_DIGEST`,
the pinned judge image.

`LAB_ARENA_MODE` selects `off` (default: nothing starts, nothing is served),
`shadow` (full rounds, publication marked shadow, no reward basis is
governing), or `live`. The reward release itself is a separate step (plan
section 19 step 9) and is not enabled by this service.

Start: `python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8791`.
The driver thread advances the current round once per `--tick-seconds`; the
API and the driver share one process and one service role.

### Daily rounds

V1 runs one round at a time: runner-facing handlers resolve the newest
round that is not published or cancelled, so the next round is created only
after the previous one ends. Set `LAB_ARENA_DAILY_CUTOFF_UTC=<hour>` and the
service driver creates the next round itself whenever no round is open or
running: its cutoff is the next occurrence of that UTC hour at least six
hours ahead (the submission window), and a date whose round already exists
moves to the next day, because a round id is its cutoff date. Leave the
variable unset to create rounds by hand:

```bash
python3 scripts/lab_arena_admin.py create --cutoff 2026-09-05T00:00:00Z
```

The command refuses while a round is open or running and when that date's
round exists. A day's cycle is the submission window plus the stage minutes
in the round configuration, so with a six-hour window and the default stage
minutes a round publishes well inside the day it is named for.

## 4. Runner environment

Runners are Linux x86_64 hosts with the pinned gVisor release from
`lab_arena/runtime.lock.json` (runsc release release-20260706.0, lock hash
`sha256:f373a13e56e2c609eb239121a8f2401fdd33bdc7e0a6cc426a5b815fac8aaea9`). Verify a host before enrolling it:
`sudo python3 scripts/_lab_arena_runsc_probe_ci.py`.

Each runner needs `LAB_ARENA_API_BASE_URL`, `LAB_ARENA_ROUND_ID`, a Bittensor
wallet (`LAB_ARENA_WALLET_NAME`, `LAB_ARENA_HOTKEY_NAME`) whose hotkey is
allowlisted for the round, optional `LAB_ARENA_MAX_PARALLEL_RUNS`,
`LAB_ARENA_RUNNER_WORK_DIR`, `LAB_ARENA_RUNSC_PATH`, and the same
`LAB_ARENA_REPOSITORY_COMMIT` the service pins. A runner refuses to start
when its worker release, runtime lock, shim, or operation table differs
from the signed round configuration. Runners hold no provider credential,
no database credential, and no signing key.

Start: `python3 scripts/run_lab_arena_runner.py --round-id <round>`.

## 5. Miner keys and call quotas

Miners bring their own provider keys and there is no TAO deposit. Each miner
registers one encrypted key per provider through
`POST /arena/v1/credentials/{provider}` for `scrapingdog`, `deepline`, and
`openrouter`; the miner CLI encrypts each key from
`LAB_ARENA_<PROVIDER>_RUNTIME_KEY`. The service decrypts once in memory,
runs a read-only probe (OpenRouter key info, the Scrapingdog account
endpoint, one Deepline tool schema), and stores only the ciphertext envelope
and a non-secret record. A miner is eligible only when all three keys have
passed; a miner with a failed or missing key stays in the round with zero
records, like a king that fails preflight.

Advise miners to register keys with spend limits where the provider offers
them. The Arena host would expose stored keys in a breach, and limited keys
bound that damage.

Fairness is a fixed call quota per provider per ICP attempt, pinned on every
round configuration: scrapingdog 30, deepline 30, openrouter 60. The stage quota is
the per-ICP quota times the stage's ICP count times the attempt limit.
Deepline calls are limited to the tools exa_answer, exa_company_search, exa_contents, exa_people_search, exa_search, free_simple_company_search;
their payloads are Deepline's own schemas and pass through unchanged apart
from size bounds and a credential-name scan.

Deepline contract, pinned from live calls on 2026-09-02: the broker sends
`POST https://code.deepline.com/api/v2/integrations/<tool>/execute` with the
miner's bearer key, the header `x-deepline-execute-response-intent: raw`, and
the body `{"provider": "exa", "operation": "<tool>", "payload": {...}}`, exactly
as Deepline's own client does. The reply is
`{"job_id", "status", "result": {"data": <raw provider response>}, "billing": {"credits_charged", "cost_usd"}}`
and the broker records `cost_usd` as the settled amount. `text.maxCharacters`
is honored and `ids` aliases `urls`. Exa attaches `entities` (people with
employers and education) to search and contents results; the broker drops
person entities before a model or the ledger sees them, except for
`exa_people_search`, whose purpose is people. Fixtures from the live calls
live in `tests/lab_arena/fixtures/deepline/`.

## 6. Validator scoring

Validators run the judge as well as the models, and one validator is enough.
After a stage closes and the scoring plan is committed, the round enters
`stageN_scoring`: one scoring assignment per work item is claimable like an
ICP run by any allowlisted validator, its own executions included. The
validator runs the Arena-built judge image, whose digest is pinned in the
round configuration from `LAB_ARENA_SCORER_IMAGE_DIGEST`, in the sandbox with
the trusted-scorer shim mode; the judge's Exa-compatible (routed to
Deepline), Scrapingdog, and OpenRouter calls cross the same broker on the
scored miner's keys under the per-work-item scoring quotas, and the judge models come from the signed
scorer policy. Completion carries the breakdown document under a signed
receipt. When every assignment is terminal the window closes to
`stageN_judged`; an assignment left unjudged for any reason other than the
scored miner's own key cancels the round, exactly like an execution gap.

The replay is the single integrity check. In the judged state the Arena runs
the same judge entrypoint again, in a subprocess on the Arena host, against
the responses the broker recorded for each scoring run; no provider is
called. Reproduced with the same numbers: accepted. Reproduced with different
numbers: the replayed numbers are scored and the validator is listed as a
mismatch in the stage timing report. Not reproducible at all (judge requests
with no recorded answer, a failed replay, an invalid output): the scoring is
rejected as `replay_rejected`, the round returns to `stageN_scoring` for a
second attempt of that item, and an item that fails twice cancels the round
rather than becoming a miner's zero. A scoring the scored miner's own key or
quota refused (`judge_key_refused`, decided by the worker from the refusals
it recorded, or a provider 401/403 on a scoring run) is that miner's zero:
the signed score bundle declares the work item under `refused_work_items`
and every submission sharing that output gets a zero row with that cause,
which the public verifier accepts only against that declaration. Two miners
with byte-identical outputs share one work item, so a refusal on the first
miner's keys also zeroes the second; identical outputs are rare enough that
V1 accepts this. The rejected attempt keeps its receipt,
output, and event chain unchanged in the ledger, so anyone can rerun the
replay from the ledger and check the rejection; this is the one exception to
terminal-attempt immutability, and it applies only to accepted score runs. `LAB_ARENA_REPLAY_VERIFICATION=0`
disables the replay only for local rehearsal; production keeps it on.

Build the judge image from this repository with the scorer entrypoint
(`lab_arena/scorer_entrypoint.py`) and the evaluator's dependencies, pin its
digest, and give every validator the same image through the registry the
runner already pulls miner images from. There is no committed build recipe
for the judge image yet; until one lands, the image is an operator artifact
and its digest must be recorded with the round configuration that pins it.
The image must hold, on the same pinned Python base as miner images:
`/model/scorer_entrypoint.py` (this repository's `lab_arena/scorer_entrypoint.py`;
the runner starts `python3 /model/scorer_entrypoint.py` with the input and
output directories and the worker socket mounted exactly as for a model);
the packages `lab_arena`, `research_lab`, `qualification`, `gateway`, and
`leadpoet_verifier` importable at their repository paths, with their
dependencies from `requirements.txt`; a `sitecustomize.py` on the import
path that runs `from lab_arena import shim; shim.install()`; user 65534; no
network at build or run time beyond the wheel install. Prove a candidate
image before pinning it by running `tests/lab_arena/test_lab_arena_real_judge.py`
inside it against the test's fake provider socket: the same 30 matched calls
and the same redacted breakdowns as on the host.

### How the judge reaches the web

The Research Lab judge (`qualification.scoring.lead_scorer` behind
`research_lab.eval.evaluator`) reads company homepages, evidence pages, and
the Wayback Machine with plain GETs, calls Exa contents, scrapes with
Scrapingdog, and calls OpenRouter with six models. Inside the judge sandbox
every one of those crosses the shim in trusted-scorer mode:

- A plain HTTPS GET to a host that is not a provider becomes the closed
  `scrapingdog.scrape` operation on that URL (`lab_arena.shim`), accounted
  against the scored miner's Scrapingdog quota. The judge sees the web only
  through Scrapingdog; egress stays closed.
- The judge's OpenRouter privacy request (`provider: data_collection deny,
  zdr`) is accepted only as a subset of the table's pinned policy and dropped;
  the table's policy (deny, no fallbacks, ZDR) is what goes out. JSON reply
  formats, its JSON schema, and `include_reasoning` are accepted fields.
- The signed scorer policy pins exactly the models the judge's source calls
  (`lab_arena.scoring.DEFAULT_JUDGE_MODELS`); the broker refuses any other
  model on a scoring run, and `tests/lab_arena/test_lab_arena_judge_routes.py`
  scans the judge's source so a model change fails the test before a round.
  The round prices the judge models next to the miners' allowed models.
- `tests/lab_arena/test_lab_arena_real_judge.py` runs the real evaluator
  through the entrypoint and the shim against a shape-valid fake provider:
  every request must match one operation, use only pinned models, and
  reproduce identical requests and redacted breakdowns on a second run.
- Scoring quotas per work item are sized from that run (about 6 Scrapingdog,
  3 OpenRouter, and 1 Deepline call per company) with retry headroom.
- `LAB_ARENA_SHIM_TRACE_PATH=<file>` makes the shim append one JSON line per
  intercepted request (method, host, path, matched operation or refusal
  code; never bodies, headers, or queries). Off by default; use it in a
  rehearsal when a judge or a model cannot reach a provider.

## 7. Model release

After each published round with a crowned or defended king, the driver
commits the king's frozen source to `leadpoet/leadpoet-sales-agent` on
`main` as the `model/` tree, with the signed manifest at
`arena/current.json` and one copy per round under `arena/history/`. The
commit is atomic and fast-forward only; a defended king or a retry makes no
new commit. The signed receipt is `arena/<round>/public/model_release.json`
and is shown on `GET /rounds/{id}` as `model_release`.

Live mode requires `LAB_ARENA_GITHUB_TOKEN` (a fine-grained token with
contents write on that one repository) and accepts
`LAB_ARENA_MODEL_REPOSITORY` and `LAB_ARENA_MODEL_BRANCH` overrides. Shadow
mode never releases. An empty repository is bootstrapped with one README
commit on the first release.

## 8. Go-live gates

- Seven consecutive shadow rounds published with the throughput gate
  (`ArenaService.shadow_report`) satisfied, including at least one round at
  the challenger count you intend to admit at launch.
- Capacity sizing: every miner may enter one agent per round, up to 256.
  Each admitted challenger costs about 100 sandbox-minutes in Stage 1 (20
  ICPs at the 5-minute cap) and each finalist 150 in Stage 2. Size the floor
  so admitted challengers × 100 fits in 80 percent of the 210-minute Stage 1
  window: 257 participants need about 160 slots, or 20 hosts at 8 slots.
  Stage 1 judge executions equal participants × 20 (5,140 at 257), so raise
  `LAB_ARENA_SCORING_WORKERS` with the participant count. Scores are written
  in batches of 500 and scoring plans live in the object store, so round rows
  stay small at any participant count.
- Scorer egress enforced with the nftables ruleset from
  `lab_arena.egress.scorer_nftables_ruleset` on the scoring host.
- Runner floor sized from the shadow timing reports before the pilot.
- Paid pilot, then the reward release (plan section 19 step 9), which is the
  first change to files outside `lab_arena/`, `scripts/`, and `tests/`.

## 9. Invariants to re-check after any change

- `python3 -m pytest tests/lab_arena -q`: the boundary tests prove no
  Arena module imports `gateway.tee` or `gateway.db`, no measured package
  imports `lab_arena`, and the enclave allowlists exclude it.
- Operation table hash `sha256:22bc6c5df8b3c950478c6923a6c3f741874673522929568fbd7bd921d112b6af` and call quota
  hash `sha256:9576d3f811ba884c2b06beed105a5471a8ba310a321257e513d45ccf98d8bac4` are pinned on every round
  configuration; changing either changes the round identity.
- No secret value may appear in any table, object, event, log, or public
  bundle; the round tests inject canary provider keys and assert absence.
