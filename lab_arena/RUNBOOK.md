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
  prefix is the only public prefix; price tables, timing, and score bundles
  stay private.
- Registry: one repository in `LAB_ARENA_REGISTRY_REPOSITORY` (for example
  `ghcr.io/leadpoet/lab-arena-models`) that every accepted miner image is
  mirrored into by digest, plus the judge image. The Arena pushes with
  `LAB_ARENA_REGISTRY_USERNAME` and `LAB_ARENA_REGISTRY_PASSWORD`; runners
  and the public pull anonymously, so choose a registry whose repositories
  can be public. Nothing is ever unpacked on the Arena host: blobs are copied
  as bytes and the manifest is written byte-identical, so the digest a miner
  named is the digest everyone pulls.

## 3. Service environment

Required: `LAB_ARENA_SUPABASE_URL`, `LAB_ARENA_SUPABASE_ANON_KEY`,
`LAB_ARENA_SERVICE_JWT`, `LAB_ARENA_SIGNING_KEY_ID`, `LAB_ARENA_BUCKET`,
`LAB_ARENA_CHAIN_ENDPOINT`, `LAB_ARENA_GENERATION_OPENROUTER_API_KEY`,
`LAB_ARENA_OPENROUTER_KMS_KEY_ID`, `LAB_ARENA_SCORING_CACHE_DIR`,
`LAB_ARENA_REGISTRY_REPOSITORY`, `LAB_ARENA_REGISTRY_USERNAME`,
`LAB_ARENA_REGISTRY_PASSWORD`, `LAB_ARENA_SCORER_IMAGE` (the judge image
reference with its digest, printed by
`scripts/build_lab_arena_judge_image.sh`; the service resolves it at startup
and pins its single-platform digest and entry command on every round), and
one scorer credential per name:
`LAB_ARENA_SCORING_OPENROUTER_API_KEY`, `LAB_ARENA_SCORING_QUALIFICATION_OPENROUTER_API_KEY`, `LAB_ARENA_SCORING_SCRAPINGDOG_API_KEY`, `LAB_ARENA_SCORING_EXA_API_KEY`.

Optional: `LAB_ARENA_NETUID` (71), `LAB_ARENA_NETWORK` (finney),
`LAB_ARENA_CHAIN_TIMEOUT_SECONDS`, `LAB_ARENA_FLOOR_RUNNER_HOTKEYS`
(comma-separated), `LAB_ARENA_OPENROUTER_ALLOWED_MODELS`,
`LAB_ARENA_MAX_IMAGE_BYTES` (2 GiB of compressed layers), `LAB_ARENA_REPOSITORY_COMMIT`,
`LAB_ARENA_SCORING_WORKERS` (4), `LAB_ARENA_BANNED_HOTKEYS_PATH` (JSON list),
`LAB_ARENA_MAX_CHALLENGERS` (256, the admitted challengers per round; lower
it only while capacity is being commissioned),
`LAB_ARENA_PUBLIC_REGISTRY_REPOSITORY` (a public repository on the same
registry host as `LAB_ARENA_REGISTRY_REPOSITORY`; at publication every
participant image is copied into it by digest and the bundle names the
public reference; unset, the bundle names the Arena reference), `AWS_REGION`.

`LAB_ARENA_MODE` selects `off` (default: nothing starts, nothing is served),
`shadow` (full rounds, publication marked shadow, no reward basis is
governing), or `live`. Rewards on chain are section 10: they need the
validator and gateway flags there, never the Arena mode alone.

Start: `python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8791`.
The driver thread advances every active round (not published or cancelled),
oldest first, once per `--tick-seconds`; `--driver-only` runs the ticks
without the API and `--no-driver` serves the API without the ticks, so the
two can run as separate processes. The API and the driver share one service
role. While a round is open, each tick also admits every uploaded submission
(resolve, check, mirror, pin) for up to five minutes before advancing, and at
the cutoff it rejects any image that still cannot be resolved. After a round
publishes, each tick releases the king model (section 7) until its receipt
exists.

A round is one stage: 30 ICPs for every participant, then one validator
scoring window, then the bundle. The statuses are `open`, `committed`,
`stage1`, `stage1_closed`, `stage1_scoring`, `stage1_judged`, `scored`,
`published`, and `cancelled`. The default schedule after the cutoff is 30
minutes for the benchmark, 300 minutes for the stage, and 90 minutes for
scoring, so a round publishes about seven hours after its cutoff.

### Daily rounds

Rounds overlap: the day's round runs its benchmark while the next round is
already open, so miners can always submit. Every signed request names its
round (`round_id` in the envelope): a submission goes to the open round it
names, and a claim or completion to the running round it names; a request
naming an unknown round is refused as `round_unknown`, and a submission to a
round past its cutoff as `submission_window_closed`. `GET /arena/v1/current`
reports `open_round` and `running_rounds` (plus `round`, the newest active
round, for older clients). Set `LAB_ARENA_DAILY_CUTOFF_UTC=<hour>` and the
service driver creates the next round itself whenever no round is open: its
cutoff is the next occurrence of that UTC hour at least six hours ahead (the
submission window), and a date whose round already exists moves to the next
day, because a round id is its cutoff date. Leave the variable unset to
create rounds by hand:

```bash
python3 scripts/lab_arena_admin.py create --cutoff 2026-09-05T00:00:00Z
```

The command refuses while a round is open for submissions and when that
date's round exists; a running round never blocks it.

## 4. Runner environment

Runners are Linux x86_64 hosts with the pinned gVisor release from
`lab_arena/runtime.lock.json` (runsc release release-20260706.0, lock hash
`sha256:f373a13e56e2c609eb239121a8f2401fdd33bdc7e0a6cc426a5b815fac8aaea9`). Verify a host before enrolling it:
`sudo python3 scripts/_lab_arena_runsc_probe_ci.py`.

Each runner needs `LAB_ARENA_API_BASE_URL`, a Bittensor wallet
(`LAB_ARENA_WALLET_NAME`, `LAB_ARENA_HOTKEY_NAME`) whose hotkey is
allowlisted for the round, optional `LAB_ARENA_MAX_PARALLEL_RUNS`,
`LAB_ARENA_RUNNER_WORK_DIR`, `LAB_ARENA_RUNSC_PATH`, and the same
`LAB_ARENA_REPOSITORY_COMMIT` the service pins. Without `--round-id` a runner
follows every running round: it asks `GET /arena/v1/current` at every idle
poll for the `running_rounds` that have work (`stage1`, `stage1_scoring`),
verifies each new round's signed release identity (worker release, runtime
lock, shim, operation table) before its first claim, keeps each round's image
sources by round id, claims the oldest round first, and rolls over to the
next daily round without a restart. `--round-id` (or `LAB_ARENA_ROUND_ID`)
pins one round instead. A runner refuses a round whose identity differs from
its own. Runners hold no provider credential, no database credential, and no
signing key.

Runners need no Docker daemon. A lease names the image by its Arena
repository reference and digest; the runner pulls the manifest and layers
anonymously from the registry, verifies every blob against the manifest and
the manifest against the pinned digest, and unpacks the layers itself with a
hardened extractor (path and hard-link escapes refused, setuid bits cleared,
device nodes skipped, whiteouts applied, a byte budget on the root
filesystem). A lease whose image lies outside the round's Arena repository,
or whose judge reference differs from the round's pinned judge, is refused.
Set `LAB_ARENA_REGISTRY_USERNAME` and `LAB_ARENA_REGISTRY_PASSWORD` on a
runner only when the Arena repository is not readable anonymously.

Start: `python3 scripts/run_lab_arena_runner.py`.

## 4.1 Miner submissions: image by digest

A miner submits one container image, named by digest, in any public
registry: `POST /arena/v1/submissions` with a signed body of
`{"image_reference": "<registry>/<repo>[:tag]@sha256:<digest>", "consent":
{"public_rerun": true, "image_publication": true}}`
(`scripts/lab_arena_miner.py submission-body`, then `sign`). The Arena
builds nothing. The driver resolves the manifest (a multi-platform index is
resolved to its `linux/amd64` child, whose digest is what gets pinned),
checks the round's public image rules (compressed size, layer count, gzip or
plain tar layers, platform, a non-empty `ENTRYPOINT` or `CMD`, no
`LAB_ARENA_*` names in `ENV`), mirrors the blobs into the Arena repository,
and records the pinned digest, reference, entry command, environment, and
working directory on the submission. Rejections carry the public
`image.*` rule ids; a second miner naming an already accepted digest is
rejected as `image.duplicate_artifact`. The status is served at
`GET /arena/v1/submissions/{id}`.

Inside the sandbox the image's own `ENTRYPOINT` plus `CMD` runs as user
65534 with a read-only root, writable `/tmp` and `/output`, its `ENV` and
`WORKDIR` honored, and the Arena's names always set: `LAB_ARENA_INPUT_PATH`
(`/input/icp.json`), `LAB_ARENA_OUTPUT_PATH` (`/output/companies.json`),
`LAB_ARENA_EVALUATION_DATE`, `LAB_ARENA_RANDOM_SEED`, and
`LAB_ARENA_WORKER_SOCKET`. There is no network. Providers are reached by
sending the provider's own HTTP request, with its real `Host` header and no
credential, over the Unix socket named by `LAB_ARENA_WORKER_SOCKET` (any
language: `curl --unix-socket`, an `httpx` UDS transport, Node's
`socketPath`). The broker matches the request to the closed operation table,
adds the miner's key, enforces the quotas, and answers with the provider's
status and body; a credential header or an unknown host is refused with a
JSON `{"error": {"code": ...}}` body. The judge image keeps the shim's frame
protocol on the same socket.

The Arena repository is private while a round runs, so a rival cannot pull
a competitor's image before the round ends; runners read it with the
registry credential. At publication every participant image is copied by
digest into `LAB_ARENA_PUBLIC_REGISTRY_REPOSITORY` (blob mounts on the same
registry host; a retried publish skips what exists), and the bundle names
each `public_image_reference` next to the Arena reference, so anyone can pull
the digest and rerun the round.

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
After the stage closes and the scoring plan is committed, the round enters
`stage1_scoring`: one scoring assignment per work item is claimable like an
ICP run by any allowlisted validator, its own executions included. The
validator runs the Arena-built judge image, whose digest, reference, and
entry command are pinned in the round configuration from
`LAB_ARENA_SCORER_IMAGE`, in the sandbox with
the trusted-scorer shim mode; the judge's Exa-compatible (routed to
Deepline), Scrapingdog, and OpenRouter calls cross the same broker on the
scored miner's keys under the per-work-item scoring quotas, and the judge models come from the signed
scorer policy. Completion carries the breakdown document under a signed
receipt. When every assignment is terminal the window closes to
`stage1_judged`; an assignment left unjudged for any reason other than the
scored miner's own key cancels the round, exactly like an execution gap.

The validators' numbers are the round's numbers. A scoring the scored
miner's own key or quota refused (`judge_key_refused`, decided by the worker
from the refusals it recorded, or a provider 401/403 on a scoring run) is
that miner's zero: the signed score bundle declares the work item under
`refused_work_items` and every submission sharing that output gets a zero row
with that cause, which the public verifier accepts only against that
declaration. Two miners with byte-identical outputs share one work item, so a
refusal on the first miner's keys also zeroes the second; identical outputs
are rare enough that V1 accepts this.

The replay is a report after publication, not a gate. Once a round publishes,
each driver tick replays `LAB_ARENA_REPLAY_ITEMS_PER_TICK` (50) accepted
scorings with `LAB_ARENA_SCORING_WORKERS` subprocesses: the same judge
entrypoint runs again on the Arena host against the responses the broker
recorded for that scoring run, no provider is called, and the score-bearing
form of the breakdowns is compared. When every scoring is replayed the Arena
signs one public report, `arena/<round>/public/replay_report.json` (also
shown on `GET /rounds/{id}` as `replay_report`), with per-validator counts of
scorings that reproduced (`match`), differed (`mismatch`), or could not be
replayed (`rejected`), and the flagged items. A validator's wrong numbers
therefore stand for that one round; the operator removes that validator from
the allowlist before the next round. This is acceptable only while Leadpoet
runs every validator. Exit criterion: before an external validator is
admitted, the replay returns as a pre-publication gate, or a stake-and-slash
design replaces it. `LAB_ARENA_REPLAY_VERIFICATION=0` disables the report
only for local rehearsal; production keeps it on.

Build and publish the judge image from this repository with
`bash scripts/build_lab_arena_judge_image.sh <registry>/<repository>`. The
recipe is `lab_arena/judge/Dockerfile`: the `python:3.11-slim` base (override
with `--build-arg BASE_IMAGE=...@sha256:...` to pin it), the pinned
dependency set from `requirements.txt`, the packages `lab_arena`,
`research_lab`, `qualification`, `gateway`, and `leadpoet_verifier` plus the
five repository packages they import (`leadpoet_canonical`,
`leadpoet_observability`, `Leadpoet`, `validator_models`, `validator_tee`)
at their repository paths under `/model`, `/model/scorer_entrypoint.py` as the
`ENTRYPOINT`, a `sitecustomize.py` that installs the shim into every Python
process, and user 65534. The script builds `linux/amd64` without provenance
or SBOM attestations so the pushed reference is a single-platform manifest,
pushes it, and prints the value for `LAB_ARENA_SCORER_IMAGE`. Runners pull it
like any miner image. Prove a candidate image before pinning it by running
`tests/lab_arena/test_lab_arena_real_judge.py` inside it against the test's
fake provider socket: the same matched calls and the same redacted breakdowns
as on the host.

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
commits a pointer to the king's pinned image (its public copy when the
round published one) to `leadpoet/leadpoet-sales-agent` on `main` as the `model/` tree
(`model/IMAGE`, `model/DIGEST`, `model/ENTRYPOINT.json`, `model/README.md`),
with the signed manifest at `arena/current.json` and one copy per round under
`arena/history/`. The model itself is the image, which anyone pulls by
digest. The commit is atomic and fast-forward only; a defended king or a
retry makes no new commit. The signed receipt is
`arena/<round>/public/model_release.json` and is shown on `GET /rounds/{id}`
as `model_release`. The release runs as its own driver step after
publication, so it never waits for a manual advance of the published round.

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
  Each participant costs about 150 sandbox-minutes (30 ICPs at the 5-minute
  cap) in the one stage. Size the floor so participants × 150 fits in 80
  percent of the 300-minute stage window: 257 participants need about 160
  slots, or 20 hosts at 8 slots. Judge executions equal participants × 30
  (7,710 at 257). They are validator work: size the scoring window from the
  judge's real duration. A judge run may take up to
  `SCORING_WALL_CLOCK_SECONDS` (15 minutes) against live providers, so the
  90-minute scoring window holds at most `slots × 90 / minutes-per-item`
  items; at 7,710 items and a five-minute average, that needs about 430
  slots, or a longer window. A window that closes with items unjudged
  cancels the round. The replay report runs after publication, one chunk of
  `LAB_ARENA_REPLAY_ITEMS_PER_TICK` scorings per driver tick with
  `LAB_ARENA_SCORING_WORKERS` subprocesses (about 4 s each on the test Mac,
  because the judge entrypoint imports the evaluator), so it never delays a
  round: at 7,710 items, 50 per tick and 16 workers finish in about eight
  hours of ticks. Scores are written in batches of 500 and scoring plans live
  in the object store, so round rows stay small at any participant count.
  Rounds overlap, so the next round's submission window costs no runner time
  while this one runs.
- Runner floor sized from the shadow timing reports before the pilot.
- Paid pilot with rewards enabled (section 10).

## 9. What a validator may assert

- A failed attempt never stands on one validator's word. Every failure but
  the miner's own quota exhaustion or key refusal gets one confirmation
  attempt (`lab_arena_complete_attempt`), claimable by a different validator
  when the round's allowlist has more than one; a second failure stands. A
  stage that closes before the confirmation ran keeps the first failure as
  the miner's zero rather than cancelling.
- A `judge_key_refused` receipt is accepted only with Arena-recorded
  evidence: a ledger refusal of a reservation or a settlement whose provider
  status the broker saw as 401 or 403 (`lab_arena.service.refusal_evidenced`).
  Nothing the runner writes counts.
- Accepted scorings stand as delivered; the post-publication replay report
  (section 6) is how a validator's wrong numbers are found, and the operator
  removes that validator. The report is signed and public.
- A model that keeps calling after its quota is refused costs the Arena at
  most `MAX_REFUSED_FRAMES` round trips per run; the worker then answers
  its frames locally.
- Run the service behind a reverse proxy that caps request bodies
  (`client_max_body_size 1m`; no route takes a large body now that images
  are named, not uploaded); the application refuses an oversized declared
  length before reading and rejects an oversized body after, but a proxy
  stops the bytes earlier.
- Anyone can rebuild a published round from the bucket's public prefix
  alone (`arena/<round>/public/`: `publication.json` with the signed
  publication, reward basis, and signing key; `bundle.json`;
  `benchmark.json`; `outputs/<hash>.json`). `--api` adds a check that the
  Arena serves the same publication:

```bash
python3 scripts/lab_arena_verify.py --round arena-2026-09-05 \
    --bucket-url https://bucket.example [--api https://arena.example]
```

## 10. Rewards on chain

The king's weight is the existing champion slot of the canonical weight
computation, filled from the Arena's signed reward basis (plan section 13).
One kernel, `leadpoet_canonical/lab_arena_rewards.py`, derives the champion
triple from a basis; the Arena imports it back, the validator proposes with
it, the gateway coordinator re-derives with it, and the canonical weight
computation refuses a snapshot whose triple differs from the basis it names.

Pieces, all in this repository:

- Migration `178` stores the Arena signing-key document with every published
  basis and exposes the signed columns through the view
  `lab_arena_reward_basis_v1`, readable with `service_role`; the Arena tables
  stay closed to it.
- The gateway serves the governing row for a weight epoch at
  `GET /fulfillment/lab-arena-reward-basis?epoch=<n>` (the newest published
  round whose effective reward epoch is at most `n`, or `null` before the
  first publication). The Arena host is never on the weight path.
- The validator (`neurons/validator.py`) reads that row when
  `LAB_ARENA_REWARDS_ENABLED=1`, verifies the signature against
  `LAB_ARENA_SIGNING_PUBLIC_KEY_HASH`, derives the triple on its metagraph,
  and carries the basis in its calculation snapshot. Any failure refuses
  weight publication: an unreachable or invalid basis is never an empty king.
- The gateway coordinator measures the same row through the query policy
  `lab_arena_reward_basis`, verifies the signature against the same pinned
  hash (set `LAB_ARENA_SIGNING_PUBLIC_KEY_HASH` in both gateway environment
  documents), re-derives the triple, and requires equality; without a basis in
  the proposal it requires an empty champion slot. A governing row newer than
  the proposal is reported as such and the next proposal picks it up.
- With the flag off every snapshot, input document, and weight bundle is
  byte-identical to before.

The pool is `LAB_ARENA_POOL_PERCENT` of total emissions (default 25, any whole
percent 0..100) in week one, then 80, 60, 40, and 20 percent of that, from
the king's start epoch; a defended king keeps its start epoch, a new king
restarts the decay. The percent is announced in every round configuration
and carried by every reward basis, so a change applies from the next round
the Arena creates and never rewrites a published basis; validators need no
change to follow it. The pinned signing-key hash is
`python3 -c 'import json,sys; print(json.load(sys.stdin)["public_key_hash"])' <<< "$(curl -s https://arena.example/arena/v1/signing-key)"`.

Order of operations: apply migration 178 (or its view part on an Arena
database that already has the tables), publish at least one round in live
mode, set `LAB_ARENA_SIGNING_PUBLIC_KEY_HASH` on the gateway and the primary
validator, restart the gateway on the attested release that carries this
code (the validator enclave image copies `leadpoet_canonical/` and
`neurons/validator.py`, so this is an attestation-changing release: rebuild
the validator image, allowlist its PCR0, and rebuild both protected-workflow
manifests with `python3 -m gateway.tee.protected_workflows --root . --write`
and `python3 -m validator_tee.host.protected_workflows_v2 --root . --write`),
then set `LAB_ARENA_REWARDS_ENABLED=1` on the primary validator. Audit
validators copy the canonical bundle unchanged.

## 11. Invariants to re-check after any change

- `python3 -m pytest tests/lab_arena -q`: the boundary tests prove no
  Arena module imports `gateway.tee` or `gateway.db`, no measured package
  imports `lab_arena`, and the enclave allowlists exclude it.
- Operation table hash `sha256:22bc6c5df8b3c950478c6923a6c3f741874673522929568fbd7bd921d112b6af` and call quota
  hash `sha256:9576d3f811ba884c2b06beed105a5471a8ba310a321257e513d45ccf98d8bac4` are pinned on every round
  configuration; changing either changes the round identity.
- No secret value may appear in any table, object, event, log, or public
  bundle; the round tests inject canary provider keys and assert absence.
