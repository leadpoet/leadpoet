# Lab Arena V1 operator runbook

This runbook covers the Arena service, floor runners, migration, and the
go-live gates from `labarena.md`. Everything here is repository-owned; the
Arena never touches enclave, validator, or weight code paths.

## 1. Database

1. Apply `scripts/174-lab-arena-v1.sql` once to the hosted Supabase project
   with an owner connection. It is idempotent and creates the
   `lab_arena_owner` and `lab_arena_service` roles, six `lab_arena_*` tables,
   append-only and write-once triggers, RLS, and the SECURITY DEFINER
   service functions. Existing tables and roles are never altered.
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
  `GET /arena/v1/recipient`.
- Object store: one bucket in `LAB_ARENA_BUCKET`. The `arena/<round>/public/`
  prefix is the only public prefix; packages, price tables, timing, and
  score bundles stay private.

## 3. Service environment

Required: `LAB_ARENA_SUPABASE_URL`, `LAB_ARENA_SUPABASE_ANON_KEY`,
`LAB_ARENA_SERVICE_JWT`, `LAB_ARENA_SIGNING_KEY_ID`, `LAB_ARENA_BUCKET`,
`LAB_ARENA_CHAIN_ENDPOINT`, `LAB_ARENA_GENERATION_OPENROUTER_API_KEY`,
`LAB_ARENA_EXA_API_KEY`, `LAB_ARENA_SCRAPINGDOG_API_KEY`,
`LAB_ARENA_OPENROUTER_KMS_KEY_ID`, `LAB_ARENA_SCORING_CACHE_DIR`,
`LAB_ARENA_TAO_RECIPIENT_WALLET`, and one scorer credential per name:
`LAB_ARENA_SCORING_OPENROUTER_API_KEY`, `LAB_ARENA_SCORING_QUALIFICATION_OPENROUTER_API_KEY`, `LAB_ARENA_SCORING_SCRAPINGDOG_API_KEY`, `LAB_ARENA_SCORING_EXA_API_KEY`.

Optional: `LAB_ARENA_NETUID` (71), `LAB_ARENA_NETWORK` (finney),
`LAB_ARENA_CHAIN_TIMEOUT_SECONDS`, `LAB_ARENA_FLOOR_RUNNER_HOTKEYS`
(comma-separated), `LAB_ARENA_OPENROUTER_ALLOWED_MODELS`,
`LAB_ARENA_BASE_IMAGE_DIGEST`, `LAB_ARENA_REPOSITORY_COMMIT`,
`LAB_ARENA_SCORING_WORKERS` (4), `LAB_ARENA_BANNED_HOTKEYS_PATH` (JSON list),
`AWS_REGION`.

`LAB_ARENA_MODE` selects `off` (default: nothing starts, nothing is served),
`shadow` (full rounds, publication marked shadow, no reward basis is
governing), or `live`. The reward release itself is a separate step (plan
section 19 step 9) and is not enabled by this service.

Start: `python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8791`.
The driver thread advances the current round once per `--tick-seconds`; the
API and the driver share one process and one service role.

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

## 5. Daily round

1. The driver creates and advances rounds by wall clock; the admin CLI is
   for supervised steps only:
   `python3 scripts/lab_arena_admin.py status --round <id>`,
   `... advance --round <id> --expect-status <status> [--dry-run]`,
   `... cancel --round <id> --reason <closed reason> [--dry-run]`.
2. Submissions arrive through `POST /arena/v1/submissions`; admission
   (`lab_arena/admission.py`) builds each package offline from the pinned
   base image and wheelhouse, screens it on a floor runner, and freezes or
   rejects it under a published rule.
3. The benchmark is generated after the submission cutoff and committed by
   ordered root before stage 1 opens. A root that differs at read time
   cancels the round (`benchmark_root_changed`).
4. Stage 1 scores all participants on 20 ICPs; the
   top 10 plus the king enter stage 2 for the
   remaining 30. Infrastructure gaps cancel the
   round; model-caused failures score zero rows.
5. Publication writes the signed public bundle, the signed publication
   document, and the signed reward basis. Publication re-scans every frozen
   source archive; a sanitizer failure keeps the round unpublished.

Standalone verification of any published round uses
`lab_arena.verify.rebuild_round(public_bundle, signing_key_document)` with
only the public bundle and the round's signing key document.

## 6. Go-live gates

- Seven consecutive shadow rounds published with the throughput gate
  (`ArenaService.shadow_report`) satisfied, including at least one round at
  the maximum challenger count.
- Scorer egress enforced with the nftables ruleset from
  `lab_arena.egress.scorer_nftables_ruleset` on the scoring host.
- Runner floor sized from the shadow timing reports before the pilot.
- Paid pilot, then the reward release (plan section 19 step 9), which is the
  first change to files outside `lab_arena/`, `scripts/`, and `tests/`.

## 7. Invariants to re-check after any change

- `python3 -m pytest tests/lab_arena -q`: the boundary tests prove no
  Arena module imports `gateway.tee` or `gateway.db`, no measured package
  imports `lab_arena`, and the enclave allowlists exclude it.
- Operation table hash `sha256:e466eafdc20d20b33feba8dd99544759e2dd66b1a4bcf05da35e61d1d7187abb` and price list
  hash `sha256:6784881f2a79a0d758c773cc47a0630d495dba0c972e32a949e793f33bba41e9` are pinned on every round
  configuration; changing either changes the round identity.
- No secret value may appear in any table, object, event, log, or public
  bundle; the round tests inject canary provider keys and assert absence.
