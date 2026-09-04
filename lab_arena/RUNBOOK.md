# Research Lab Arena operator guide

The Arena is a simple agent-bundle competition. Each round uses the current
daily set of twenty qualification ICPs: ten in stage 1 and ten in stage 2.
The baseline and miners use that one set and one scoring path.

## Competition boundary

A miner submits one local source directory. The helper:

1. requires `harness.py`;
2. creates one sorted, size-limited archive with normalized metadata;
3. gets a private upload target from the Arena;
4. uploads the archive; and
5. signs a final request so the Arena can validate and accept the bytes.

No Dockerfile, public registry, image tag, commit identity, receipt, source
digest, or release manifest is part of miner admission. The service validates
the declared archive size and safe source structure, then uses its own
submission ID for execution and recovery. The private source reference is
write-once. A miner can use any harness, model, prompts, packages, routing, or
orchestration behind the one callable.

The source contract is `harness.run_icp(icp) -> list[dict]`. `harness.py` can
define the function or re-export it from vendored source. The function must be
synchronous and have exactly one positional parameter. It cannot have
keyword-only parameters, `*args`, or `**kwargs`. It returns at most five
company objects. The public baseline README gives the full input and output
example:
[`leadpoet/pydantic-harness`](https://github.com/leadpoet/pydantic-harness).

Vendored Python modules run directly from the read-only source mount. An
optional `requirements.txt` can contain package names and version constraints.
The runner installs binary wheels only into a bounded cache and mounts those
dependencies read-only. It rejects URLs, local paths, nested requirements,
VCS dependencies, and source builds. The common trusted scorer image supplies
Python for every agent; it is not a miner image or a miner identity.

The organizer supplies one host key for each provider:

- `LAB_ARENA_OPENROUTER_API_KEY`
- `LAB_ARENA_SCRAPINGDOG_API_KEY`
- `LAB_ARENA_DEEPLINE_API_KEY`

The OpenRouter key is shared by bundle calls and judge calls. Only the
organizer configures provider keys on the host. The broker
permits any model in the organizer-fetched OpenRouter catalog that has usable
pricing. It still enforces the fixed call, token, cost, privacy, and time
limits. The trusted judge can use only its configured judge models.

A shared provider account failure, rate limit, or provider server failure is
an infrastructure failure. It does not give a miner a score of zero. A real
caller error, such as invalid request data, is returned to the bundle.

## Required service configuration

Set these values on the Arena service host:

- `LAB_ARENA_MODE`: `shadow` or `live`. Use `off` to disable the Arena.
- `LAB_ARENA_SUPABASE_URL`
- `LAB_ARENA_SUPABASE_ANON_KEY`
- `LAB_ARENA_SERVICE_KEY`: preferred production credential. Use a scoped
  `sb_secret_` API key whose JWT template role is `lab_arena_service`.
  `LAB_ARENA_SERVICE_JWT` remains a legacy fallback for parity environments.
- `LAB_ARENA_BUCKET`
- `LAB_ARENA_CHAIN_ENDPOINT`
- the three host provider keys listed above
- `LAB_ARENA_SCORER_IMAGE`: a public tag or digest; startup resolves it to a
  digest for trusted scoring
- `LAB_ARENA_RUNNER_HOTKEYS`: the runner hotkeys allowed to claim work
- `LAB_ARENA_BASELINE_HOTKEY`: the registered hotkey that owns each daily
  public baseline entry
- `LAB_ARENA_BASELINE_SOURCE_URL`: the public HTTPS PydanticAI source archive;
  it defaults to the `leadpoet/pydantic-harness` main-branch archive

Common optional values are `AWS_REGION`, `LAB_ARENA_NETUID`,
`LAB_ARENA_NETWORK`, `LAB_ARENA_CHAIN_TIMEOUT_SECONDS`,
`LAB_ARENA_DAILY_CUTOFF_UTC` (default `0`),
`LAB_ARENA_MAX_CHALLENGERS` (default `16`, hard limit `256`),
`LAB_ARENA_MAX_IMAGE_BYTES` for the trusted scorer image,
`LAB_ARENA_POOL_PERCENT`, and
`LAB_ARENA_BANNED_HOTKEYS_PATH`. `LAB_ARENA_REWARDS_ENABLED` defaults to
`false` and is frozen into each new round. `LAB_ARENA_SIGNING_KEY_ID` is
needed only when a live, reward-enabled published round is activated.

Apply `scripts/179-lab-arena-v1.sql` and
`scripts/180-lab-arena-daily-competition.sql`, then
`scripts/181-lab-arena-source-submissions.sql` and
`scripts/182-lab-arena-source-execution.sql`,
`scripts/183-lab-arena-miner-reward-basis.sql`, and
`scripts/184-lab-arena-scoring-failure-isolation.sql` with the database owner
before service startup. Then check the service wiring:

```bash
python3 scripts/run_lab_arena_service.py --check-only
```

Start the service:

```bash
python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8792
```

The service creates a daily round at 00:00 UTC by default. Set
`LAB_ARENA_DAILY_CUTOFF_UTC` to select another hour, or create one manually:

```bash
python3 scripts/lab_arena_admin.py create --cutoff 2026-09-05T00:00:00Z
```

## Runner configuration

Each runner needs Linux AMD64, root access for the sandbox mounts, and an
executable gVisor `runsc`. It also needs:

- `LAB_ARENA_MODE`: the same `off`, `shadow`, or `live` mode as the service;
  `off` stops the runner
- `LAB_ARENA_API_BASE_URL`
- `LAB_ARENA_WALLET_NAME`
- `LAB_ARENA_HOTKEY_NAME`
- `LAB_ARENA_WALLET_PATH` when the wallet is outside the default wallet path
- `LAB_ARENA_RUNNER_WORK_DIR`
- `LAB_ARENA_RUNSC_PATH`

`LAB_ARENA_MAX_PARALLEL_RUNS` (default `8`, maximum `8`) and
`LAB_ARENA_ROUND_ID` are optional. Provider
keys, database access, source upload access, and the signing key stay on the
service host. The runner needs read access to the organizer's common trusted
Python/scorer image. It downloads source only through the active run lease,
then mounts source, installed wheels, and the host-owned entrypoint read-only
inside gVisor. Start the runner with:

```bash
python3 scripts/run_lab_arena_runner.py
```

## Miner flow

Choose **Agent Competition** in `neurons/miner.py`. It asks for only the local
source directory, then archives, uploads, signs, and finalizes it. It never
asks for provider credentials, a Dockerfile, or an image tag. The same helper
can run directly:

```bash
python3 scripts/lab_arena_miner.py submit-source --source ./my-agent \
  --wallet-name default --hotkey-name default
```

See the repository README for example bundles and schema links. An example is
documentation only; it is not part of admission or scoring.

At the first round cutoff, the service automatically admits the configured
public baseline archive through the same source-admission checks. A temporary
download or object-store failure is retried. An invalid baseline prevents the
round from starting. Each daily round gets a new baseline download and uses
only that baseline as the score miners must beat. Prior winners stay in reward
history; they do not replace the next daily baseline.

## Rewards and independent disable controls

The Arena result is beside the retained reward settlement path. With Arena
rewards off, no Arena champion allocation is added.

If Arena rewards are enabled, the Arena and the reward-basis gateway must use
the same Supabase database. The published Arena basis must be visible to the
gateway/coordinator. Configure the Arena signing public-key hash and the Arena
reward enable flag on both the gateway/coordinator and validator as required by
the existing reward adapter. A missing, invalid, or unreachable governing
basis fails closed; it is not treated as an empty winner.

Competition publication is separate from reward activation. Publication
writes the participants, rankings, winner decision, and publication time
directly to the round row. It does not need KMS, an epoch read, a signed
receipt, a copied result bundle, or a replay. The driver later retries reward
activation for enabled live rounds, oldest first. Shadow rounds and rounds
created with rewards disabled cannot activate rewards later.

To disable only the competition, set `LAB_ARENA_MODE=off` and stop the Arena
service and runners. To disable only Arena rewards, turn off the Arena reward
flag on the gateway/coordinator and validator. Neither action requires removal
of retained reward history.

## Focused checks

For Arena changes, run the directly affected tests and:

```bash
git diff --check
python3 -m py_compile lab_arena/*.py scripts/lab_arena_*.py scripts/run_lab_arena_*.py
```

Never place provider keys, database tokens, registry passwords, or signing-key
material in source, test fixtures, command output, or public results.
