# Research Lab Arena operator guide

The Arena is a simple agent-bundle competition. Each round uses the same
twenty ICPs as that day's public-baseline rebenchmark: ten in stage 1 and ten
in stage 2. The Arena reads only the current active set through one database
function; it does not generate a separate benchmark.

## Competition boundary

A miner submits one public OCI image by tag or digest. At intake, the Arena:

1. pulls the source anonymously;
2. resolves a tag to one immutable manifest digest;
3. checks the public size, layer, and Linux/AMD64 limits; and
4. copies the image bytes to the Arena registry.

The submitted tag or digest is only an image location. It does not determine
identity, rank, or the winner. The runner ignores OCI command, environment,
and work-directory settings. It always starts the miner bundle with:

```text
/agent/run
```

The bundle reads `/input/icp.json` and writes `/output/companies.json` by the
documented schemas. It has no direct network interface. Provider calls go
through the runner socket and the Arena broker. A miner can use any internal
harness, model, prompts, packages, routing, or orchestration.

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
- `LAB_ARENA_SERVICE_JWT`
- `LAB_ARENA_BUCKET`
- `LAB_ARENA_CHAIN_ENDPOINT`
- the three host provider keys listed above
- `LAB_ARENA_REGISTRY_REPOSITORY`
- `LAB_ARENA_REGISTRY_USERNAME`
- `LAB_ARENA_REGISTRY_PASSWORD`
- `LAB_ARENA_SCORER_IMAGE`: a public tag or digest; startup resolves it to a
  digest for trusted scoring
- `LAB_ARENA_RUNNER_HOTKEYS`: the runner hotkeys allowed to claim work
- `LAB_ARENA_BASELINE_HOTKEY`: the registered hotkey that submits the initial
  public baseline

Common optional values are `AWS_REGION`, `LAB_ARENA_NETUID`,
`LAB_ARENA_NETWORK`, `LAB_ARENA_CHAIN_TIMEOUT_SECONDS`,
`LAB_ARENA_DAILY_CUTOFF_UTC`, `LAB_ARENA_MAX_CHALLENGERS`,
`LAB_ARENA_MAX_IMAGE_BYTES`, `LAB_ARENA_POOL_PERCENT`, and
`LAB_ARENA_BANNED_HOTKEYS_PATH`. `LAB_ARENA_REWARDS_ENABLED` defaults to
`false` and is frozen into each new round. `LAB_ARENA_SIGNING_KEY_ID` is
needed only when a live, reward-enabled published round is activated.

Apply `scripts/179-lab-arena-v1.sql` and
`scripts/181-lab-arena-daily-icp-source.sql` with the database owner before
service startup. Then check the service wiring:

```bash
python3 scripts/run_lab_arena_service.py --check-only
```

Start the service:

```bash
python3 scripts/run_lab_arena_service.py --host 127.0.0.1 --port 8791
```

Create rounds automatically with `LAB_ARENA_DAILY_CUTOFF_UTC`, or create one
manually:

```bash
python3 scripts/lab_arena_admin.py create --cutoff 2026-09-05T00:00:00Z
```

## Runner configuration

Each runner needs Linux AMD64, root access for the sandbox mounts, and an
executable gVisor `runsc`. It also needs:

- `LAB_ARENA_API_BASE_URL`
- `LAB_ARENA_WALLET_NAME`
- `LAB_ARENA_HOTKEY_NAME`
- `LAB_ARENA_RUNNER_WORK_DIR`
- `LAB_ARENA_RUNSC_PATH`

`LAB_ARENA_MAX_PARALLEL_RUNS` and `LAB_ARENA_ROUND_ID` are optional. Provider
keys, database access, registry push access, and the signing key stay on the
service host. Start the runner with:

```bash
python3 scripts/run_lab_arena_runner.py
```

## Miner flow

Create and sign a submission with `scripts/lab_arena_miner.py`. The body has
only the image reference and one public benchmark/reuse consent:

```json
{
  "image_reference": "ghcr.io/example/agent:latest",
  "consent": {"public_rerun": true}
}
```

See the repository README for example bundles and schema links. An example is
documentation only; it is not part of admission or scoring.

Before the first round cutoff in each mode, the baseline hotkey submits the
public baseline through this same signed request and image-admission flow.
Freeze fails if that submission did not reach `accepted`. The first accepted
baseline is the incumbent. After a same-mode winner exists, that incumbent is
carried forward normally, or its hotkey can submit a fresh bundle for the next
round.

## Rewards and independent disable controls

The Arena is beside the old Research Lab reward path. With Arena rewards off,
the old path continues without an Arena champion allocation.

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
of the old Research Lab system.

## Focused checks

For Arena changes, run the directly affected tests and:

```bash
git diff --check
python3 -m py_compile lab_arena/*.py scripts/lab_arena_*.py scripts/run_lab_arena_*.py
```

Never place provider keys, database tokens, registry passwords, or signing-key
material in source, test fixtures, command output, or public results.
