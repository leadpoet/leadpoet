# Locked SN71 collection

Miners enroll an OpenRouter funding key for operator-run synthetic sales workflows. The initial harness is the reviewed source in [sales-agent PR 7](https://github.com/leadpoet/leadpoet-sales-agent/pull/7). The host freezes the source revision, scenario pack, generator schedule, rubric, tags, judge versions and routes. The assigned generator handles every model step, including Tyche child sessions.

## Enrollment

The pilot uses **your OpenRouter credits and pays no emissions**. Leadpoet pays for verification. Choose one scope per hotkey. A comparison funds one assigned open model on a simulated sales task. The experiment scope funds either a continuation from a smaller model’s mistake or repeated attempts from an approved task’s initial state.

1. Use a registered Finney subnet 71 hotkey and the miner runtime from [Install and register](../README.md#install-and-register). If you already have the miner dependencies, update your checkout, activate its miner environment, and run `python -m pip install --no-deps -e .` from the repo root. This refreshes the installed `leadpoet` command. The examples use wallet `miner` and hotkey `default`. Substitute your names or add `--wallet-path` for a custom location.
2. Create a dedicated funded [OpenRouter API key](https://openrouter.ai/settings/keys) and a [read-only management key](https://openrouter.ai/blog/announcements/scoped-management-keys/) for the same workspace. Disable prompt and output logging, sharing, and external trace broadcast. The gateway uses the management key once to verify ownership and stores only the encrypted runtime key.
3. Choose a scope and read the published terms. Use one of these commands:

   ```bash
   leadpoet collection-terms --task-scope synthetic_workflows
   leadpoet collection-terms --task-scope synthetic_corrections
   ```

4. To enroll, copy the full `terms_sha256` value for your chosen scope. Run the matching command below, replacing `DIGEST` with that value. The example authorizes up to **$0.30 per rolling 24 hours**. Choose your own cap. A smaller cap may end a run early. At the two hidden prompts enter the runtime key and management key. For unattended use, supply `OPENROUTER_API_KEY` and `OPENROUTER_MANAGEMENT_KEY` in the environment.

   ```bash
   leadpoet collection-register --mode pilot --task-scope synthetic_workflows --wallet-name miner --hotkey-name default --daily-cap-usd 0.30 --accept-terms DIGEST
   ```

   ```bash
   leadpoet collection-register --mode pilot --task-scope synthetic_corrections --wallet-name miner --hotkey-name default --daily-cap-usd 0.30 --accept-terms DIGEST
   ```

5. Save the returned registration ID. The worker polls every five minutes. The default limit is one new slot per funding owner per UTC day. Check your enrollments and jobs with the same hotkey:

   ```bash
   leadpoet collection-statement --mode pilot --wallet-name miner --hotkey-name default
   ```

   Each job shows confirmed miner charges and unresolved reservations in micro-USD (1,000,000 = $1). Match the charges to your dedicated key’s OpenRouter activity. A pilot quote is zero. To switch scopes or stop new calls, use the registration ID from your statement. Calls already dispatched may still be billed.

   ```bash
   leadpoet collection-revoke --mode pilot --wallet-name miner --hotkey-name default --registration-id REGISTRATION_ID
   ```

Paid assignments require fresh live consent after the pilot qualifies.

For setup help or to volunteer for a blind workflow review, use the [public pilot thread](https://github.com/leadpoet/leadpoet/issues/1343). Reviewing does not require a wallet or API credits. Submit provider keys only through the local miner CLI.

The six slots in each comparison pack use the same scenario and the approved generator routes below. Each subsequent complete group advances to a fresh scenario. A provider receipt records known spend even when the run fails. Unknown bills retain their reservation. The worker cannot issue the same paid request identifier twice. Synthetic tools run locally. The miner funds the billed generator calls. Private verifier calls use an owner-controlled credential and count against the job budget.

A verified model-only 429 may receive one replacement on its original pilot slot. The first charge can stay unknown. The worker reserves it in full and subtracts it from the replacement budget. This requires an authenticated refusal before generation and provable absence of hosted services or per-request charges. It makes no assumption about the eventual billed amount. [OpenRouter pricing](https://openrouter.ai/pricing) and [error semantics](https://openrouter.ai/docs/api/reference/errors-and-debugging).

An unresolved miner charge retains its original reservation. If OpenRouter supplied a generation ID on a finished job, the gateway operator can run `python -m scripts.reconcile_collection_generation --job JOB_ID --request REQUEST_ID --environment-file GATEWAY_ENV`. This reads the original generation on its authorized account and records the exact charge once. A missing, revoked or unpriced generation keeps its reservation. The job's outcome and earnings stay tied to the original execution.

### A witnessed route change in a private reference pack

For a capacity outage, a private synthetic pilot can use a separate immutable successor. It pins an approved alternative endpoint for the same generator revision and precision. Its `reference_successor` field identifies the original batch ID, the next unassigned slot and a private owner-funded test receipt. All other frozen batch settings must match. The database owner records the successor and verifies that the parent has no running lease and its next real slot still matches. The previous batch then closes to new claims. Historical jobs stay under their original IDs, and the next claim uses the original slot number, model, scenario, spending cap and daily allowance.

The first witnessed alternative uses CoreWeave FP8 for DeepSeek V4 Pro-0813. The worker preflights its current strict zero-retention endpoint and pins its route on every call. The successor starts with private reference inspection. It cannot price work or contribute a mixed-route qualification cohort.

A reference successor keeps its parent’s unassigned slot numbers and identifies the exact owner-funded route witness. Its rows cannot qualify a judge panel, calibrate a price or supply missing parent runs. To measure the same route independently from slot zero, freeze a fresh complete pilot manifest with the witnessed route in `routes` and its exact `route_witness_sha256`. It receives a new `rq-` batch ID. The worker verifies the original route witness before funding a job.

After independent qualification, the trajectory exporter can read the successor’s actually completed slots under their own signed consents. It verifies the owner witness, matching route, model revision, split, policy, budgets and settled generator costs in the complete qualification cohort. Training metadata retains the parent ID, absolute slot, successor ID and owner witness. Pending jobs stay out of that snapshot.

## Running one batch

Use an approved operator environment on the gateway. Source revisions are in `lab_arena/collection_workflow.py`. The public champion and authorized private Poet9B Git checkouts must contain those exact commit objects. The worker materializes the source trees from Git. The private checkout supplies guarded scenario generation, reference answers and deterministic graders.

1. Create a private batch JSON with `name`, `mode`, `seed`, a list of distinct `scenario_ids`, an exact provider tag for each of the six generators and two judges in `routes`, and each generator's `budgets_microusd`. The maximum batch liability is bounded by the sum of its scheduled job caps. The worker hashes these settings into the batch ID. Synthetic IDs include C26-s2027, K02-s2027, B1-s2027, E4-s2027 and P5-s2027. Guarded builders reject held-out seeds, products and tool formats. All synthetic worlds operate on simulated files, people, inboxes and CRM records.
2. Store the private rubric and examples outside Git. Use the format in the [judge contract](judge-output-spec.md). Supply `control_slots`, `drift_slots` and `control_artifacts` alongside a private case pack for a release pilot. Empty maps allow a preliminary reference batch. See the contract for freezing independently labeled planted cases and 30 previously recorded drift cases. A release pilot requires 100 judged ordinary runs spanning the six models, both judges on the pilot, and 50 actual blind human marks.
3. Run `python -m scripts.run_locked_collection --preflight --batch BATCH --prompt PROMPT --control-pack CONTROLS --champion-repo AGENT --poet-repo POET`. This validates and builds the private inputs offline. Separately test actual routes on synthetic content under the approved retention, training-use and price policies before opening funding to miners. Catalog presence is insufficient evidence of an available route.
4. In the approved gateway environment with `LAB_ARENA_COLLECTION_MODE=pilot`, run the same command without `--preflight` and with `--max-jobs 1` to execute one available funded job. Later invocations use the durable batch/slot record. The operator sets the daily slot limit in `collection_dispatch_settings.daily_slots_per_owner` (default 1, maximum 4,096). The limit applies to each hotkey, funding owner and funding key per UTC day and mode, across both task scopes. An active job must finish before the same hotkey, owner or key can start another. A failed assignment counts toward the daily limit. A settled pilot failure can use one replacement on the same slot, with only its remaining budget. The signed rolling 24-hour spending cap still applies. Confirm progress from persisted artifacts, statements and receipts. Keep the reviewer private key and private judge inputs off miner accounts.
5. Miner-funded enrollment covers synthetic scenarios only. Live Tyche research requires separate rights for the sources, credential delegation and zero retention across search providers. For an authorized operator research run, build a private index with `scripts.prepare_collection_research_index`, bind its digest in `research_index_sha256`, and pass `--research-case-index INDEX --research-stdio` on the gateway. Connect its binary streams to `python -m scripts.collection_research_worker` on the validator using `python -m scripts.collection_relay --worker-commands COMMANDS` on the authorized operator machine. COMMANDS is private JSON with gateway and validator command argument lists for their existing authenticated SSH/SSM aliases. Use the normal host release environment, exact scorer image cache, pinned dependency tree and champion checkout. A research process runs in gVisor with the ordinary broker and web sockets. A case with missing validation evidence stays out of training until it has sufficient evidence.
6. Build the private review bundle with `python -m scripts.collection_batch bundle`, prepare the blind sample with `scripts.collection_review sample`, and have the reviewer sign the completed marks. Keep the reviewer key off the worker. Before exporting a comparison or experiment, verify each funder’s signed rights and the release checks in the [judge contract](judge-output-spec.md). Older comparison enrollments permit private review only.

A failed pilot generation can receive one replacement on the same scenario and model with a new job ID. Its spending limit is the original slot limit less the confirmed cost of the first attempt. While a bill is unresolved, other slots can proceed. The miner's statement and the private review keep each attempt and its receipts. Only completed, evidenced runs count toward pilot calibration.

A verifier outage after generation preserves the original trace and any valid judge record. The trace stays out of training until its required judgments are complete. For the first staged live job, the operator can retry verification with `--rejudge-ready JOB_UUID ATTEMPT_UUID` against the same batch and prompt. Reusing the attempt ID protects against duplicate dispatch. The retry uses the operator's key and the job's remaining budget. The original fixed quote can enter settlement once, after the usual cutover checks.

The supplied `deploy/leadpoet-collection-pilot.timer` runs the existing worker on the gateway every five minutes. It handles one job at a time. Install the matching service and timer under systemd after verifying the pinned runner and the private batch, judge prompt and source repositories at the paths in the service. Start the service once and inspect its result before enabling the timer. The batch stops issuing jobs when its six slots per scenario are filled. Removing the timer stops polling. Revocation prevents new calls on the enrolled key.

## Activation and settlement

Apply collection migrations 720–729, 732 and 733 in numeric order through the [reviewed RDS migration channel](arena-rds-migrations.md). Deploy all required validator, signer and gateway revisions through the [paired release controller](arena_paired_restart.md). Confirm native routing and data readback before opening paid collection. The existing configuration helper supports a scoped `--collection-mode pilot` check/apply. Prepare each approved live batch within one task family so its model prices and caps come from one measured pilot class.

Run `python -m scripts.collection_batch prices` on the verified, signed pilot evidence. Champion retirement closes legacy submissions and routes unallocated emissions to the designated burn hotkey. Collection qualification continues during this period. After the pilot qualifies, stage the approved prices under the [validator weights procedure](arena_normal_validator_weights.md). The first accepted, miner-funded live job activates collection rewards for the next epoch. The worker can recover its completed result after a restart. The subnet then quotes each verified completed job 125% of measured pilot cost for that class and model. Payout weights are proportional to those fixed quotes. An epoch with insufficient emissions scales every quote by the same factor. Surplus goes to burn. Each job contributes once, with no carry-forward.

## Route check on November 23, 2026

These generator routes passed owner-funded, synthetic tool calls and continuations on both Chat Completions and stateless Responses. Each request pinned the endpoint, disabled fallbacks and required author-approved text distillation, zero retention and denied data collection. Every selected endpoint reported no safety retention. The table shows actual cost across the four successful calls and the number of assistant turns with readable reasoning.

| Slot | Generator | Required endpoint | Cost (USD) | Turns with reasoning |
| --- | --- | --- | ---: | ---: |
| 1 | Kimi K3.1 Pro | `deepinfra/bf16` | 0.00556890 | 4/4 |
| 2 | DeepSeek V4 Pro-0813 | `deepinfra/fp8` | 0.00355640 | 4/4 |
| 3 | Mistral Large 4.0 | `mistral/zdr` | 0.00143360 | 4/4 |
| 4 | Qwen3.8-27B | `deepinfra/fp8` | 0.001410150 | 4/4 |
| 5 | Nemotron 3.5 Super VL | `deepinfra/fp4` | 0.00072340 | 4/4 |
| 6 | DeepSeek V4.1 Flash-1017 | `particle-ai/us/flex` | 0.000210405 | 2/4 |

All six returned the expected tool call and decision. The Responses case required reconciling several consent events. It counted original reasoning text from the same field the research recorder reads. Flash omitted reasoning on the simple Responses case and exposed it on the harder case. The measurements establish compatibility. Fixed fees require full miner-funded pilot jobs and signed human review.

The author licenses are [Kimi K3.1 Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/LICENSE), [DeepSeek Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/LICENSE), [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/af26e7869af601441d5cdf75a1272469ddc3d059/LICENSE), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0), [Nemotron OpenMDW 1.1](https://openmdw.ai/license/1-1/) and [DeepSeek Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/LICENSE). Text distillation applies to model-generated text. Rights to third-party documents, tool results and personal data require separate clearance. [OpenRouter routing policy](https://openrouter.ai/docs/guides/routing/provider-selection) · [DeepInfra terms](https://deepinfra.com/terms) · [Mistral commercial terms](https://legal.mistral.ai/terms/commercial-terms-of-service/) · [Particle terms](https://particle.ai/terms).

OpenRouter returned HTTP 404 for Inkling 2 Small, Tencent HY4 and GLM-5.3 under the author-permission filter, including earlier checks without a provider selector. An earlier check found no Fable 5.2 endpoint under zero retention. Generator route tests do not qualify a judge or authorize a cutover.

For a private qualification pilot, use Kimi K3 on `deepinfra/mxfp4` and Mistral Large 3 on `mistral/zdr`. Each generator is scored only by independent families. A paid batch needs successful blind calibration, a separate signed operator approval for live use and a staged price schedule naming that approval. See the [qualification procedure](judge-output-spec.md#qualification-and-split-evidence).

The validator can verify a newly discovered company using exact, passive, same-run source fetches when the source URL, company identity, dated signal and quoted facts all match. Its evidence packet identifies these observations separately from previously verified Arena results. Incomplete research and unsupported context remain visible for review. A checkpoint alone cannot earn the fixed fee: the worker must also prove Tyche finalized the run.

Use `scripts.collection_relay` for one research job per invocation. For unattended operation run the bounded command under the operator's service supervisor while the approved pack is active; the next invocation reads the next durable slot. Empty funding queues produce an idle result. Never forward a personal SSH agent or an OpenRouter/Deepline key into the sandbox.

The optional private manifest field `thresholds` contains `accept_sum`, `salvage_min_steps` and `agreement_percent`. Defaults of 10/15, three steps and 85% are provisional. A preliminary calibration batch can establish 30 real reference outputs before that sample is frozen and rejudged as drift controls. Sealed controls are verified against private immutable artifacts, use owner-funded judge calls, and produce neither training rows nor miner rewards.

One additional owner-funded synthetic integration check ran DeepSeek V4 Pro on B1-s2057 with the public workflow runner and original Poet9B grader, then assessed the resulting trace with Astra. The model completed seven assistant turns, passed every deterministic check and received an accepted Astra record. The eight billed calls totaled $0.12361139. This compatibility run did not release a training row or earn a miner payment.

The six candidate generators also completed the same B1 sales workflow with the pinned public harness, the original Poet9B grader and the normal output limit. Each passed all six deterministic checks and received two valid qualification assessments. Across 39 original assistant turns, 29 exposed reasoning. Generation cost $0.07311137. These were owner-funded checks. The private comparison records cannot authorize training exports, miner payments or a payout rate.

## Running a correction canary

A correction branches an approved open expert from the exact step where an untuned student first made the synthetic task irrecoverable. A fresh expert arm starts at the beginning of the same task. The experiment manifest freezes the question, pass mark, source cases, split, arms, repeats, model routes, reasoning settings, output-token and step limits, and total authorized spend. The manifest sets its slot limit with `max_slots`. The default is 12 and the maximum is 4,096. The total run budget must cover every scheduled slot. These are unpaid pilot assignments and require separate signed consent. Comparison packs retain their original pricing and schedule.

To measure pass rates, use `python -m scripts.prepare_collection_experiment prepare-expert --help`. Supply the frozen split, one or more scenario IDs and approved models, a repeat count and budget. The worker rebuilds and checks the original task before leasing a slot. Use a continuation arm when an authenticated student checkpoint is available.

The gateway publishes the released synthetic terms for the active mode. The existing configuration helper can close the scope with `off` or pin an exact text digest. Miners can inspect the published text with `leadpoet collection-terms --task-scope synthetic_corrections`. To opt in after reading it, run `leadpoet collection-register --mode pilot --task-scope synthetic_corrections --daily-cap-usd AMOUNT --accept-terms DIGEST`. A previous comparison registration must be revoked before enrolling the same hotkey under new terms. Transferring a miner’s rights in original model output cannot transfer a publisher’s rights in a third-party page.

On the training machine, import the five original Poet run files using `python -m scripts.import_collection_student --help`. The training custodian then runs the following command. The private Ed25519 signing key stays on that machine. The source must be a genuine original student run. Its weight revision is a commit or a SHA-256 digest. The split file lists disjoint training and held-out template IDs, rule pools and tool variants from the actual evaluation design.

```sh
python -m scripts.prepare_collection_experiment prepare \
  --source student.json --split split.json --private-key producer.pem --output canary \
  --producer PRODUCER --producer-run-id RUN_ID --student-revision WEIGHTS_REVISION \
  --experiment-id corrections-v1 --model deepseek/deepseek-v4-pro-0813 \
  --question "Do corrections improve held-out sales task completion?" \
  --pass-mark "Positive lower bound against untuned and own-passing baselines" \
  --terms-sha256 APPROVED_DIGEST --cap-microusd 300000 --max-steps 20 --max-generated-tokens 8192
```

The gateway verifies the source against its independently pinned `LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE`. The operator verifies the private pack with `scripts.run_locked_collection --pilot-only --preflight`, passing the normal rubric and source repositories, `--batch canary/correction-batch.json` and `--experiment-inputs canary/correction-inputs`. Stage the protected inputs beside the existing pilot worker. Publish the preflighted manifest last. The same timer tries one correction assignment before each qualification pack assignment. An empty correction queue leaves that poll available for the existing pack. Removing the optional manifest stops new experiment assignments.

Use `scripts.collection_batch experiment-audit` with `--batch`, `--prompt` and a private `--output` ZIP to inspect the generator, original student, ordered event journal, checks, claim evidence, receipts and interrupted attempts. The audit includes no approved training targets. Use `corrections` with the same arguments plus `--panel-batch`, `--panel-prompt`, the signed human `--receipt` and independent `--qualification` to release eligible correction pairs. It reconstructs the approved panel and the original experiment objects. The output keeps the rejected action and the separately labeled expert target beside their common visible prefix. The release identifies the private checkpoint by hash and reports confirmed miner cost per kept row.

Poet trains and evaluates the dataset independently. Its signed `collection.poet-evaluation.v1` record identifies the release digest, untuned, tuned and own-runs revisions, server, held-out set, split and evaluator code revision. Each paired task supplies its hashed ID, SHA-256 hashes of the three held-out partition identities in `partition_sha256`, and Boolean `base`, `tuned`, and `own_runs` outcomes. Hash each partition identity's exact UTF-8 string, with the `sha256:` prefix. Use `scripts.prepare_collection_experiment sign-feedback` on the evaluator's machine. Then run `scripts.collection_batch training-feedback --batch BATCH --prompt PROMPT --dataset-metadata METADATA --evaluation SIGNED_RESULTS --output RECEIPT`. The subnet links the verified results to the exact release and records paired task-success lift and its interval against both baselines.

The research relay uses the existing worker observation socket and immutable private collection store. Before sending an input, the gateway checks a deployment-pinned approval against the funding registration, signed terms, batch, scenario, model and input. The approval must identify the applicable delegation, zero-retention, source-reuse and personal-data records. Each licensed provider request is frozen by digest. The worker restricts direct HTTPS access to the approved hosts. Existing synthetic enrollments do not carry this research permission.

The private research journal records each approved provider request and result and each HTTP observation as it arrives. Records bind the grant, job, sequence and original body digest. A reported page is evidence of what the sandbox observed. Direct TLS connections can carry unobserved responses, so this path holds those runs out of automated training release. A capture gap, redaction or unavailable right also holds release. No research grant is installed by deployment of the code.

### Pricing a compensated correction class

Compensated ordinary collection must be active before a correction class can earn rewards. A paid correction is a separate bounded assignment with a separately approved price, legal text and live signature from its funder. The existing synthetic pilot remains unpaid.

1. Before running the pricing pilot, set `min_pricing_runs` in its manifest, or supply `--min-pricing-runs` to the private preparation command. Use one continuation arm and a single comparable template, student weight revision, rule pool, tool variant, expert route, step/token limits, reasoning settings, split and cap. Run the pilot. Wait for all attempts and provider charges to settle and complete the independent panel and training-rights review. Failed model decisions from authentically completed jobs contribute their measured generation cost.
2. Prepare the private prospective manifest for the same measured class with `mode` set to `live`, the pilot's batch ID in `pricing_pilot`, and the live consent digest in `terms_sha256`. Omit `min_pricing_runs` from this manifest. Each tariff covers one continuation arm and the fixed slots in its approved manifest. The original pilot can price one such batch. A new class requires new measurements.
3. Use `python -m scripts.collection_correction_tariff prepare --help` for the required private inputs. The command rebuilds the registered dataset and its signed human and panel approval, checks class equivalence, and computes the fixed quote as the ceiling of 1.25 times average eligible confirmed miner cost, in micro-USD. The open panel approval must explicitly cover compensated use and match the active ordinary cutover.
4. The pricing authority reviews the full candidate and signs it offline with the command's `sign` action. Configure its public Ed25519 PEM at `LAB_ARENA_COLLECTION_TARIFF_PUBLIC_KEY_FILE` in the operator environment. The `stage` action independently prepares the candidate again, verifies the signed document, and installs it through the authorized RDS owner channel. It requires the same private inputs, the signed approval, the local tunnel port and the verified CA file. The worker's service credential can read the approval. The installation is immutable and an identical retry is idempotent.
5. Compensated enrollment opens when the live mechanism has an available approved quote. The miner selects `--task-scope synthetic_corrections --mode live` and signs the new terms. Available class quotes and caps appear with those terms. An ordinary or pilot signature cannot fund these jobs. Enrollment continues to honor the one active registration per miner and mode.
6. Use the same worker command and source stores as the pilot with the approved live manifest. On the live ordinary worker, the optional `--next-correction` manifest must also be live. Keep `--pilot-only` on unpaid workers. The runtime checks the original evidence, pricing signature, finalized ordinary cutover epoch and signed funding owner before dispatch. Removing the optional manifest stops new correction assignments. The existing settlement counts a completed live correction once at its preassigned quote, including when the model makes a mistake. Shortfalls scale in the shared collection pool. Access to prior finalized payments is unchanged. Reserve any decision to expand the arm for actual held-out lift evidence.

### Recording comparison workflows

New synthetic comparison manifests use `collection.workflow-events.v2` in `capture_version`. The core runner preserves verified provider replies, visible tool results, usage, the original workflow output and per-step native checks in private storage. Interrupted and capped attempts retain their recorded prefix. Sensitive material is withheld and the private record identifies the reason. Previously frozen versions keep their source pins and batch identities.

New experiment manifests set `max_tokens` per call and `max_run_tokens` for the whole run. The preparation command defaults the latter to 65,536. Unknown token counts stay unknown in the data and reserve the requested allowance against the run limit. A truncated final reply is recorded as unanswered. Long judge inputs are read in bounded, byte-indexed parts before a whole-run verdict. Incomplete evidence or a bounded-reader limit holds that verdict, and the original size and disposition are recorded.

### Using a gateway-funded student as the source

A gateway-funded student can provide the original failed step. The trusted operator captures the five original Poet files and the actual owner-funded requests and responses in a private, write-once artifact. The fixed namespace is `collection/owner-students/RUN/native-SHA256.json`. Its JSON has schema `collection.gateway-native.v1`, the original UTF-8 files under `original`, and the existing `leadpoet.collection.owner-student.v1` receipt pack under `provenance`. No miner upload endpoint supplies these objects.

Run `python -m scripts.prepare_collection_experiment prepare-gateway --help` for the operator command. It takes the run ID, private artifact SHA-256, frozen split file, exact repository checkouts, canary settings and protected gateway environment. It replays the original student and looks up every generation on the owner’s OpenRouter account. The ID, model, provider and confirmed cost must match. No inference runs during preparation. The worker verifies the original bytes and saved readbacks again before a miner lease. Release uses the same provenance check. External Poet sources continue to use the signed producer handoff. The funding consent, review gates, training-only split and payout rules apply to both paths.

### Private participation watch

The gateway's independent `leadpoet-collection-watch.timer` checks participation every ten minutes. It counts active enrollments, jobs, immutable artifact references, settled miner bills, pending launch and cutover. For each of primary, Rizzo and Yuma it compares the finalized on-chain vector at one block with the signed state for that block's epoch. From the first retired epoch until paid collection starts, the watch verifies that the original designated burn hotkey receives the entire signed weight on all three validators. Actual miner emissions are reported at the closing chain distribution. After paid collection starts, the watch checks the signed collection basis against the eligible job quotes, checks each funder has a weight, and verifies a zero champion share on all three validators. Former champions can earn through funded jobs. An accepted job, a cost receipt and a published weight have distinct meanings. Candidate review counts include controls. Confirm genuine ordinary cases and signed human marks with the private review commands above.

The watch also counts public terms, registration, revocation and statement HTTP responses from new entries in the existing edge logs. Counts start at installation and include operator probes. Log rotation or an oversized window is marked as a coverage gap. Requests, query strings and identities stay out of the snapshot.

Read `~/.local/share/leadpoet/collection/watch/status.json` on the gateway for current counts and block details. Read `journalctl -u leadpoet-collection-watch.service` for transitions, including an arm held while the worker reported success. The operator Mac runs `scripts/collection_watch_desktop.py` through launchd. It sends changed states and stale or unavailable gateway reads to Notification Center and saves the latest private snapshot under `~/Library/Application Support/Leadpoet/collection-watch/latest.json`. For a first billed run, inspect its receipt and immutable artifact. For a review threshold, build the private review bundle. For a weight difference, follow the [validator verification checklist](arena_normal_validator_weights.md).

Install migration 742 before enabling the gateway watch. Use a reviewed detached checkout at the service's working directory and install its matching service and timer. Rollback is `systemctl disable --now leadpoet-collection-watch.timer` and unloading `com.leadpoet.collection-watch` on the operator Mac. Both observers perform read-only checks against production.

To preserve the first funded evidence automatically, install the reviewed private `first-funded-evidence` reader at the path in `deploy/leadpoet-collection-watch-first-funded.conf`. Install that file as `/etc/systemd/system/leadpoet-collection-watch.service.d/first-funded.conf` and reload systemd. Each watch pass can then retain one verified private receipt per arm in the gateway's `watch/first-funded/` directory. The reader uses the miner's settled billing and that same job's immutable objects. It can run again after an incomplete receipt or restart. Remove the drop-in and reload systemd to disable this readback.

### Exporting full synthetic trajectories

Use the operator’s existing `python -m scripts.collection_batch export` command with the registered `--batch`, its frozen `--prompt`, a new private `--output` ZIP and the independent panel’s `--panel-batch`, `--panel-prompt`, `--receipt` and `--qualification`. Both public verification keys must be installed. Six-model comparisons require checkpointed capture and the training-enabled terms signed by each funder. Earlier comparison enrollments permit private audit. Correction and repeat experiments use the signed grant for their mode and scope.

Each `trajectories/<run>.jsonl` record carries the original model-visible messages and tools, grounded step and outcome labels, finish reason, measured token counts, settled miner cost, model and source revisions, and repeat-group ID. Unknown usage or unverified outcomes are null. Unsafe, missing or unverified prefixes have an explicit hold and no released transcript. Controls, judge prose, hidden world state and reference answers stay out of the training rows.

For the same starting state, `preferences.jsonl` references the complete passing and failing trajectories by path and digest. Verified student corrections also include both actions in `pairs/<run>.jsonl`. Inspect `metadata.json` for the included observations and dataset digest. The existing `corrections` command retains its reviewed pair-only format. The `training-feedback` command accepts an identified release and genuine, signed Poet evaluation results for either recipe.

The existing one-job pilot poller accepts an optional private `--next-reference` setting. It checks the correction canary first. The setting names one exact predecessor and one complete 10-scenario reference manifest. Each owner’s signed cap and daily quota still apply. A follow-up can start only after the successor’s real slots are terminal, its charges are settled, no affordable pilot replacement remains, and at least one complete paid run passes the original task, claim, artifact and judgment checks. It uses the normal collection worker for each new job. Qualification drafts, live prices and payments remain operator actions.
