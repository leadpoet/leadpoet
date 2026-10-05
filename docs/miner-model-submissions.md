# Submit a competing model

The miner menu opens the **Submit Model** flow.

## Source and credentials

Fork the public [public baseline](https://github.com/leadpoet/leadpoet-sales-agent),
or provide another Python agent with this entrypoint in `harness.py`:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return up to five companies with ICP-fit and intent evidence."""
```

Put `LICENSE`, `LICENSE.txt`, or `LICENSE.md` beside `harness.py`. The file must
contain the complete [Tyche AGPL-3.0 license](https://github.com/gzaentz/tyche/blob/main/LICENSE).
The miner CLI validates the full text before any network request. The gateway
validates the uploaded bytes again before it stores credentials or admits the
model for review and scoring.

Use the baseline's input/output shapes and broker transport. You can change
the model, Python harness, prompts, and tool routing. Python dependencies can
be listed in `requirements.txt`; the runner accepts binary wheels, not source
builds or arbitrary dependency URLs. Other language runtimes are not supplied
by the current Python runner.

Choose **Submit Model**, then enter the source directory and these keys in
the masked prompts:

- OpenRouter API key for model execution and judging.
- OpenRouter management key that controls that API key.
- Deepline API key for provider calls.

The gateway checks the keys without changing the OpenRouter account. It
discards the management key after the check. It stores only encrypted runtime
keys. The OpenRouter runtime key must be usable and non-management. If
OpenRouter reports a numeric `limit_remaining`, it must be above zero. The
management key must control that exact runtime key; the gateway checks
`/api/v1/keys/{hash}` using the runtime key's SHA256 hash. A runtime-only key,
a management key in the runtime slot, a disabled key, or a key with no
reported credit is rejected. The gateway reports safe error codes and never
includes key values in them. Neither the validator sandbox nor the submitted
code receives real keys; the gateway adds them to approved provider calls for that
submission.
The miner pays those upstream charges. There is no organizer-key fallback.

| Error | Action |
|---|---|
| `openrouter_api_key_invalid` | Supply an enabled runtime key, not a management key. |
| `openrouter_api_key_no_credit` | Add credit or increase the runtime key's exhausted limit. |
| `openrouter_management_key_invalid` | Supply the management key that controls this exact runtime key. |
| `deepline_api_key_invalid` | Check that the Deepline key is active and correct. |
| `credential_validation_unavailable` / `credential_kms_unavailable` | Retry later; a dependency could not complete the check. |

Rejected admission codes can have the prefix `submission_rejected:`.

For automation, set `OPENROUTER_API_KEY`, `OPENROUTER_MANAGEMENT_KEY`, and
`DEEPLINE_API_KEY` through your secret manager, then run:

```bash
python3 scripts/lab_arena_miner.py submit-model --source ./my-agent \
  --wallet-name YOUR_WALLET --hotkey-name YOUR_HOTKEY
```

Do not put keys in command arguments or model source. An archive rejects every
file whose basename is `.env` or starts with `.env.`, except for the exact
templates `.env.example`, `.env.sample`, and `.env.template`. Those templates
must contain placeholders only; no archive may contain a submitted key.
For example, a secret-free loader named `.env.local.sh` is still rejected by
its basename; rename it to a neutral name such as `arena-env-loader.sh` before
including it.

Admission requires a registered miner hotkey and an open submission window.
If the chain no longer registers the selected hotkey, presign returns
`hotkey_unregistered`. Check registration before submitting and update the
wallet or miner configuration to a currently registered hotkey. Do not rotate
a registered key just because an older key was pruned. Keep the signing key
that owns an unfinished submission. Published results are public. A cancelled
round exposes its status, but keeps accepted work private.
The archive limits are 10 MiB compressed, 50 MiB unpacked, and 1,000 entries.
The result gives a submission ID and round ID. **Accepted means admitted, not
scored.** Validator execution and scoring follow through that round's queue.

## Duplicate models

The gateway checks a model against earlier accepted submissions from other
hotkeys in the same round, and against the current champion. The first accepted
submission has priority; reserving an upload earlier does not give priority.
An exact copy, including safely normalized Python comments and formatting,
returns HTTP 409 with `submission_rejected:duplicate_submission` at finalize.

Supported near matches enter the existing source review before execution.
This includes local variable and parameter renames and minor cosmetic edits.
The submitting miner's OpenRouter key pays for that single bounded review.
The review rejects a duplicate only with strong evidence that its behavior is
unchanged. Meaningful changes and uncertain comparisons can pass. A review
failure or missing credit remains an incomplete review, never a pass.

Private source comparisons stay inside the gateway. The review can receive
only the candidate's code and locally verified edit categories for a private
reference. It can compare full reference code only after that code is public.
An arbitrary rewrite of an undisclosed prompt cannot be judged from those
categories, so it is allowed rather than assumed to be a duplicate. No
comparison changes a round's existing disclosure time.

Near-duplicate rejections appear in the submission's code review as the
`duplicate_submission` category. An accepted upload still requires a passing
review before execution. The gateway verifies the admitted archive's hash
again when it supplies source to the validator.

## Recover after a provider credit failure

Add credit to the same provider account, or increase its exhausted key limit.
Then use the hotkey that owns the submission:

```bash
python3 scripts/lab_arena_miner.py retry-credit-failures \
  --round-id ROUND_ID --submission-id SUBMISSION_ID \
  --wallet-name YOUR_WALLET --hotkey-name YOUR_HOTKEY
```

This signed request creates attempt 2 only for a failed attempt 1 with a
gateway-verified, settled zero-charge credit error. The original execution or
scoring stage must still be open and before its frozen deadline. It uses the
existing stored keys and two-attempt limit. Top up before retrying: another
credit failure can use the remaining attempt. Recovery is not automatic.

The receipt reports `queued`, `replayed`, or `no_eligible`, with a run count.
`no_eligible` includes a safe reason. Old failures without the new zero-charge
proof, uncertain charges, accepted results, closed stages, and closed rounds
cannot use this recovery path. Existing spend still counts against the same
budget; a top-up does not raise Arena spending limits.

Credit errors are excluded from shared result caches. Their error and billing
records remain stored for audit and exact-call replay. A recovery uses a fresh
attempt, so it does not reuse the failed call's identity or delete its history.

## Submission status and results

Retry an unchanged archive with the same hotkey. The gateway reuses its upload
reservation. If an unfinished archive changes, the gateway keeps the old row
and bytes and assigns a new upload target. A late finalize for the replaced
reservation returns `submission_superseded`. One replacement is allowed before
the published replacement cutoff, including an accepted source that has not
started review or execution. The prior accepted source remains available until
the replacement is finalized. The duplicate check excludes that same hotkey's
prior source. The existing MD5 transport checksum prevents a
same-size changed archive from silently finalizing older bytes.

Use the returned IDs to read the result after the round publishes:

```bash
curl "$GATEWAY_URL/arena/v1/rounds/ROUND_ID/results/SUBMISSION_ID"
```

A published result includes its aggregate score. For a round frozen with
`after_scoring_day2_v1`, companies, run results, and per-ICP scores remain empty
until the published round reaches its submission cutoff plus 24 hours. The
benchmark needs the same time boundary and a published or cancelled state.
While a round is nonterminal, this
endpoint returns HTTP 403 with `results_not_public`. That response does not mean
the submission failed. Check the round status at `/arena/v1/rounds/ROUND_ID`;
do not submit again just to check progress. Earlier rounds without the marker
keep their original disclosure timing.

If a round is cancelled, accepted work remains persisted for private recovery
and audit, but the public result and source routes return HTTP 403. The round
does not publish aggregate scores, ranking, a king decision, or rewards. A
valid committed benchmark can still become public at the applicable disclosure
boundary. This does not publish miner outputs, run results, per-ICP scores, or
source from the cancelled round. Another round's submission ID does not grant
access.

Provider calls made while a round is running can incur the miner's upstream
charges even if a later infrastructure failure cancels the round. A cancelled
round does not publish a ranking and does not automatically refund provider
charges. When a required judge assignment exhausts its infrastructure retries,
the driver cancels the remaining work; already dispatched provider calls can
still incur charges. Arena reward activation is a separate setting: disabling
rewards prevents champion allocation, while scoring and competition execution
remain separate configuration behavior.

## Decision summaries

New or updated model harnesses should report short decision summaries through
Arena's existing checkpoint helper. Use the same format as the public baseline:

```python
from lab_arena_checkpoint import log_decision

log_decision(
    objective="Check whether this company fits the requested industry.",
    candidate="example.com",
    decision="accept",
    evidence=["https://example.com/products"],
    rationale="The product page identifies the requested software category.",
    next_action="Check the required intent signal.",
)
```

Report the initial approach, material candidate accept/reject/defer decisions,
and the final outcome. Explain the decision briefly using the evidence actually
observed. Do not send hidden reasoning, credentials, full page bodies, or invented
explanations. Supported decisions are `investigate`, `accept`, `reject`, `defer`
and `finish`. Evidence references are informational model claims; the judge
continues to verify company evidence independently.

The helper sends a bounded record to the local worker socket. The shared
validator batches it into the existing private trajectory table with the
lease-derived round, submission, ICP, attempt and validator identity. Models
cannot set those identities. No Supabase or provider key is needed in the model
or validator, and no new environment setting is required.

Logging is best effort and does not change company outputs, admission, scoring,
provider quotas or rewards. A return value of `False` means the record was not
accepted by the local worker; it is not a provider failure and must not cause a
research retry. Old validators can continue to run models that use the helper
with a guarded import; upgraded validators are required to capture decisions.
The trajectory includes an explicit coverage summary when a model supplies no
records or reaches the capture limit. Existing frozen submissions are not
rewritten or assigned explanations they did not supply.

See [the decision capture contract](arena-decision-logging.md) for exact limits,
privacy, failure behavior and examples.

## Operator setup

Use this deployment order. Do not apply migration 185 while a service that
requires schema 184 can still restart.

1. Keep `LAB_ARENA_CREDENTIAL_KMS_KEY_ID` empty. Deploy this service version,
   which supports schema 184 or 185 with miner admission disabled. Verify that
   baseline operation is unchanged. Do not interrupt another active deployment.
2. Check that no older accepted or running miner submissions remain. Those
   submissions do not have runtime credentials and cannot use organizer keys.
   Let them finish under the old version before cutover; do not cancel them
   without operator approval.
3. Apply `scripts/185-lab-arena-miner-credentials.sql` after migration 184.
   Verify the service and baseline again before enabling miner admission.
4. Set `LAB_ARENA_CREDENTIAL_KMS_KEY_ID` to an immutable symmetric KMS key ARN
   with gateway Encrypt/Decrypt access, then restart the service through the
   normal deployment path. Keep the same key ARN for existing ciphertexts.
   Native KMS key-material rotation is supported; changing the key ARN or
   retargeting an alias needs a separate credential migration.

The narrow configuration command changes no baseline, schedule, reward,
service-role, or validator setting:

```bash
python3 scripts/configure_lab_arena_production.py --miner-credentials-only \
  --miner-credential-kms-key-id 'arn:aws:kms:REGION:ACCOUNT:key/KEY_ID' \
  --allowed-account 493765492819 --check
```

For an authorized apply, use the same command with `--apply` and
`LEADPOET_LAB_ARENA_PRODUCTION_APPLY=1`. An empty key-ID argument disables new
admission for the staged deployment. It does not delete stored ciphertexts.
Do not give KMS access to submitted code. No new production dependency is
required.

Organizer keys remain for the explicitly identified daily baseline only.
With the miner vault unset, baseline operation remains available, but new
model admission fails closed.

For a hosted testnet check, the public gateway can route
`https://gateway.subnet71.com/testnet/arena/...` to a separate native Arena
service on loopback port 8793. It never falls back to the mainnet service.
Use `--testnet-proxy enabled` (or `disabled`) with the same configuration
tool, `--allowed-account`, and `--check`/authorized `--apply` workflow.
This changes only `LAB_ARENA_TESTNET_ENABLED`; it does not start a service or
change the baseline, schedule, rewards, or credentials. Load the setting with
the normal gateway restart. Miner and validator clients use
`https://gateway.subnet71.com/testnet` as their API base URL.

## Live validation and deployment boundary

The isolated testnet-401 run completed real CLI admission, source loading in
gVisor, provider calls, scoring, persistence, publication, and API restart
recovery. See [the live validation report](miner-model-live-validation-20260905.md).
This proves the workflow, not model quality: the test miner scored zero.

Production was not changed. The test used the real gateway and validator hosts
but separate processes, PostgreSQL, and an S3 prefix. It exercised the SQL RPCs
through `PsycopgTransport`, not the production PostgREST transport. Complete the
staged operator setup and production transport checks before enabling intake.

The scorer's webpage, job, LinkedIn post, and X post requests use Deepline
for miner-funded scoring only. Public webpages use Firecrawl; LinkedIn jobs
and posts use HarvestAPI; X posts use TwitterAPI; the three supported public
job-board APIs use bounded public HTTP reads. The adapters retain company and
post identity, dates, and closed-job status. The daily baseline keeps its
existing provider routes.

Miner model code must use OpenRouter and approved Deepline routes. Direct
ScrapingDog requests from miner code are refused; there is no organizer-key
fallback. In a fork of the public baseline, change its Arena transport's
search and page-fetch tools to Deepline before submission. Do not change the
public baseline just to run a miner test.
