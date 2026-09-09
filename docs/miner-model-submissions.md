# Submit a competing model

The miner menu has two actions: **Submit SOURCE_ADD** and **Submit Model**.
SOURCE_ADD status is available in the SOURCE_ADD submenu.

## Source and credentials

Fork the public [PydanticAI baseline](https://github.com/leadpoet/pydantic-harness),
or provide another Python agent with this entrypoint in `harness.py`:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return up to five companies with ICP-fit and intent evidence."""
```

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
that owns an unfinished submission; published and cancelled results are public.
The archive limits are 10 MiB compressed, 50 MiB unpacked, and 1,000 entries.
The result gives a submission ID and round ID. **Accepted means admitted, not
scored.** Validator execution and scoring follow through that round's queue.

Retry an unchanged archive with the same hotkey. The gateway reuses its upload
reservation. If an unfinished archive changes, the gateway keeps the old row
and bytes and assigns a new upload target. A late finalize for the replaced
reservation returns `submission_superseded`. An accepted submission cannot be
replaced during that round. The existing MD5 transport checksum prevents a
same-size changed archive from silently finalizing older bytes.

Use the returned IDs to read the result after the round publishes:

```bash
curl "$GATEWAY_URL/arena/v1/rounds/ROUND_ID/results/SUBMISSION_ID"
```

The result includes companies, per-ICP scores, and the aggregate score. While a
round is nonterminal, this endpoint returns HTTP 403 with
`results_not_public`. That response does not mean the submission failed. Check
the round status at `/arena/v1/rounds/ROUND_ID`; do not submit again just to
check progress.

If a round is cancelled after work has completed, the same result endpoint
returns the completed data for participants frozen into that round. The response
sets `round_status` to `cancelled`, includes `cancel_reason`, and sets
`incomplete` to `true`. `judge_jobs` reports only terminal status and a safe
cause. Its `evidence_status` says whether redacted evidence is `available`,
`unavailable`, or `invalid`; it never includes an object-store or validation
error. `execution_jobs` uses the same safe approach and labels each output as
`available`, `unavailable`, or `invalid`. `judge_evidence` contains validated,
redacted evidence for accepted judge jobs. Missing outputs and scores remain
missing, and aggregate scores, ranking, king decisions, and rewards are not
created for a cancelled round. Results stay private for every nonterminal round,
and another round's submission ID does not grant access.

Provider calls made while a round is running can incur the miner's upstream
charges even if a later infrastructure failure cancels the round. A cancelled
round does not publish a ranking and does not automatically refund provider
charges. When a required judge assignment exhausts its infrastructure retries,
the driver cancels the remaining work; already dispatched provider calls can
still incur charges. Arena reward activation is a separate setting: disabling
rewards prevents champion allocation, while scoring and competition execution
remain separate configuration behavior.

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
