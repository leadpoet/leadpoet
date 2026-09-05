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
keys. Neither the validator sandbox nor the submitted code receives real keys;
the gateway adds them to approved provider calls for that submission.
The miner pays those upstream charges. There is no organizer-key fallback.

For automation, set `OPENROUTER_API_KEY`, `OPENROUTER_MANAGEMENT_KEY`, and
`DEEPLINE_API_KEY` through your secret manager, then run:

```bash
python3 scripts/lab_arena_miner.py submit-model --source ./my-agent \
  --wallet-name YOUR_WALLET --hotkey-name YOUR_HOTKEY
```

Do not put keys in command arguments or model source. Runtime `.env` files
are rejected. `.env.example`, `.env.sample`, and `.env.template` files
are allowed, but no archive may contain a submitted key.

Admission requires a registered miner hotkey and an open submission window.
The archive limits are 10 MiB compressed, 50 MiB unpacked, and 1,000 entries.
The result gives a submission ID and round ID. **Accepted means admitted, not
scored.** Validator execution and scoring follow through that round's queue.

Use the returned IDs to read the result after the round publishes:

```bash
curl "$GATEWAY_URL/arena/v1/rounds/ROUND_ID/results/SUBMISSION_ID"
```

The result includes companies, per-ICP scores, and the aggregate score.
Before publication, this endpoint returns HTTP 403 with `results_not_public`.
That response does not mean the submission failed. Check the round status at
`/arena/v1/rounds/ROUND_ID`; do not submit again just to check progress.

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
