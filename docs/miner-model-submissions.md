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

## Operator setup

Apply `scripts/185-lab-arena-miner-credentials.sql` after migration 184.
Set `LAB_ARENA_CREDENTIAL_KMS_KEY_ID` on the gateway and grant that service
KMS Encrypt/Decrypt access to the configured key. Do not give KMS access to
submitted code. No new production dependency is required.

Organizer keys remain for the explicitly identified daily baseline only.
With the miner vault unset, baseline operation remains available, but new
model admission fails closed.

## Remaining live validation

This change is not yet proof of a live miner submission. Validate admission,
source loading, actual provider calls, persisted scoring, and failure isolation
on the production gateway and validator before cutover.

The current scorer still requests ScrapingDog operations. Those operations
cannot use the three submitted keys and are refused for miner submissions.
A verified Deepline-backed route is still needed before claiming that the
three-key sourcing-and-scoring path works end to end. Do not enable an
organizer-funded fallback to hide this gap.
