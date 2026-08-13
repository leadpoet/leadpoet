# Physical V2 staging

The local restart rehearsal validates deterministic application contracts with
strict adapters. It cannot validate Nitro AF_VSOCK behavior, host networking,
kernel/runtime interactions, real provider TLS, or an actual chain
finalization. Physical V2 staging covers those boundaries before a release is
used in production.

## Acceptance contract

The `Physical V2 Staging Acceptance` workflow starts automatically after the
exact `Attested V2 Release` workflow succeeds for `main`. It:

1. invokes `scripts/restart_attested_release_local.sh` against dedicated
   staging gateway and primary-validator hosts;
2. uses separate staging Secrets Manager documents and an isolated,
   production-shaped Supabase project;
3. starts the same gateway and validator launchers, Nitro enclaves, Caddy,
   credential envelopes, provider transport, model artifacts, PCR0 checks,
   and exact-commit checks as production;
4. starts at least two independent audit validators from the exact candidate;
5. waits for the real candidate to generate and finalize one canonical bundle
   on Bittensor testnet; and
6. requires every auditor's finalized `submission_success` event to carry the
   same epoch, bundle hash, and weights hash as the gateway authority.

The controller does not reproduce scoring, ICP, settlement, allocation, or
weight logic. Those behaviors come from the candidate runtime and its
configuration, so changing their implementation does not require rewriting
the acceptance test. The stable boundary is behavioral: one exact release,
one finalized canonical authority, and identical primary/auditor submissions.

## One-time infrastructure

Create a GitHub environment named `physical-v2-staging` with:

- a dedicated controller carrying the
  `leadpoet-v2-physical-staging` self-hosted-runner label;
- distinct Nitro-capable gateway and primary-validator hosts;
- at least two distinct audit-validator hosts and registered testnet hotkeys;
- staging-only gateway and validator Secrets Manager documents;
- an isolated Supabase project with the production migration set; and
- the same KMS, ECR, S3, Caddy, provider, and enclave permissions as
  production, scoped to staging resources.

Do not use either production host as a staging runner or target.

Set `LEADPOET_V2_STAGING_SSH_KEY` as an environment secret. Set
`LEADPOET_V2_STAGING_CONFIG_JSON` as an environment variable using this shape:

```json
{
  "schema_version": "leadpoet.physical_v2_staging_config.v1",
  "environment": "physical-v2-staging",
  "network": "test",
  "netuid": 1,
  "gateway_public_url": "https://staging-gateway.example",
  "timeout_seconds": 7200,
  "poll_seconds": 10,
  "gateway": {
    "ssh_host": "ec2-user@staging-gateway.internal",
    "restart_path": "/home/ec2-user/gw_restart.sh",
    "secret_id": "leadpoet/staging/gateway/env"
  },
  "primary_validator": {
    "ssh_host": "ec2-user@staging-validator.internal",
    "restart_path": "/home/ec2-user/validator_restart.sh",
    "secret_id": "leadpoet/staging/validator/env",
    "repo_root": "/home/ec2-user/leadpoet/leadpoet",
    "container_name": "leadpoet-validator-main"
  },
  "audit_validators": [
    {
      "ssh_host": "ec2-user@staging-auditor-a.internal",
      "repo_root": "/home/ec2-user/leadpoet/leadpoet",
      "unit_name": "leadpoet-auditor-a.service",
      "expected_hotkey": "TESTNET_AUDITOR_A_HOTKEY"
    },
    {
      "ssh_host": "ec2-user@staging-auditor-b.internal",
      "repo_root": "/home/ec2-user/leadpoet/leadpoet",
      "unit_name": "leadpoet-auditor-b.service",
      "expected_hotkey": "TESTNET_AUDITOR_B_HOTKEY"
    }
  ]
}
```

The workflow injects the ephemeral SSH-key path. Configuration validation
rejects production IPs, production secret names, shared hosts, non-testnet
networks, malformed service names, and fewer than two auditors.

## Release use

Treat the workflow result for the exact SHA as a required deployment check.
Do not restart production on a candidate whose physical acceptance is absent,
failed, cancelled, or superseded. Keep the bounded local `prepush` profile as
the fast pre-push gate; physical staging is the post-attestation proof of the
external runtime boundaries that a local adapter cannot reproduce.

Mainnet and testnet do not share chain state. Production still requires
post-restart exact-SHA, PCR0, canonical-bundle, finalization, and readback
monitoring. The staging lane removes application/runtime discovery from that
step; it does not fabricate mainnet state or weaken fail-closed checks.
