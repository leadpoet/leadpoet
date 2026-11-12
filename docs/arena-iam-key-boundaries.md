# Arena key boundaries

The gateway signs Arena authority documents. The normal validator verifies
those signatures and signs chain transactions with its own hotkey. The validator
must not sign documents with the gateway authority key. Its hotkey-envelope
decrypt permission, public-key reads, and existing miner-credential decrypt deny
must remain unchanged.

Routine operators must not disable or schedule deletion of the five production
keys that their broader operator policy permits them to use. Intentional key
retirement uses the separately authorized Admin maintenance role. The deny does
not affect key use, rotation, reads, or cancellation of pending deletion.

## Deployment

Use the exact reviewed JSON policies in `deploy/aws/` and verified temporary
authority for account `493765492819`, region `us-east-1`. Preserve existing
policies and record private before/after receipts. Do not print credentials.

* Merge `arena-routine-operator-kms-deny.json` into the **inline policy of the
  LabOperator Identity Center permission set**,
  `arn:aws:sso:::permissionSet/ssoins-722381b1112ac249/ps-72230b52f6eacc70`.
  Preserve its existing customer-managed policy references. Re-provision this
  permission set to account `493765492819` and require `SUCCEEDED`. Do not edit
  only the generated `AWSReservedSSO` role; a later provisioning could erase it.
* Install `arena-validator-authority-deny.json` as the dedicated inline policy
  `leadpoet-arena-authority-signing-deny` on
  `leadpoet-validator-s3-cloudwatch-role`. Do not change the gateway role or its
  key policy. Verify the role is attached only to the intended validator host.

Before either write, re-read the exact source policy and refuse any drift or
conflicting statement with the same Sid. Afterward, verify exact policy readback.
An uncertain write response requires a readback before another write. Do not
automatically overwrite a concurrent policy or broaden access during rollback.

## Acceptance checks

Use AWS policy validation and IAM simulation before and after deployment:

1. Routine operator `DisableKey` and `ScheduleKeyDeletion` must be explicitly
   denied on each of the five keys. Existing key-use and read decisions must
   remain unchanged. Admin maintenance access must remain unchanged.
2. The validator's authority-key `Sign` must be explicitly denied. Its
   `GetPublicKey`, hotkey-envelope `Decrypt`, artifact access, and existing
   miner-credential deny must remain unchanged. Gateway signing must still work.
3. Confirm the signing-key public-key hash against the validator's pinned key.
   Test `Sign` with `DryRun=True` under actual host roles; never publish a test
   signature or submit a test weight transaction.
4. Check fresh stored validator/gateway progress, existing signed public results,
   source identity, and final weight status. No service restart, historical
   rescore, model change, or round-policy change is required.

When replacing a production key, update the scoped resources before activation.
These guards intentionally use exact keys instead of denying unrelated account
resources. They supplement existing policy boundaries, not an account-wide IAM
redesign.
