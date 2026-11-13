# Arena disaster recovery

This runbook covers recovery verification on **new, isolated targets**. It does
not authorize a production restore, endpoint switch, service restart, wallet
activation or chain write. The database named `leadpoet-subnet-arena` is
production. The [old migration restore plan](aws-migration-restore-plan.md) and
its fixed-target utilities are historical records, not a rehearsal procedure.

## Verified state and remaining gaps

Evidence was captured on November 12–13, 2026, in account `493765492819`,
region `us-east-1`. Recheck live state before recovery. Private receipts are in
`/Users/pranav/Downloads/Arena_Backup_Access_Audit_20261112/`; do not commit their
contents or source object bodies.

| Area | Verified result | Limit |
|---|---|---|
| Production RDS | Native 14-day PITR is active. Database is private, encrypted and deletion-protected. Automated snapshots exist; no manual snapshots were found. | No current RDS PITR restore rehearsal has been proved. Deletion protection does not prevent bad SQL. |
| Primary corpus bucket, `leadpoet-attested-v2-artifacts-493765492819` | Versioning, default AES256 encryption, all four public access blocks and default 365-day COMPLIANCE Object Lock are enabled. Sampled request, internal-capture and Arena objects have individual COMPLIANCE retention. | Samples do not prove every historical object is locked. No replication configuration or lifecycle configuration was found. |
| Isolated S3 recovery test | Three actual production object versions were copied to a private, encrypted, versioned test bucket and matched by hash. Overwrite/delete-marker recovery passed. Locked-version deletion and shortened retention were denied. Test resources were cleaned. | Zero production writes. This was a bounded version-recovery proof, not a complete corpus or application restore. |
| Host storage | One completed encrypted gateway EBS snapshot was found, dated October 28. No snapshots were found for the live validator or dashboard volumes. | Current local journals, configuration and other persistent files are not proved recoverable from a recent host backup. |
| Supplemental AWS Backup | [Draft template](../deploy/aws/arena-recovery-backups.yml) defines daily periodic RDS snapshots at 11:00 UTC retained 35 days and the three exact EBS volumes at 12:00 UTC retained 14 days. | **NOT DEPLOYED.** Native RDS PITR remains separate. Plan creation is not evidence of a successful backup or restore. |

Relevant receipts: `inventory-lab.private.json`,
`recovery-inventory.private.json`, `buckets-lab.private.json`,
`object-retention-samples.private.json`, `host-storage.private.json` and
`s3-recovery-proof.private.json`. The last receipt records successful cleanup
at `2026-11-13T07:59:01Z`.

## Before a recovery test

### Routine operator protection

The proposed `deploy/aws/arena-routine-operator-corpus-deny.json` is an additive
deny overlay for the existing LabOperator Identity Center inline policy. It
blocks object/version deletion and bucket retention/versioning changes on the
two exact corpus buckets. It grants no permissions and preserves capture writes,
version reads, retention extensions and isolated test-bucket cleanup. **It is not
deployed yet.** Merge it by Sid with the existing KMS deny; do not replace other
statements or edit only the generated SSO role. Re-provision the existing
permission set and verify its status and exact readback, then test denied
actions on synthetic objects and successful normal capture/version reads.
Admin maintenance stays separate. This does not replace Object Lock or protect
the currently unlocked private-model bucket from a compromised administrator.

### Isolated test gates

1. Verify the AWS principal, account and region through the canonical access
   workflow. Inventory existing plans, vaults and recovery points before applying
   the draft template. Use exact resource identifiers; do not select broad tags.
2. Choose the intended recovery time and record the exact RDS recovery point or
   PITR timestamp, available restore window, source identity and encryption key.
   Record each required S3 key, version ID, checksum and retention date. Keep
   manifests and application data private. A current object listing alone cannot
   identify the versions required by an older database.
3. Record read-only production state: active endpoints, publication identifiers
   and hashes, current accepted/signed weight state, and relevant execution and
   accounting state. Use these records to check that the test changes none of
   them. Do not invoke provider calls or chain submission to collect evidence.
4. Select distinct target names and a private network boundary. The restored
   database must have no ingress from gateway, validator or dashboard clients.
   Give only the verification runner access. Provide no route from restored
   workloads to production services, providers or chain endpoints.
5. Review the exact restore request and target configuration. Do not use commands
   that contain the production database as a destination. Check restored database
   scheduling, subscriptions and other automatic work before allowing access;
   do not enable copied jobs. Do not start restored host images or attach their
   disks to production hosts. Mount inspection copies read-only where practical.

## Restore and verify on isolated targets

1. Restore RDS to a **new** identifier and endpoint. Restore selected corpus
   versions into a distinct private target. Preserve the production database,
   buckets and object versions. The verifier must use the new target identities
   and hostname-verified TLS; fail if any destination is a production endpoint.
2. Compare the restored database with a reference captured for that recovery
   point. Check schema and table inventories, owners, grants/RLS, functions,
   constraints, indexes, sequences and required extensions. Compare table row
   counts and deterministic full-row hashes where a matching reference exists.
   For sampled checks, record the exact scope and hashes. Later moving production
   counts are not an exact reference for an older restore.
3. Resolve restored database references to the matching source, input, output,
   score, request, response and journal artifacts. Read exact S3 version IDs and
   verify expected hashes. Include the configured application prefix and
   `private/arena-training/`; do not treat only the training folder as the corpus.
   Reject missing or ambiguous versions. A hash proves matching bytes, not a
   complete set of required artifacts.
4. Verify execution-journal identity, lease generation, claim hashes and ordered
   segment hashes through the existing readers. A recovered journal prefix is
   incomplete; do not promote it to a final trajectory. The unsent tail of a
   crashed worker cannot be recovered from S3. For host-state inspection, include
   the validator's configured state directory (normally
   `/var/lib/leadpoet/arena-validator`) and its signed/outcome journals. Preserve
   wallet files, but do not activate wallets, signing services, validators or
   other production daemons on a recovery target.
5. Repeat the read-only production checks. Confirm endpoint configuration,
   publications, accepted/signed weight history and recorded execution/accounting
   state were not changed by the test. Distinguish normal concurrent progress
   from test actions; investigate any unexplained difference. Do not publish a
   restored round or broadcast weights as a test.
6. Save a private receipt with source and target identities, recovery-point age,
   duration, checksums, passes, failures and exclusions. Remove only the exact
   disposable targets under the approved cleanup scope. Verify cleanup. Preserve
   production resources, source recovery points and their keys.

## Limits and completion criteria

Current protection and the draft template are in the same account and region.
They do not prove recovery after loss of that account, region or its access.
Encrypted RDS and EBS snapshots depend on their source KMS keys and usable key
permissions. A vault does not remove that dependency. The live validator and
dashboard volumes are unencrypted; their ordinary EBS snapshots remain
unencrypted. Do not claim that vault encryption changes this. EBS snapshots are
crash-consistent; they do not prove application-consistent recovery.

The draft vault uses a seven-day grace period before optional COMPLIANCE lock
becomes irreversible, with minimum seven-day and maximum 90-day retention.
Review this before deployment. Its planned finite 14/35-day expiration is not
permission to delete production data, existing snapshots or corpus objects.

A recovery test is complete only when the selected recovery point restores,
the declared database/artifact checks pass, production remains unaffected and
cleanup is verified. Record failures and missing references explicitly. The S3
proof does **not** establish current RDS/host restore success, complete corpus
coverage, an end-to-end application failover, or measured recovery-time and
recovery-point targets. Those remain unproved until their own checks pass.

## Cost basis

The November 13 inventory found 366 GiB provisioned across the three root
volumes and about 120 GiB in PostgreSQL. These are not measured snapshot sizes.
At the published US East standard EBS snapshot rate of $0.05/GB-month, a first
snapshot set has an upper planning estimate of about $18.30/month, plus retained
changed blocks. RDS backup storage above the regional free allocation is
$0.095/GB-month. For example, 100 GB of additional retained RDS blocks costs
$9.50/month. Each extra 100 GB of EBS changed blocks costs $5/month.

Use $20–40/month only as an initial planning range for the supplemental backups;
database and journal churn can exceed it. Daily incremental backups are not
35 or 14 full copies. Check actual billed storage after the first backup cycles.
The dedicated Secrets Manager secret adds about $0.40/month plus API requests.
Existing S3 Object Lock/versioning costs are not newly introduced by this plan.
Restore rehearsals also incur temporary database/volume/runner costs. No paid
restore or new recurring backup plan has yet been deployed by this audit.

References: [EBS snapshot pricing](https://aws.amazon.com/ebs/pricing/),
[RDS backup allowance](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html),
[current US East RDS price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonRDS/current/us-east-1/index.json),
[AWS Backup pricing](https://aws.amazon.com/backup/pricing/),
[Secrets Manager pricing](https://aws.amazon.com/secrets-manager/pricing/).
