# AWS migration: isolated restore rehearsal

This stage copies a consistent, read-only source snapshot into the dedicated
`leadpoet-subnet-arena` PostgreSQL 17.9 target. Production still uses Supabase.
Do not change production endpoints or restart gateway/validator services.
The coordinator owns all AWS and source permission writes.

## October 28 rehearsal results

The isolated RDS restore committed successfully. Verification compared the
restored database with the frozen source snapshot, including all 48 tables and
11,627,285 rows. Every table's row count and full-row content digests matched.
Stored generated values, protected history, all three archived sequence states,
application owners, recorded grants and all 12 catalog comparison categories
passed. Temporary target CREATE grants and the target export identity were
removed before the final verification.

The final proof is `restore-verification4-20261028/verification-completed.json`;
its SHA-256 is
`919b24a7fc0c7f626c206f9e689419b1345c106a487f7a7870e935f2346434f9`.
The captured source manifest SHA-256 is
`d93ddef1740ecfa1bb74a186f36c918ca731e4973cfd756fa69f9a62d8138f56`.
Private receipts and the native dump stay outside Git. The comparison includes
the exact reviewed RDS platform differences documented in the restore plan;
it does not silently omit application objects to obtain a match.

All 12 read-only database capability checks also passed. They cover the restored
auth claim getter, installed extension capabilities and catalog permissions.
They do not test an Auth server, an external dblink connection, application
requests, provider calls, scoring, publication or chain writes.

The temporary source exporter is disabled, has no active sessions and has lost
its recorded data access. The existing parity reader's recorded role metadata
and 51 direct relation grants match the before-state. The runner's temporary
secret-read policy is removed, and both credential reads are denied. Only the
source-export secret is scheduled for seven-day deletion; the RDS-managed
target secret remains active.

The encrypted backup snapshot `snap-0ab1ae007cff3f842` was independently restored
to a separate encrypted volume. A no-modify filesystem check passed, followed
by read-only checks of all 49 archive files and 36 final proof/code/access files.
Every checksum matched. Normal unmount and a separate all-process mount check
passed. The external backup receipt SHA-256 is
`4b70cf70c07a26ce52444f239a710cc23449b4ec46aa56cbfc1bceea7cf9f681`.
The snapshot is private and remains retained. The disposable verification
volume is deleted. The temporary runner is terminated and its stack is
`DELETE_COMPLETE`; the original runner disk, network interface, public IP,
security group, role, instance profile and exact temporary database ingress
are gone. Other target security-group rules are unchanged.

Final AWS reads confirm the database remains available, private, encrypted and
deletion-protected, with 14-day automated backup retention. The foundation,
target managed secret, verified snapshot and encryption key remain present.
These retained resources continue to incur their normal AWS charges. No source
application rows or production endpoint settings were changed, and no gateway
or validator restart was run as part of this rehearsal.

## Work required before production cutover

This is a verified point-in-time copy, not a live replica. The source has
continued to accept writes and schema changes, including migration 584 after
the snapshot. A final synchronization must include all subsequent changes.

The current Arena store uses PostgREST table and RPC HTTP endpoints, and the
gateway uses a Supabase HTTP client. An RDS SQL endpoint cannot replace those
HTTP URLs directly. The next stage must provide and verify the compatible API
and authentication path, then test the current gateway, validator and dashboard
clients against it. Preserve existing schemas, accounting, scoring and recovery
semantics; avoid a client rewrite unless compatibility testing requires it.

The source and target also use different libc collation versions. Exact copied
bytes and catalog parity do not prove identical text ordering, equality or
index behavior. Include these semantics in the pre-cutover compatibility tests;
this rehearsal does not change or rebuild source indexes.

After those tests, prepare the final data synchronization and controlled
endpoint switch with a rollback path. Verify real submissions, rebenchmarks,
provider accounting, scoring, persistence, publication, promotion and weights
through the canonical release process. Those checks and production restarts
are not part of this isolated restore stage.

The foundation PR's repository-wide check currently times out and its dependent
Docker check is skipped. The stacked restore PR remains unmerged. Focused
migration tests and live database proofs do not replace that merge gate; no
branch-protection checks have been bypassed.

## Temporary runner

Deploy `deploy/aws/subnet-restore-runner.yml` as
`leadpoet-subnet-restore-runner-20261028` in account `493765492819`, `us-east-1`.
It uses existing VPC `vpc-0c975a643bc1e0e79` and public subnet
`subnet-025170c1eff61494d` in `us-east-1d`. Verify the subnet's internet-gateway
route and available addresses before deployment. No route or subnet changes
are part of this template.

The runner is `t4g.medium`, with the standard AL2023 arm64 AMI from the AWS
public SSM parameter. Record the resolved AMI ID and installed package versions
in the sanitized deployment receipt. Its only disk is encrypted 200 GiB gp3.
There is no SSH key, public ingress, NAT, load balancer, application API or S3
bucket. A temporary public IPv4 address permits outbound HTTPS for Systems
Manager, official package repositories and certificate downloads. Access uses
Systems Manager only. IMDSv2 is required, with response hop limit 1.

The runner security group allows only outbound TCP 443 to IPv4 internet,
outbound TCP 5432 to the supplied source pooler IPv4 `/32` addresses, and
outbound TCP 5432 to target security group `sg-0081d026fe5311783`. DNS to the
VPC resolver is an AWS security-group exception. A separate ingress resource
allows target TCP 5432 only from the runner group. The original database
foundation template stays unchanged. Never allow source TCP 5432 to
`0.0.0.0/0`. If source DNS changes, stop and review exact replacement addresses.

The instance role has `AmazonSSMManagedInstanceCore` plus `GetSecretValue` on
exactly two parameterized Secrets Manager ARNs: the dedicated source-export
credential and the RDS-managed target credential. It has no S3, RDS control,
secret enumeration or secret mutation permission. These secrets must use the
AWS-managed Secrets Manager key; a customer-managed key requires a separate
review of narrow decrypt access. Never pass secret values as template parameters.

Bootstrap installs the official AL2023 `postgresql17` and `python3` packages,
checks `pg_dump`, `pg_restore` and `psql` major version 17, and checks AWS CLI
v2. It enables the existing SSM agent and creates
`/var/lib/leadpoet-migration/bootstrap.ready` only after those checks pass.
Bootstrap does not fetch credentials or execute SQL, export or restore.
It downloads the official regional RDS CA bundle to
`/etc/leadpoet-migration/rds-us-east-1-bundle.pem`. The AL2023 system trust bundle
is `/etc/pki/tls/certs/ca-bundle.crt`. The runtime job must separately verify the
source's actual certificate chain; do not assume its issuer or lower TLS
verification after a failure.
Do not install external PostgreSQL RPM repositories or PyPI packages as a
bootstrap workaround. Inspect a failure before changing the reviewed image.

## Deployment and access checks

1. Confirm the selected caller is account `493765492819` in `us-east-1`.
   Read current CloudFormation, EC2, IAM, subnet, route and target metadata.
   Verify both secret ARNs without printing their values. Resolve the exact
   source session-pooler hostname and confirm its current IPv4 addresses.
   The October 28 observation for `aws-0-us-east-1.pooler.supabase.com` was
   `44.208.221.186/32`, `44.216.29.125/32` and `52.45.94.125/32`. Supply the three
   freshly verified addresses as the required `SourcePoolerIpv4CidrA/B/C`
   parameters; the template deliberately has no source-address defaults.
2. Validate the exact reviewed template. Create a named `CREATE` change set
   with `CAPABILITY_IAM`. Check that all declared resources are additions.
   The only existing-resource access change is the separately owned target
   ingress rule. Reject any replacement or unrelated modification.
3. Execute the reviewed change set and enable stack termination protection.
   Record commit, template SHA-256, change-set ID, stack ID, instance ID,
   resolved AMI ID, volume ID, role ARN and both security-group IDs.
   Do not record credentials or secret contents.
4. Wait for stack completion, EC2 status checks and SSM `Online`. Then use an
   explicit SSM command to check bootstrap completion, client versions, disk
   size/free space, volume encryption, IMDS settings, role policy and effective
   ingress/egress. Stack completion alone does not prove bootstrap succeeded.
5. Verify a real TLS connection to both database endpoints with hostname and
   certificate validation. Use the source session pooler in session mode;
   transaction pooling is unsuitable for a consistent native dump. Confirm the
   target endpoint is the dedicated rehearsal database before any target write.

## Export, restore and proof

The separate reviewed export/restore job must retrieve secrets at runtime,
without shell tracing. Keep credentials out of process arguments, SSM command
content, stdout, stderr, CloudWatch output and logs. Write escaped libpq password
entries to separate mode-0600 temporary `PGPASSFILE` files in a mode-0700
directory. Remove those files when the job exits. Use verified CA files and
`sslmode=verify-full`. Do not use `PGPASSWORD` in a diagnostic environment dump.

Use the dedicated, temporary, read-only export identity. Do not change the
existing parity reader. Its reviewed grants must cover the full intended
schema, all 48 application/history tables, sequences and protected `SOURCE_ADD`
history. Verify effective access, row-level-security behavior and consistent
snapshot coverage before declaring the export complete. Never put source SQL
that modifies application records in this stage.

Keep the native dump, its SHA-256, schema/role/extension inventory and sanitized
verification reports on the encrypted runner disk under
`/var/lib/leadpoet-migration`. Keep the artifact directory mode 0700 and files
mode 0600. Check free space before export and restore. Do not place credentials
in dump comments, role exports, SQL receipts or filenames.

The coordinator must independently review source grants and the exact target
restore plan before running either. Restore only into `leadpoet-subnet-arena`.
Use explicit, reviewed mappings for hosted-only roles, extensions, ownership,
grants, auth/storage objects and platform-managed objects. A dump success code
does not prove that these objects or their behavior work on RDS.

Verify every application/history table and `SOURCE_ADD` history against the
same captured source snapshot, not a later moving source count. Compare table
and sequence inventories, row counts, deterministic content checks, constraints,
indexes, required functions, extensions and effective role permissions. Save
redacted failures as well as successes. Confirm source endpoints, source data,
running jobs, accounting, scoring and chain state remain unchanged. Record
client/API compatibility and production cutover as remaining stages.

## Retention and cleanup

The instance has CloudFormation `Retain` and `UpdateReplacePolicy: Retain`,
plus EC2 termination protection. These protect the local dump during stack
deletion or replacement; they do not make it a backup. Do not delete the stack
while the runner is still needed: its role, profile and network rules are not
retained, and stack deletion can strand the retained instance. Keep the stack
and instance IDs in the handoff and track their ongoing charges.

The root disk has `DeleteOnTermination: true`. **Do not terminate the instance
until the dump and verification evidence have a verified encrypted backup
outside that instance.** A checksum on the same disk is insufficient. The
coordinator chooses and verifies the backup destination separately; this
template grants no backup write permission.

After restored-state verification, remove the temporary source-export access
and its secret access. Confirm the original parity-reader permissions stay
unchanged. After the independent backup is verified and continued runner use
is no longer needed, perform the reviewed exact cleanup: remove the temporary
target ingress, disable instance termination protection, terminate the exact
runner, remove its stack resources and confirm that no runner EBS volume,
public IP, role/profile or unexpected ingress remains. Preserve the target,
its managed secret, snapshots and migration evidence. Report any retained
resource that still incurs charges.

References:
- [Official AL2023 PostgreSQL package installation](https://repost.aws/knowledge-center/ec2-linux-install-package-unavailable)
- [Standard AL2023 AMI package inventory](https://docs.aws.amazon.com/linux/al2023/ug/amzn2-al2023-ami.html)
- [EC2 CloudFormation properties](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ec2-instance.html)
- [CloudFormation retention behavior](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-attribute-deletionpolicy.html)
