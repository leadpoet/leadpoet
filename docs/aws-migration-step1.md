# AWS migration: isolated database foundation

This is step 1 of the authorized, staged Arena migration. It creates an empty
destination, not a second active Arena. The source remains hosted Supabase.

## Resources

Deploy `deploy/aws/subnet-database-foundation.yml` as
`leadpoet-subnet-database-foundation` in account `493765492819`, `us-east-1`.

- Existing lab VPC: `vpc-0c975a643bc1e0e79`.
- New subnets: `172.31.96.0/24` in `us-east-1d` and `172.31.97.0/24` in
  `us-east-1c`. The October 28 preflight found no overlap with existing subnets.
- A new route table has only the VPC-local route. Both subnet associations must
  complete before RDS creation. No existing route table or subnet changes.
- A new security group has no ingress. An explicit loopback egress rule removes
  the automatic allow-all egress. No application or operator can connect yet.
- `leadpoet-subnet-arena`: RDS PostgreSQL 17.9, `db.m7g.large` (2 vCPU, 8 GiB),
  150 GiB gp3, storage autoscaling maximum 500 GiB. This is an initial restore
  rehearsal size, not a production capacity commitment.
- Single-AZ for the rehearsal. Review production availability and measured load
  before cutover. The template supports enabling Multi-AZ later.
- Encryption at rest, forced TLS, no public address, 14-day automated backups,
  deletion protection, retained data on stack removal/replacement, and an
  RDS-managed master secret. No plaintext credential is in the template.
- Minor updates are explicit for the rehearsal. Establish the normal patch
  schedule before production cutover; do not leave it without an owner.

The AWS Pricing API on October 28 returned $0.168/hour for the instance and
$0.115/GiB-month for Single-AZ PostgreSQL gp3. At 730 hours and 150 GiB, that is
$139.89/month plus the managed secret and any additional backup/transfer use,
approximately $140/month before credits. No reservation is purchased. Credit
eligibility and remaining balance are not verified by this step. Storage growth
can increase this amount. The API/restore runner will be scoped separately.

## Deployment procedure

1. Confirm the selected profile's STS account is `493765492819`. Read current
   subnet ranges, DB identifiers and CloudFormation stacks again. Reject a
   duplicate destination, overlapping CIDR, or unexpected existing stack.
2. Validate the template using CloudFormation `validate-template`.
3. Create a named `CREATE` change set from the exact reviewed template. Confirm
   that every change is an `Add` for a resource declared by this template; there
   must be no IAM, existing-resource modification, deletion, import or replacement.
4. Execute that change set and enable stack termination protection. Do not use a
   password argument or fetch the new master secret as part of creation.
5. Wait for `CREATE_COMPLETE` and RDS `available`. Verify the effective resource
   configuration below. Keep a sanitized receipt with template hash, commit,
   change-set ID, stack ID and resource IDs.

Use the existing authorized profiles. `leadpoet-lab` is the normal SSM/EC2 role;
it does not grant general RDS/CloudFormation administration. The existing
`leadpoet-recovery-console-20261026` session is available for this scoped setup.
No IAM policy or role needs to be changed for step 1.

## Acceptance checks

- Stack complete; database available at the selected version, class and storage.
- Database and secret encrypted; managed secret status active (metadata only).
- Public accessibility false; only the dedicated DB security group attached.
- Security group has zero ingress; its only egress is loopback.
- Both new subnets have public-IP assignment off and use the new local-only
  route table. Original subnet/route-table snapshots remain unchanged.
- `rds.force_ssl=1` in the attached parameter group, applied/in-sync. This checks
  configuration, not an actual rejected plaintext SQL connection.
- Backup retention 14 days, deletion protection and stack termination protection
  true. Backup configuration is not proof of a completed restore test.
- No source SQL, application endpoint update, deployment, restart, paid provider
  call or chain action was performed.

No database login is attempted with zero ingress. Creating a fresh RDS instance
without a snapshot or import proves no application data was restored through
this step; SQL contents and HTTP behavior are not verified yet.

## Next step and boundaries

Step 2 will establish narrowly scoped restore-runner access, complete source
export permissions and a consistent native export/restore. It will test role and
extension mapping, including all 48 application/history tables and protected
SOURCE_ADD history. It must not broaden the existing parity reader casually.
Do not connect a production writer until the separate client/API, recovery and
cutover stages pass.

Do not delete or recreate this target to change its size after copying data.
Review change-set replacement indicators. Retain policies and deletion
protection are intentional: teardown needs an explicit reviewed procedure.

References:
- [RDS CloudFormation properties](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-rds-dbinstance.html)
- [RDS-managed secrets](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/rds-secrets-manager.html)
- [Removing default security-group egress](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ec2-securitygroup.html#aws-resource-ec2-securitygroup--examples--Remove_the_default_rule)
- [PostgreSQL RDS pricing](https://aws.amazon.com/rds/postgresql/pricing/)
