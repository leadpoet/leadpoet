# Final legacy table retirement

The operator authorized removing the following twelve public tables together.
They belong to the retired lead, baseline-authority, and public Research Lab
loop mechanisms. Existing old jobs are part of the removal scope.

- `transparency_log`
- `validation_evidence_private`
- `company_information_table`
- `early_access_emails`
- `outreach_email_verifications`
- `suppression_ledger`
- `research_lab_public_loop_cards`
- `research_lab_public_loop_card_events`
- `research_lab_official_baseline_action_attempts_v1`
- `research_lab_official_baseline_action_terminals_v1`
- `research_lab_official_baseline_runs_v1`
- `research_lab_official_baseline_unit_closures_v1`

Migration 233 removes these tables and their reviewed database dependencies in
one transaction. It uses restrictive drops, exact routine-body guards, and a
short lock timeout. It also removes the obsolete dashboard refresh job and its
functions. Existing cached dashboard rows remain readable; these represent the
retired lead mechanism, not current Arena results. Current Arena competition
data comes from its own tables and API.

## SOURCE_ADD preservation

The operator's separate SOURCE_ADD retention instruction remains in force.
There is no standalone `source_add` table in the current production catalog.
Keep the eleven V2 historical receipt/ancestry/transport tables, the historical
weight bundles, and `merkle_checkpoints`.

A complete read-only scan of 41,970,974 transparency rows in 84 bounded primary
key ranges found 152 SOURCE_ADD marker rows. All are in the retained recent
history. Source inspection identifies their `RESEARCH_LAB_EPOCH_AUDIT` payloads
and weight-event references as direct SOURCE_ADD history.

Archive all original columns of the complete log suffix starting at id
`41960931`, not only the marker rows. At the reviewed snapshot it has 21,660
rows, including all 324 Research Lab epoch audits and their weight companions.
The floor includes every available earlier signed-event parent. Its JSONB
representation is approximately 616 MB before compression, substantially
smaller than the 48 GB public table and indexes.

The archive is non-public, owner-only, and protected from row mutation.
Migration 233 must compare complete source and archive row fingerprints under
the table lock before dropping the public log. The operator must check the
reviewed historical prefix fingerprint immediately before applying the exact
committed migration. The archive also retains original signed envelopes and
Arweave transaction pointers.

Some historical proof gaps already exist: four nonzero previous-event hashes
have no row in the source database, and not every old Arweave pointer resolves.
Preserve the original rows and links without fabricating proof material.
Existing generic Merkle checkpoints end below id 1,453,697 and do not cover
the SOURCE_ADD-era log suffix. Removing the old log does not create these gaps.

External checks recovered 39 associated Arweave checkpoint bodies. Their
recomputed Merkle roots and stored receipt/header roots agree. They contain
108 of the 152 SOURCE_ADD audit hashes, and every associated event is inside
the archive boundary. The other 44 signed audits refer to unavailable external
checkpoint content. Retain those signed rows and original pointers too. This
does not establish checkpoint signature validity or repair the older gaps.

The validation-evidence scan covered all 7,421,301 rows in 31 indexed epoch
ranges. Both timestamp columns end on May 14, 2026; no rows reach the June 17
SOURCE_ADD introduction. Source inspection also finds no SOURCE_ADD writer to
that table. It does not need to be retained for the SOURCE_ADD exception.

## Runtime and deployment order

Deploy the source change before applying migration 233. Gateway startup must
not read or write the retired relational log, and it must not start the old
relational checkpoint or anchor loops. Coordinator event signing and Arweave
buffering remain available. Each new enclave boot starts its own signed buffer
chain; current Arena receipts, provider execution, rewards, and normal-validator
weights use separate authorities and do not depend on the retired log tip.

Preserve current ICP storage and generation, testnet behavior, enclave identity,
Arena routes, provider and scoring rules, rewards, weight submission, and the
website's separate Supabase project. Dormant unregistered lead modules and the
historical measured SOURCE_ADD query policies do not become active callers.

The release gate must exercise the logger against the real coordinator signing
contract, prove zero database calls during signing/startup, rehearse the exact
migration with populated PostgreSQL fixtures and repeated application, and
verify the committed protected-workflow manifest. After the canonical gateway
deployment and SQL commit, compare retained schemas and SOURCE_ADD fingerprints,
published rounds/results/ICPs, public dashboard and Arena health, and observed
normal-validator behavior. A healthy endpoint alone is not evidence of a new
completed scoring run or finalized chain transaction.
