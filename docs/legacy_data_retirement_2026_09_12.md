# Retire legacy lead and Research Lab data

The operator explicitly retired the old lead corpus and selected Research Lab
telemetry. Existing old jobs and routes are part of the removal scope, not
reasons to retain these tables. Current Arena uses its own tables and the
shared `qualification_private_icp_sets` bank.

## Deployment order

1. Deploy the committed gateway change through the canonical gateway restart.
   Verify `/epoch/{epoch_id}/leads`, `/validate/`, and `/manifest/*` are absent,
   the old epoch monitor no longer starts, and Arena plus the public chain epoch
   information routes are healthy. The current ICP rotation and audit tasks stay.
2. Apply exact committed migration 232 with the existing protected helper.
   It removes the 15 tables below, seven views, 14 routines, three obsolete
   sequence names, and the `refresh-miner-test-leads` cron job. It removes only
   the lead foreign key from retained validation evidence and updates one
   shared ticket guard to omit deleted loop evidence.
3. Compare retained catalog definitions and SOURCE_ADD fingerprints before and
   after. Verify published Arena results, ICP content, gateway, dashboard,
   sidecar, and normal validator observations. Do not force a new paid scoring
   run or manual weight transaction just to test this cleanup.

## Explicit table scope

- `leads_private`
- `leads_private_backup`
- `miner_test_leads`
- `ops_research_lab_event_monitor_state`
- `ops_research_lab_event_notifications`
- `research_lab_auto_research_loop_events`
- `research_lab_provider_cost_events`
- `research_lab_provider_outcome_checkpoints_v2`
- `research_lab_scoring_category_results`
- `research_lab_scoring_dispatch_events`
- `research_lab_scoring_icp_events`
- `research_lab_scoring_icp_executions`
- `research_lab_scoring_run_events`
- `research_lab_scoring_runs`
- `test_leads_for_miners`

## Preservation and failure checks

Migration 232 uses one transaction and restricted drops. It rejects unexpected
SQL dependencies, stored-function and cron callers, changed reviewed routine
bodies, and changed shared guards on repeat application. The focused tests
exercise populated target deletion, retained rows, idempotence, and rollback.
A separate structural rehearsal uses the current production table layouts,
functions, views, indexes, constraints, triggers, and policies.

All 11 surviving V2 attestation graph tables stay intact for SOURCE_ADD history.
No standalone `source_add` table exists in this project; migration 198 already
removed its dedicated schema before this cleanup. Arena results, provider
accounting, rewards and weights use the retained `lab_arena_*` tables.

Supabase had a completed daily physical backup from September 12 at 09:52 UTC.
PITR was disabled. No fresh full-row export was created; the backup does not
promise recovery of rows written after that backup.

The current dashboard has no source reference to any of the 15 targets. Legacy
dashboard UI and other historical tables are separate follow-up candidates.
Dormant old source can still name removed tables, including an internal
qualification-admission capability without a current host or Arena caller.
This change removes the active routes and scheduler; it does not revive or
redesign those retired mechanisms.
