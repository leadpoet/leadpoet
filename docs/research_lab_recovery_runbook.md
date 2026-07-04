# Research Lab Recovery Runbook

Use this only after read-only health checks show stale hosted/scoring rows and an operator has explicitly approved production writes.

## Current Failure Signature

- Gateway processes are alive, but Supabase has `research_loop_run_queue_current.current_queue_status = 'started'` rows older than the stale threshold.
- Matching `research_lab_auto_research_loop_current.current_loop_status = 'running'` rows remain old while workers log idle.
- `gateway.research_lab.trajectory_projector` repeatedly logs `__init__() got an unexpected keyword argument 'brief_id'`.
- `ARWEAVE_CHECKPOINT` uploads succeed but the TX id is not queryable from `transparency_log.arweave_tx_id`.
- Research Lab Arweave audit anchors remain buffered after a gateway restart because the in-memory TEE buffer is empty.
- `GET /fulfillment/results/<non-uuid>` returns a gateway 500 from a PostgREST UUID parse error.

## Local Preparation

```bash
scripts/prepare_gateway_recovery_hotpatch.sh /private/tmp/leadpoet-gateway-recovery-hotpatch.tgz
```

The bundle includes only the runtime files needed for:

- hosted/scoring stale timestamp parsing;
- admin CLI import stability for flat EC2 deployments;
- trajectory backfill compatibility and attempt limiting;
- Arweave checkpoint TX id persistence on insert;
- idempotent Research Lab audit event rebuffering before each checkpoint;
- invalid fulfillment result id handling;
- reduced epoch-fallback warning noise in scoring subprocesses.

## Production Apply

Preferred path from the local workstation after explicit approval:

```bash
LEADPOET_PROD_WRITE_APPROVED=yes scripts/apply_gateway_recovery_hotpatch.sh
```

The script copies the bundle, backs up each replaced file under
`/home/ec2-user/gateway/hotpatch-backups/<timestamp>/`, extracts the hotpatch,
and compiles the touched Python files. It does not restart the gateway and does
not requeue runs.

## Admin CLI Wrapper

Install the production wrapper once per gateway host:

```bash
LEADPOET_PROD_WRITE_APPROVED=yes scripts/install_research_lab_admin_wrapper.sh
```

On the EC2 host, prefer this wrapper for Research Lab admin commands:

```bash
/home/ec2-user/bin/research-lab-admin --help
```

The wrapper runs admin commands from `/home/ec2-user/leadpoet_repo`, sets
`PYTHONPATH` to include the full repo and flat gateway runtime, and preloads
parseable values from `/home/ec2-user/gw.environ` without printing the gateway
env-loader banner. Direct `python3 -m gateway.research_lab.admin ...` from
`/home/ec2-user/gateway` can fail because that flat runtime has a partial
top-level `research_lab` package that shadows the full repo package.

Manual equivalent, if the guarded script is unavailable:

```bash
scp /private/tmp/leadpoet-gateway-recovery-hotpatch.tgz leadpoet-gateway:/tmp/leadpoet-gateway-recovery-hotpatch.tgz
ssh -o BatchMode=yes leadpoet-gateway
set -euo pipefail
cd /home/ec2-user/gateway
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "hotpatch-backups/$STAMP"
tar -tzf /tmp/leadpoet-gateway-recovery-hotpatch.tgz > "hotpatch-backups/$STAMP/file-list.txt"
while IFS= read -r file; do
  if [[ -e "$file" ]]; then
    mkdir -p "hotpatch-backups/$STAMP/$(dirname "$file")"
    cp -a "$file" "hotpatch-backups/$STAMP/$file"
  else
    echo "remote file will be created: $file"
  fi
done < "hotpatch-backups/$STAMP/file-list.txt"
tar -xzf /tmp/leadpoet-gateway-recovery-hotpatch.tgz
python3 -m py_compile \
  fulfillment/api.py \
  research_lab/admin.py \
  research_lab/arweave_audit.py \
  research_lab/chain.py \
  research_lab/maintenance.py \
  research_lab/scoring_worker.py \
  research_lab/trajectory_projector.py \
  research_lab/worker.py \
  tasks/hourly_batch.py \
  utils/logger.py \
  utils/tee_client.py \
  gateway/__init__.py
```

Restart using the same supervisor/manual command currently used for the gateway fleet. If using the current `nohup` style, stop the old `main.py` process and start the gateway again from `/home/ec2-user/gateway`.

## Stale Hosted Run Recovery

After restart, wait one stale-reaper interval and re-run health. If the two stale rows are still `started`, dry-run append-only requeue:

```bash
/home/ec2-user/bin/research-lab-admin requeue-stale-started-runs --dry-run
```

If the discovered plans are correct, apply:

```bash
/home/ec2-user/bin/research-lab-admin requeue-stale-started-runs --write
```

If discovery is unavailable on an older deploy, use the exact-run fallback:

```bash
/home/ec2-user/bin/research-lab-admin requeue-loop --run-id 452f4a0f-acbb-426a-9102-8b2f24addfb2 --reason operator_requeue_stale_started --force --dry-run
/home/ec2-user/bin/research-lab-admin requeue-loop --run-id 0b99bcda-ce90-49b5-a715-391ad7fce632 --reason operator_requeue_stale_started --force --dry-run
```

## Stale Candidate Claim Recovery

Candidate scoring claims are separate from hosted run queue rows. If the
dashboard shows candidates stuck in `evaluating` after a worker restart, or if
stale candidate rows are older than the scoring timeout plus 60 seconds, dry-run
recovery across all scoring shards:

```bash
/home/ec2-user/bin/research-lab-admin recover-stale-candidate-claims --dry-run
```

If the plan only contains expected stale claims, apply the append-only recovery:

```bash
/home/ec2-user/bin/research-lab-admin recover-stale-candidate-claims --write
```

This command reuses the gateway scoring worker's stale-claim recovery path, so
recovered candidates requeue and resume from per-ICP checkpoints where
available.

## Rebase-Failed Candidate Recovery

For candidates stuck at `stale_parent_rebase_failed`, use the recovery command
instead of manually requeueing the old candidate. Dry-run first:

```bash
/home/ec2-user/bin/research-lab-admin recover-rebase-failed --max-batch-size 20
```

If the plan is correct, apply:

```bash
/home/ec2-user/bin/research-lab-admin recover-rebase-failed --max-batch-size 20 --apply
```

## Verification

Run the read-only health script:

```bash
set -a
source .env
set +a
python3 scripts/check_research_lab_operator_health.py
```

Expected recovery signals:

- `alert_stale_started_queue_count=0`
- `alert_stale_running_loop_count=0`
- `alert_stale_active_scoring_count=0` or only fresh active work under the stale threshold
- no repeating `research_lab_corpus_traces_backfill_failed ... brief_id` flood after restart
- next successful Arweave checkpoint logs `ARWEAVE_CHECKPOINT` with `arweave_tx_id` populated
- buffered Research Lab Arweave audit anchors start clearing after the next checkpoint rehydrates missing buffered audit events
- malformed fulfillment result ids return 404 instead of surfacing a PostgREST UUID parse failure
