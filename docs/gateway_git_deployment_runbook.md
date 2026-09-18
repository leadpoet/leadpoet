# Gateway deployment

The gateway serves the public API and forwards competition requests to the
Arena service. The normal validator runs scoring jobs and submits weights
with its local hotkey.

## Release

1. Fetch `origin/main`. Keep local and concurrent changes intact.
2. Run the current tests in [the deployment checklist](v2_deployment_verification_checklist.md)
   from the exact candidate checkout. Run real PostgreSQL checks for changes
   to database contracts.
3. Commit and merge the reviewed change. Fetch again and record the exact full
   SHA to deploy. Apply required committed migrations through the configured
   migration helper before starting code that needs them.
4. Invoke the installed canonical gateway controller:

   ```bash
   /home/ec2-user/gw_restart.sh --commit <full-commit-sha>
   ```

   It owns release verification, claim draining, process replacement,
   runtime measurements, and readiness checks. Do not invoke inner controllers
   or replace a live source tree by hand. Respect a rejected epoch gate and
   retry when the canonical controller permits it.
5. If the validator changed, run the exact committed `validator_restart.sh`
   with `VALIDATOR_DEPLOY_COMMIT` set to the same full SHA. See
   [normal validator operations](arena_normal_validator_weights.md).

GitHub workflow status is diagnostic. It does not authorize or block a
canonical restart. Local release identity and cryptographic checks remain
mandatory.

## Verification

Verify the gateway build identity, Arena API, validator installed identity,
and canonical restart results. Check accepted execution and scoring records,
publication, reward state, and finalized chain outcomes. Use a reward-disabled
shadow round for paid diagnostic runs. Do not inject test submissions or
synthetic rewards into the live competition.

A healthy HTTP endpoint alone does not prove scoring or weights are healthy.
Keep the prior release identity and accepted work for recovery, and use the
canonical controllers for any rollback.
