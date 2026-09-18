# Restart and weight diagnostics

Sentry is a best-effort host diagnostic. Its configuration, collector, or SDK
must never change application results, release checks, or weight submission.
See [configuration and redaction rules](sentry_error_monitoring.md).

## Current boundaries

| Boundary | Evidence |
|---|---|
| Source preparation and release | Exact commit, local source/build hashes, runtime identity, PCR0, and failed stage |
| Gateway schema preflight | Current Arena and epoch authority capabilities; bounded status and schema identifiers |
| Canonical restarts | Invocation ID, stage durations, readiness result, and process ownership |
| Arena authority reads | Dependency type, bounded retry count, error class, and finalized block |
| Weight signing and submission | Epoch, accepted-state hash, transaction hash, and inclusion status |
| Chain readback | Finalized reveal block and comparison with the expected weight vector |

Release, restart, and weight correlation IDs link sanitized observations.
They are not added to signed requests, reward state, or chain payloads.
No diagnostic event triggers an extra database or chain read.

## Failure behavior

Retries produce bounded breadcrumbs. A terminal logical failure produces one
error event after retry exhaustion. Unknown failures keep their original error
class; a diagnostic label must not hide the cause. Application exceptions and
fail-closed cryptographic checks remain unchanged.

Do not export secrets, source, prompts, private ICPs, contacts, full vectors,
attestation documents, request bodies, signatures, stack locals, or raw process
environments. Failed scrubbing drops the telemetry event. A telemetry failure
must not delay or fail a restart.

## Validation

Run the focused Sentry scrubbing, boundary, operations, and host-runtime tests
with controlled local transports. Verify production health separately from
telemetry. A Sentry event, HTTP response, signed commitment, or restart-ready
marker does not prove a finalized chain reveal.
