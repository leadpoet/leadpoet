# Paired normal-Arena restart

Use `python3 -B -m scripts.restart_arena_pair` from a clean checkout of the
exact current `origin/main`. Pass `--commit`, explicit SSH `--gateway` and
`--validator` targets, `--identity`, a new private `--logs` directory, and the
same shared `--deployment-lock` used by the deployment owner. Do not run this
alongside another deployment. The canonical scripts retain their epoch,
source, wallet, process, host-lock, readiness and rollback checks.

The gateway must already have the exact committed `gw_restart.sh` bytes
installed by its canonical controller. The paired command checks its hash,
owner, mode and type before starting. A controller that lacks paired support
is refused before deployment. First installation needs one ordinary canonical
controller installation at a safe time; this command does not bootstrap or
silently upgrade the installed controller. An application-only subsequent
release can use the pair if its controller bytes are unchanged.

The opt-in gateway path acquires the existing `all` guard, drains once, and
retains that guard after gateway readiness. The coordinator renews that same
owner and generation. It sends only the existing private, nonce-bound permit
over SSH stdin to the canonical normal-validator restart. The validator
checks and consumes the permit after ordinary preflight and before stopping.
Standalone gateway and validator restarts keep their current default paths.

After the validator returns, the coordinator reads its actual PID, creation
time, cwd and exact source. Within the existing 300-second readiness budget it
requires the existing central startup, weights-ready and scoring-setup-ready
events for that new process and one source/session, with positive slots. The
query has a process-start lower bound, capture-time upper bound and row cap.
It then rereads the same process/source before marking validator ready and
releasing the guard. This proves loop setup, not successful sandbox execution
or a finalized weight transaction. Those remain separate production checks.

A failure before gateway destruction keeps the canonical owner's abort
behavior. After gateway activation, any validator preflight, restart,
rollback, renewal, readiness or authorization failure leaves the exact guard
retained. No failure path releases another owner, clears an operator hold,
restarts again, or forces expiry. Inspect the private canonical logs and the
existing guard state before exact-owner recovery; do not rerun with a fresh
owner or clear the guard. A lost release response is ambiguous and requires
readback before any next action. No service credentials leave the gateway.
