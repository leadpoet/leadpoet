# Gateway process supervision

`gateway.main` runs under its own systemd unit, `leadpoet-gateway.service`,
with `Restart=always`. If the process exits for any reason, systemd starts it
again about ten seconds later.

Before this, `gw_restart.sh` launched the gateway with `setsid` and then
exited. Nothing on the host owned the process afterwards, so any exit was an
outage that lasted until a person noticed and ran a deploy — hours, in the
worst cases.

## There is no install step

The unit is installed by the deploy itself. On every run, before the
destructive phase, `gw_restart.sh` calls `ensure_gateway_supervisor_unit`,
which:

1. renders `config/systemd/leadpoet-gateway.service` from the *deployed* tree,
   substituting the repo root, the run user and the launch-environment path;
2. compares it against `/etc/systemd/system/leadpoet-gateway.service` and
   installs plus `daemon-reload`s it only if it differs;
3. `systemctl enable`s it, so the gateway also comes back after a host reboot.

So editing the template in this repository is the whole change: the next
release picks it up. There is nothing to run by hand, and nothing drifts —
the installed unit is regenerated from the tree on every deploy.

## How the unit knows what to launch

The gateway's launch environment is built over the length of `gw_restart.sh`:
secrets pulled from Secrets Manager, release pinning, AWS and Research Lab
configuration. None of that is reproducible from a unit file.

Instead, at the moment it would previously have called `setsid`, the restart
script snapshots the environment with `export -p` into
`/home/ec2-user/.config/leadpoet/gateway-launch-env.sh` (mode `0600`, same
directory and owner as the existing `gateway.env`). The unit's `ExecStart`,
`scripts/gateway_supervised_launch.sh`, sources that snapshot, drops the
variables that only make sense during a restart, and `exec`s
`$GATEWAY_PYTHON_BIN -u -m gateway.main`.

Because the launcher `exec`s, the supervised process is the same process
`gw_restart.sh` used to launch directly: `pgrep -f "^$GATEWAY_PYTHON_BIN -u -m
gateway[.]main$"` still finds it, and the post-restart `/health/v2-authority`
poll is unchanged.

The snapshot is rewritten on every deploy, so a systemd-initiated restart
always replays the environment of the most recent successful release.

## Interaction with a deploy

`Restart=always` means a deploy cannot simply `pkill` the gateway — systemd
would put the outgoing release straight back. So the destructive phase now
runs `systemctl stop leadpoet-gateway.service` immediately before the existing
`pkill` block, and the same stop runs if post-launch verification fails and
the new gateway has to be torn down.

Two readiness loops (`base health` and `wait_for_gateway_v2_authority`) used to
treat "no gateway process right now" as a fatal deploy error. Under
supervision that can just mean systemd is between restarts, so both loops now
keep polling while the unit reports `active`/`activating`. Their intervals and
target routes are untouched.

## Crashloops

`StartLimitIntervalSec=0` disables systemd's start-rate limit, and
`RestartSec=10` bounds how hard a crashlooping gateway hits the host. This is
deliberate: a unit that gives up is a gateway that stays down, which is the
failure being removed here. A gateway that restarts every ten seconds and
never becomes healthy is still visibly broken — `systemctl status
leadpoet-gateway` and `journalctl -u leadpoet-gateway` show the restart count —
but it is broken while serving some traffic rather than none.

## If systemd is unavailable

`ensure_gateway_supervisor_unit` requires `systemctl` on `PATH`, a booted
systemd (`/run/systemd/system`), a readable template and an executable
launcher, and it must be able to install the unit and enable it. If any of
that fails it prints

```
WARNING: systemd process supervision is unavailable on this host
WARNING: gateway.main will be launched unsupervised; if it exits, it will stay
down until someone runs this script again
```

and the deploy continues, launching the gateway exactly as it did before: the
original `setsid` invocation, kept verbatim and kept inline at the launch site.
The same fallback triggers if `systemctl start` fails at the launch point. A
host without systemd therefore loses supervision, not deploys.

All of this runs before the destructive phase, so a systemd problem is visible
while the old gateway is still serving.

## Operating it

```sh
systemctl status leadpoet-gateway
journalctl -u leadpoet-gateway -n 50
tail -f /home/ec2-user/gateway/gateway.log   # the gateway's own log, appended across restarts
sudo systemctl stop leadpoet-gateway         # take the gateway down and keep it down
```

Never `pkill` the gateway directly while the unit is running; stop the unit.
Deploys are still `gw_restart.sh`, unchanged in how they are invoked.
