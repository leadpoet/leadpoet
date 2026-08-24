# Gateway liveness watchdog

## The gap this closes

`gw_restart.sh` launches the gateway as a detached background process:

```bash
setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main > "$GATEWAY_LOG_FILE" 2>&1 < /dev/null &
```

The script then polls `http://localhost:8000/health` until startup succeeds and
exits. From that point nothing on the host supervises `gateway.main`. There is
no systemd unit for it, no restart policy and no watchdog, so if the process
dies — a crash, an OOM kill, an unhandled exception in a background task — the
host is left with nothing listening on port 8000 until a person notices and
re-runs `gw_restart.sh` by hand.

That is the whole reason gateway outages here are measured in hours. In the
three weeks to 2026-08-24 the gateway went completely silent five times:

| start (UTC) | duration | how it ended |
|---|---|---|
| 2026-08-08 08:21 | 9h 38m | manual restart |
| 2026-08-15 01:36 | 4h 13m | manual restart |
| 2026-08-16 02:37 | 62m | manual restart |
| 2026-08-20 05:19 | 95m | manual restart |
| 2026-08-24 17:08 | — | manual restart |

Every one of them stopped mid-second with no errors and no slowdown
beforehand, and every one of them ran until someone was available to act.

## What the watchdog does

`scripts/gateway_liveness_watchdog.sh` is a single liveness check, run once a
minute by `gateway-liveness-watchdog.timer`. It considers the gateway healthy
when both of these hold:

- a `gateway.main` process matching the production interpreter is running, and
- `GET http://localhost:8000/health` returns 2xx within 5 seconds.

When they do not hold for three consecutive checks, it runs `gw_restart.sh` —
the same supported recovery path an operator would run. It does **not** change
how the gateway is launched.

## What it deliberately will not do

The failure mode a watchdog has to avoid is thrashing a heavy deploy script, so
recovery is gated four ways:

- **Three consecutive failures** (about three minutes) before it acts, so a
  restart already under way or a momentary stall is never interrupted.
- **It stands down while the restart lock is held.** `gw_restart.sh` holds
  `gateway-restart.lock` for the duration of a deploy; if the watchdog cannot
  take that lock, a restart already owns the host and a dead gateway is
  expected.
- **A 30-minute cooldown** between watchdog-initiated restarts.
- **A circuit breaker** at three watchdog restarts in six hours. Past that it
  logs that the gateway is not staying up and stops trying. A crash loop is a
  bug for a human to read, not something to relaunch against a script that
  rebuilds enclaves.

Only one watchdog runs at a time; it takes its own `flock` first, so a timer
firing during a several-minute recovery is a no-op.

Everything it does is appended to `/home/ec2-user/gateway/watchdog.log`, one
line per decision, including the ones where it stood down.

## Install

```bash
sudo install -m 0644 config/systemd/gateway-liveness-watchdog.service \
  /etc/systemd/system/gateway-liveness-watchdog.service
sudo install -m 0644 config/systemd/gateway-liveness-watchdog.timer \
  /etc/systemd/system/gateway-liveness-watchdog.timer
sudo systemctl daemon-reload
sudo systemctl enable --now gateway-liveness-watchdog.timer
```

Verify:

```bash
systemctl list-timers gateway-liveness-watchdog.timer
tail -f /home/ec2-user/gateway/watchdog.log
```

To take it out of the loop without uninstalling anything:

```bash
sudo systemctl disable --now gateway-liveness-watchdog.timer
```

## Tuning

Every threshold is an environment variable with the default in
`scripts/gateway_liveness_watchdog.sh`; override them in the unit with
`Environment=` lines.

| variable | default | meaning |
|---|---|---|
| `GATEWAY_WATCHDOG_REQUIRED_FAILURES` | `3` | consecutive failed checks before acting |
| `GATEWAY_WATCHDOG_HEALTH_TIMEOUT_SECONDS` | `5` | health probe timeout |
| `GATEWAY_WATCHDOG_COOLDOWN_SECONDS` | `1800` | minimum gap between restarts |
| `GATEWAY_WATCHDOG_MAX_RESTARTS` | `3` | restarts allowed in the window |
| `GATEWAY_WATCHDOG_WINDOW_SECONDS` | `21600` | circuit-breaker window |
| `GATEWAY_HEALTH_URL` | `http://localhost:8000/health` | liveness endpoint |
| `GATEWAY_WATCHDOG_LOG_FILE` | `/home/ec2-user/gateway/watchdog.log` | decision log |

## What this is not

This is a stopgap that shortens an outage; it is not a fix for whatever kills
the gateway, and it is not a substitute for running the gateway under a real
supervisor. Two things would still be worth doing:

- **Find the cause.** `gateway.main` writes to `$GATEWAY_LOG_FILE` on the host
  and nothing ships it, which is why all five outages above are unexplained. The
  last thing that log holds before each death is the only evidence there is.
- **Supervise the process properly.** Running `gateway.main` under its own
  systemd unit with `Restart=always` would recover in seconds rather than
  minutes, and would not involve re-running a full deploy. That means moving the
  launch environment `gw_restart.sh` builds into something a unit can load, which
  is a larger change to the most sensitive script in the repo — worth doing
  deliberately, not during an outage.
