#!/usr/bin/env python3
"""Run the two canonical restarts under one existing all-component claim guard.

Service credentials stay on the gateway. Failed post-gateway deployments retain
that exact guard for owner recovery; this command never clears another owner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time

from scripts.lab_arena_restart_guard_handoff import _permit

PREFIX = "ARENA_PAIRED_GUARD="
GATEWAY_ROOT = "/home/ec2-user/leadpoet_repo"
VALIDATOR_ROOT = "/home/ec2-user/leadpoet/leadpoet"
GATEWAY_PYTHON = "/home/ec2-user/venv311/bin/python3"
GATEWAY_ENV = "/home/ec2-user/.config/leadpoet/gateway.env"


class PairError(RuntimeError):
    pass


def ready_session(rows, *, candidate, started_at):
    """Use existing same-session readiness, never another host/old process."""
    sessions = {}
    for row in sorted(rows, key=lambda row: row["validator_event_id"]):
        content = row.get("content") or {}
        if (content.get("validator_source_commit") != candidate
                or datetime.fromisoformat(row["created_at"]) < datetime.fromisoformat(started_at)
                or datetime.fromisoformat(row["occurred_at"]) < datetime.fromisoformat(started_at)):
            continue
        session = content.get("session_id")
        if not isinstance(session, str):
            continue
        entry = sessions.setdefault(session, {"startup": False, "ready": set(), "errors": set()})
        phase = content.get("phase")
        if row.get("event_kind") == "validator.startup":
            entry["startup"] = True
        elif row.get("event_kind") == "validator.ready":
            if phase == "weights" or (phase == "scoring_setup" and type(content.get("ready_slots")) is int
                                      and content["ready_slots"] > 0):
                entry["ready"].add(phase)
        elif row.get("event_kind") == "validator.error":
            if phase in {"startup", "weights", "scoring_setup", "scoring_cycle"}:
                entry["errors"].add(phase)
        elif row.get("event_kind") == "validator.recovered":
            entry["errors"].discard(phase)
    valid = [session for session, value in sessions.items()
             if value["startup"] and not value["errors"] and value["ready"] >= {"weights", "scoring_setup"}]
    return valid[0] if len(valid) == 1 else None


def shell(argv):
    return shlex.join(str(item) for item in argv)


class Pair:
    def __init__(self, args):
        self.args = args
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
                    "-i", str(args.identity)]
        self.invocation = None
        self.generation = None
        self.stop = threading.Event()
        self.renew_error = None

    def remote(self, host, command, **kwargs):
        return subprocess.run([*self.ssh, host, command], check=True, **kwargs)

    def guard(self, action, *extra):
        command = shell([GATEWAY_PYTHON, "-B", "-m", "scripts.lab_arena_restart_claim_guard",
                         action, "--candidate", self.args.commit, "--invocation", self.invocation,
                         "--generation", self.generation, "--environment-file", GATEWAY_ENV, *extra])
        # The canonical checkout is changed only by the gateway restart. Do not
        # run authority from a different release if another operation moved it.
        command = ("cd " + shell([GATEWAY_ROOT]) + " && test \"$(git rev-parse HEAD)\" = "
                   + shell([self.args.commit]) + " && env -u LAB_ARENA_SUPABASE_URL "
                   "-u LAB_ARENA_SERVICE_KEY PYTHONDONTWRITEBYTECODE=1 " + command)
        result = self.remote(self.args.gateway, command, capture_output=True, text=True)
        return json.loads(result.stdout)

    def renew(self):
        state = self.guard("drain", "--scope", "all", "--timeout-seconds", "30")
        if (state.get("guard_generation") != self.generation
                or state.get("restart_scope") != "all"
                or state.get("preserved") is not True):
            raise PairError("paired restart guard changed")

    def renew_loop(self):
        while not self.stop.wait(30):
            try:
                self.renew()
            except Exception:
                # Do not kill a canonical restart or release claims on RPC failure.
                self.renew_error = PairError("paired restart renewal failed; guard retained")
                return

    def run(self):
        # An N-1 controller must never acquire a gateway-only guard and then
        # enter the new paired post-activation path. Installation is explicit.
        expected = hashlib.sha256(subprocess.check_output(
            ["git", "show", self.args.commit + ":gw_restart.sh"],
            cwd=Path(__file__).resolve().parents[1])).hexdigest()
        installed = "/home/ec2-user/gw_restart.sh"
        self.remote(self.args.gateway,
                    "test ! -L " + installed + " && test -f " + installed
                    + " && test \"$(stat -c %u:%a " + installed + ")\" = \"$(id -u):700\""
                    + " && test \"$(sha256sum " + installed + " | cut -d' ' -f1)\" = " + expected,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gateway_log = self.args.logs / "gateway.log"
        with gateway_log.open("x") as log:
            self.remote(self.args.gateway,
                        "GATEWAY_PAIRED_RESTART=1 /home/ec2-user/gw_restart.sh --commit "
                        + self.args.commit, stdout=log, stderr=subprocess.STDOUT)
        receipts = [json.loads(line[len(PREFIX):]) for line in gateway_log.read_text().splitlines()
                    if line.startswith(PREFIX)]
        if len(receipts) != 1:
            raise PairError("paired gateway ownership receipt missing or duplicated")
        receipt = receipts[0]
        self.invocation = receipt["invocation"]
        state = receipt["state"]
        if (state.get("candidate_commit") != self.args.commit
                or state.get("restart_scope") != "all"
                or state.get("restart_phase") != "gateway_ready"
                or state.get("guard_active") is not True or state.get("paused") is not True
                or type(state.get("guard_generation")) is not int
                or state["guard_generation"] <= 0):
            raise PairError("paired gateway ownership receipt invalid")
        self.generation = state["guard_generation"]
        self.renew()
        renewal = threading.Thread(target=self.renew_loop, daemon=True)
        renewal.start()
        try:
            authorized = self.guard("authorize", "--phase", "validator_destructive")
            nonce = secrets.token_hex(32)
            permit_args = argparse.Namespace(candidate=self.args.commit, invocation=self.invocation,
                                             scope="all", nonce=nonce, authority_commit=self.args.commit,
                                             expected_generation=self.generation)
            permit = _permit(permit_args, authorized)
            self.restart_validator(permit, nonce)
            self.wait_validator_ready()
        finally:
            self.stop.set()
            renewal.join()
        if self.renew_error:
            raise self.renew_error
        self.renew()
        self.guard("ready", "--phase", "validator_ready")
        self.guard("release")
        return {"candidate_commit": self.args.commit, "guard_generation": self.generation,
                "invocation": self.invocation, "completed_at": datetime.now(timezone.utc).isoformat()}

    def primary_identity(self, remaining):
        program = """import hashlib,json,subprocess
from pathlib import Path
import psutil
service='leadpoet-arena-validator.service'
props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','show',service,'-p','MainPID','-p','ActiveState','-p','SubState'],text=True).splitlines())
assert props['ActiveState']=='active' and props['SubState']=='running'
p=psutil.Process(int(props['MainPID']));root=Path(p.cwd()).resolve()
assert '--check-only' not in p.cmdline() and any(x.endswith('scripts/run_arena_validator.py') for x in p.cmdline())
commit=(root/'.release-commit').read_text().strip()
files=['lab_arena/validator.py','lab_arena/runner.py','lab_arena/wiring.py','lab_arena/validator_logging.py']
assert commit==COMMIT
assert all((root/name).read_bytes()==subprocess.check_output(['git','-C',REPO,'show',commit+':'+name]) for name in files)
print(json.dumps({'pid':p.pid,'created_at':datetime.fromtimestamp(p.create_time(),timezone.utc).isoformat(),'cwd':str(root),'commit':commit}))
"""
        program = ("from datetime import datetime,timezone\nCOMMIT=" + repr(self.args.commit)
                   + "\nREPO=" + repr(VALIDATOR_ROOT) + "\n" + program)
        result = self.remote(self.args.validator,
                             shell(["sudo", "/home/ec2-user/arena-validator-venv311/bin/python3", "-B", "-c", program]),
                             capture_output=True, text=True, timeout=remaining)
        return json.loads(result.stdout)

    def readiness_events(self, hotkey, started_at, remaining):
        program = """import json,os
from pathlib import Path
from scripts.lab_arena_restart_claim_guard import _read_scoped_environment
from lab_arena.store import PostgrestTransport
config=_read_scoped_environment(Path(ENV))
from datetime import datetime,timezone
transport=PostgrestTransport(config['LAB_ARENA_SUPABASE_URL'],service_key=config['LAB_ARENA_SERVICE_KEY'],timeout_seconds=min(8,REMAINING))
# A bounded, read-only GET through the existing validated private transport.
# Its SELECT facade has no timestamp range option; do not fetch whole history.
response=transport._client.get(transport._base_url+'/rest/v1/lab_arena_validator_events',headers=transport._headers,params=[('select','validator_event_id,event_kind,created_at,occurred_at,content'),('validator_hotkey','eq.'+HOTKEY),('created_at','gte.'+STARTED),('created_at','lte.'+datetime.now(timezone.utc).isoformat()),('order','validator_event_id.desc'),('limit','128')])
response.raise_for_status()
print(json.dumps(response.json()))
"""
        program = ("ENV="+repr(GATEWAY_ENV)+"\nHOTKEY="+repr(hotkey)+"\nSTARTED="+repr(started_at)
                   +"\nREMAINING="+repr(remaining)+"\n"+program)
        result = self.remote(self.args.gateway,
                             "cd " + shell([GATEWAY_ROOT]) + " && "
                             + shell([GATEWAY_PYTHON, "-B", "-c", program]), capture_output=True, text=True, timeout=remaining)
        rows = json.loads(result.stdout)
        if not isinstance(rows, list) or len(rows) >= 128:
            raise PairError("paired readiness evidence is incomplete")
        return rows

    def wait_validator_ready(self):
        log = (self.args.logs / "validator.log").read_text()
        hotkeys = set(re.findall(r"Arena local-wallet validator readiness is valid: hotkey=([1-9A-HJ-NP-Za-km-z]{48})", log))
        if len(hotkeys) != 1:
            raise PairError("canonical validator identity missing")
        hotkey = next(iter(hotkeys))
        deadline = time.monotonic() + 300  # Existing canonical READY_TIMEOUT.
        identity = self.primary_identity(300)
        while True:
            if self.renew_error:
                raise self.renew_error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PairError("paired validator loops did not become ready; guard retained")
            session = ready_session(self.readiness_events(hotkey, identity["created_at"], remaining),
                                    candidate=self.args.commit, started_at=identity["created_at"])
            if session:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PairError("paired validator readiness deadline exceeded")
                if self.primary_identity(remaining) != identity:
                    raise PairError("paired validator process changed during readiness")
                if time.monotonic() > deadline:
                    raise PairError("paired validator readiness deadline exceeded")
                (self.args.logs / "validator-ready.json").write_text(json.dumps({**identity, "session_id": session}, indent=2)+"\n")
                return
            if time.monotonic() >= deadline:
                raise PairError("paired validator loops did not become ready; guard retained")
            time.sleep(min(2, max(0, deadline - time.monotonic())))

    def restart_validator(self, permit, nonce):
        # The private permit travels only through SSH stdin, never the logs.
        # The canonical script consumes it after normal preflight, before stop.
        install = ("import os,stat,sys; data=sys.stdin.buffer.read(16385); "
                   "assert 0<len(data)<=16384; "
                   "fd=os.open(sys.argv[1],os.O_WRONLY|os.O_NOFOLLOW); m=os.fstat(fd); "
                   "assert stat.S_ISREG(m.st_mode) and m.st_uid==os.geteuid() and "
                   "stat.S_IMODE(m.st_mode)==0o600 and m.st_size==0; "
                   "os.write(fd,data); os.fsync(fd); os.close(fd)")
        env = shell(["env", "VALIDATOR_DEPLOY_COMMIT=" + self.args.commit,
                     "VALIDATOR_PAIR_PERMIT=$permit", "VALIDATOR_PAIR_INVOCATION=" + self.invocation,
                     "VALIDATOR_PAIR_NONCE=" + nonce, "VALIDATOR_PAIR_GENERATION=" + str(self.generation),
                     "VALIDATOR_PAIR_AUTHORITY_COMMIT=" + self.args.commit])
        env = env.replace("'VALIDATOR_PAIR_PERMIT=$permit'", 'VALIDATOR_PAIR_PERMIT="$permit"')
        command = ("set -euo pipefail; cd " + shell([VALIDATOR_ROOT]) + "; git fetch --no-tags origin main; "
                   "umask 077; permit=$(mktemp /tmp/leadpoet-pair-permit.XXXXXX.json); "
                   "trap 'rm -f -- \"$permit\"' EXIT; "
                   + shell([GATEWAY_PYTHON.replace("venv311", "arena-validator-venv311"), "-B", "-c", install])
                   + ' "$permit"; git show ' + self.args.commit + ":validator_restart.sh | " + env + " bash")
        with (self.args.logs / "validator.log").open("x") as log:
            self.remote(self.args.validator, command, input=json.dumps(permit).encode("ascii"),
                        stdout=log, stderr=subprocess.STDOUT)


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("ERROR: paired restart requires Python 3.11 or newer; no production action started")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--validator", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--deployment-lock", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("exact commit required")
    if any(not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-")
           for host in (args.gateway, args.validator)):
        parser.error("explicit SSH targets required")
    os.umask(0o077)
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "fetch", "origin"], cwd=repo, check=True)
    for ref in ("HEAD", "origin/main"):
        if subprocess.check_output(["git", "rev-parse", ref], cwd=repo, text=True).strip() != args.commit:
            raise PairError("clean exact current main required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise PairError("clean exact current main required")
    if (repo / "AGENTS.md").read_bytes() != (repo / "CLAUDE.md").read_bytes():
        raise PairError("project instructions differ")
    with args.deployment_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.logs.mkdir(mode=0o700, parents=True, exist_ok=False)
        receipt = Pair(args).run()
        (args.logs / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Subprocess exceptions can include the private permit command/nonce.
        # Detailed canonical logs remain private; never render command strings.
        raise SystemExit("ERROR: paired restart failed; inspect private canonical logs and exact owned guard") from None
