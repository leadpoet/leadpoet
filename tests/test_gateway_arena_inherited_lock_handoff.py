"""Exercise the canonical pre-Docker repair with exact Git archives and CAS."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "gw_restart.sh").read_text()


def _function(name: str) -> str:
    return SOURCE[SOURCE.index(name + "() {"):].split("\n}\n", 1)[0] + "\n}\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _setup(base: Path, *, mode: str = "live", backend: str = "closed", leak: str = "leak") -> str:
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "app.txt").write_text("exact serving source\n")
    (repo / "app.py").write_text("# exact source\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "source")
    sha = _git(repo, "rev-parse", "HEAD")
    preflight = base / "preflight" / "scripts"
    preflight.mkdir(parents=True)
    (preflight / "__init__.py").touch()
    shutil.copyfile(ROOT / "scripts/gateway_git_deploy.py", preflight / "gateway_git_deploy.py")
    # Use the actual retirement checks on this fixture's files. Host process,
    # mount and ingress inputs are isolated from the operator's machine.
    (base / "mountinfo").touch()
    (base / "proc").mkdir()
    (preflight / "arena_runtime_cache_retirement.py").write_text(
        "import argparse,json,sys\nfrom pathlib import Path\n"
        f"sys.path=[p for p in sys.path if p not in ({str(preflight)!r}, {str(preflight.parent)!r})]\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts import arena_runtime_cache_retirement as retirement\n"
        f"base=Path({str(base)!r})\n"
        "def ingress(*args):\n"
        "    mode,slot,sha,backend=(base/'ingress').read_text().split()\n"
        "    return dict(mode=mode,slot=int(slot),sha=sha,backend=backend)\n"
        "retirement.inspect_ingress=ingress\n"
        "retirement.os.geteuid=lambda: 0\n"
        # Emulate root's directory-write permission only on already validated,
        # fixture-owned descriptors; keep the real cache and identity checks.
        "def fixture_remove(operation):\n"
        "    def remove(name, *, dir_fd):\n"
        "        mode=retirement.os.fstat(dir_fd).st_mode & 0o7777\n"
        "        retirement.os.fchmod(dir_fd,mode | 0o200)\n"
        "        try: return operation(name,dir_fd=dir_fd)\n"
        "        finally: retirement.os.fchmod(dir_fd,mode)\n"
        "    return remove\n"
        "retirement.os.unlink=fixture_remove(retirement.os.unlink)\n"
        "retirement.os.rmdir=fixture_remove(retirement.os.rmdir)\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--repo-root',type=Path);p.add_argument('--slot',type=int)\n"
        "p.add_argument('--sha');p.add_argument('--owner-uid',type=int)\n"
        "p.add_argument('--defer-if-busy',action='store_true');a=p.parse_args()\n"
        "print(json.dumps(retirement.retire_runtime_caches(repo_root=a.repo_root,\n"
        " runtime_root=base/'runtime',slot=a.slot,sha=a.sha,owner_uid=a.owner_uid,\n"
        " apply=True,defer_if_busy=a.defer_if_busy,mountinfo=base/'mountinfo',proc_root=base/'proc')))\n"
    )
    (preflight / "run_lab_arena_service.py").write_text(
        "import os,json\ndef load_scoped_environment(path):\n"
        "    os.environ['LAB_ARENA_MODE']=json.loads(path.read_text())['mode']\n")
    (preflight / "manage_owned_process_group.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "def _matching_processes(**kwargs):\n"
        "    return [1] if (Path(__file__).parents[2]/'running-driver').exists() else []\n"
        "if __name__ == '__main__':\n"
        "    state=Path(sys.argv[sys.argv.index('--state-file')+1]).read_text()\n"
        "    print('gateway-wrong-owner' if state == 'foreign-owner' else 'gateway-fixture-owner')\n")
    guard = {"schema_version": "leadpoet.lab_arena.restart_guard_state.v1", "guard_present": True,
             "candidate_commit": sha, "restart_scope": "gateway", "restart_phase": "gateway_destructive",
             "guard_generation": 490, "guard_active": True,
             "drain": {"preserved": True, "current_leased_count": 0,
                       "still_leased_count": 0, "lost_or_mutated_count": 0}}
    (base / "guard.json").write_text(json.dumps(guard))
    (preflight / "arena_first_cutover_idle.py").write_text(
        "from pathlib import Path\n"
        "if (Path(__file__).parents[2]/'paid-review-active').exists(): raise SystemExit('paid review active')\n")
    (preflight / "lab_arena_restart_claim_guard.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "assert sys.argv[sys.argv.index('--invocation')+1] == 'gateway-fixture-owner'\n"
        "if sys.argv[1] == 'authorize' and (Path(__file__).parents[2]/'authorize-reject').exists(): raise SystemExit('atomic authority changed')\n"
        "print((Path(__file__).parents[2]/'guard.json').read_text())\n")
    runtime = base / "runtime"
    runtime.mkdir()
    tree = runtime / f"slot1-{sha}"
    tree.mkdir()
    (tree / "app.txt").write_text((repo / "app.txt").read_text())
    shutil.copyfile(repo / "app.py", tree / "app.py")
    (tree / "app.py").chmod(0o400)
    (tree / "app.txt").chmod(0o400)
    tree.chmod(0o500)
    for component in ("api", "edge"):
        (runtime / f"{component}-slot1-process.json").write_text(leak)
    (base / "ingress").write_text(f"dual 1 {sha} {backend}\n")
    (base / "env.json").write_text(json.dumps({"mode": mode}))
    # No live gateway environment; only the hydrated clone exists. Path
    # authority is restored after loading, and mode comes from scoped config.
    (base / "clone.sh").write_text("export LAB_ARENA_MODE=off\nexport GATEWAY_ENV_FILE=/untrusted\n")
    return sha


def _run(base: Path, sha: str, *, failure: str = "") -> subprocess.CompletedProcess[str]:
    functions = "\n".join(_function(name) for name in (
        "arena_runtime_tree", "prepare_arena_runtime_tree", "retire_arena_runtime_tree", "recover_arena_restart_owner",
        "rollback_arena_candidate_before_destructive", "run_lab_arena_restart_guard",
        "require_arena_lock_recovery_idle", "recover_arena_inherited_docker_locks"))
    script = f'''set -euo pipefail
cd {shlex.quote(str(base))}
FIXTURE_BASE="$PWD"
export PYTHONDONTWRITEBYTECODE=1
GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}
LEADPOET_REPO_ROOT="$PWD/repo"
GATEWAY_PREFLIGHT_TREE="$PWD/preflight"
ARENA_RUNTIME_ROOT="$PWD/runtime"
ENV_CLONE="$PWD/clone.sh"
GATEWAY_ENV_FILE="$PWD/env.json"
GATEWAY_ACTIVE_RELEASE_COMPONENT=gateway
GATEWAY_RESTART_GUARD_SCOPE=gateway
GATEWAY_PAIRED_RESTART=0
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=fresh
PREPARED_GATEWAY_SHA={'b' * 40}
GATEWAY_DESTRUCTIVE_PHASE_STARTED=0
FAIL={shlex.quote(failure)}
{functions}
restore_gateway_restart_path_authority() {{ GATEWAY_ENV_FILE="$FIXTURE_BASE/env.json"; }}
sudo() {{ "$@"; }}
verify_controller_process_helper() {{ return 0; }}
arena_component_command() {{
  ARENA_COMPONENT_STATE_FILE="$ARENA_RUNTIME_ROOT/$1-slot$2-process.json"
  ARENA_COMPONENT_COMMAND=(fixture)
}}
arena_slot_ports() {{ ARENA_API_PORT=0; ARENA_EDGE_PORT=0; }}
arena_component_inherited_docker_locks() {{
  arena_component_command "$1" "$2"
  case "$(cat "$ARENA_COMPONENT_STATE_FILE")" in
    leak|foreign-owner) return 0 ;; clean) return 1 ;; *) return 2 ;;
  esac
}}
arena_ingress_tool() {{ cat "$FIXTURE_BASE/ingress"; }}
arena_ingress_transition() {{
  [ "$(cat "$FIXTURE_BASE/ingress")" = "$2 $3 $4 $5" ] || return 1
  [ "$FAIL" != cas ] || return 1
  printf '%s %s %s %s\n' "$6" "$7" "$8" "$9" > "$FIXTURE_BASE/ingress"
  echo "cas:$3:$7" >> "$FIXTURE_BASE/events"
  if [ "$FAIL" = guard-before-retirement ]; then
    "$GATEWAY_PYTHON_BIN" -c 'import json,sys; p=sys.argv[1]; v=json.load(open(p)); v["guard_active"]=False; open(p,"w").write(json.dumps(v))' "$FIXTURE_BASE/guard.json"
  fi
}}
start_arena_component() {{
  [ "$4" = {sha} ] && [ "$(cat "$3/app.txt")" = 'exact serving source' ]
  arena_component_command "$1" "$2"
  echo clean > "$ARENA_COMPONENT_STATE_FILE"
  echo "start:$1:$2" >> "$FIXTURE_BASE/events"
  [ "$FAIL" != "start-$1" ]
}}
stop_arena_component() {{
  echo "stop:$1:$2" >> "$FIXTURE_BASE/events"
  [ "$FAIL" != "stop-$1" ] || return 1
  rm -f "$ARENA_RUNTIME_ROOT/$1-slot$2-process.json"
  if [ "$FAIL" = guard-before-api ] && [ "$1" = edge ] && [ "$2" = 1 ]; then
    "$GATEWAY_PYTHON_BIN" -c 'import json,sys; p=sys.argv[1]; v=json.load(open(p)); v["guard_active"]=False; open(p,"w").write(json.dumps(v))' "$FIXTURE_BASE/guard.json"
  fi
}}
timeout() {{ shift; "$@"; }}
curl() {{
  [ "$FAIL" != health ] || return 1
  if [ "$FAIL" = guard-before-cas ]; then
    "$GATEWAY_PYTHON_BIN" -c 'import json,sys; p=sys.argv[1]; v=json.load(open(p)); v["guard_active"]=False; open(p,"w").write(json.dumps(v))' "$FIXTURE_BASE/guard.json"
  fi
}}
sleep() {{ :; }}
mv() {{ if [ "$1" = -T ]; then shift; fi; [ "$1" != -- ] || shift; "$GATEWAY_PYTHON_BIN" -c 'import os,sys; os.chmod(sys.argv[1],0o700); os.rename(sys.argv[1],sys.argv[2]); os.chmod(sys.argv[2],0o500)' "$1" "$2"; }}
recover_arena_inherited_docker_locks
# The repair must leave the parent candidate, owner, and environment untouched.
[ "$PREPARED_GATEWAY_SHA" = {'b' * 40} ]
[ "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" = fresh ]
[ "$GATEWAY_ENV_FILE" = "$PWD/env.json" ]
echo normal-docker-gates >> "$FIXTURE_BASE/events"
'''
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                          timeout=10, env=os.environ.copy())


def _events(base: Path) -> list[str]:
    path = base / "events"
    return path.read_text().splitlines() if path.exists() else []


def test_exact_same_source_handoff_precedes_normal_docker_gates(tmp_path: Path):
    sha = _setup(tmp_path)
    result = _run(tmp_path, sha)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {sha} closed"
    assert _events(tmp_path) == ["start:api:0", "start:edge:0", "cas:1:0",
                                  "stop:edge:1", "stop:api:1", "normal-docker-gates"]
    assert not (tmp_path / "runtime" / f"slot1-{sha}").exists()
    assert (tmp_path / "runtime" / f"slot0-{sha}" / "app.txt").read_text() == "exact serving source\n"


@pytest.mark.parametrize("failure", ["start-api", "start-edge", "health", "cas"])
def test_before_cas_failure_preserves_old_service_and_cleans_only_candidate(tmp_path: Path, failure: str):
    sha = _setup(tmp_path)
    result = _run(tmp_path, sha, failure=failure)
    assert result.returncode != 0
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {sha} closed"
    assert (tmp_path / "runtime" / "api-slot1-process.json").read_text() == "leak"
    assert (tmp_path / "runtime" / "edge-slot1-process.json").read_text() == "leak"
    assert not (tmp_path / "runtime" / f"slot0-{sha}").exists()
    assert "normal-docker-gates" not in _events(tmp_path)


def test_after_cas_partial_retirement_retries_without_restarting_serving_components(tmp_path: Path):
    sha = _setup(tmp_path)
    first = _run(tmp_path, sha, failure="stop-api")
    assert first.returncode != 0
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {sha} closed"
    assert not (tmp_path / "runtime" / "edge-slot1-process.json").exists()
    assert (tmp_path / "runtime" / "api-slot1-process.json").read_text() == "leak"
    before = len(_events(tmp_path))
    second = _run(tmp_path, sha)
    assert second.returncode == 0, second.stderr
    assert _events(tmp_path)[before:] == ["stop:api:1", "normal-docker-gates"]


@pytest.mark.parametrize("mutation", ["candidate_commit", "restart_scope", "restart_phase", "guard_generation", "source", "foreign-process", "owner"])
def test_wrong_guard_source_or_process_refuses_without_service_change(tmp_path: Path, mutation: str):
    sha = _setup(tmp_path)
    if mutation == "source":
        tree = tmp_path / "runtime" / f"slot1-{sha}"
        tree.chmod(0o700)
        (tree / "app.txt").chmod(0o600)
        (tree / "app.txt").write_text("changed")
    elif mutation == "owner":
        helper = tmp_path / "preflight/scripts/manage_owned_process_group.py"
        helper.write_text(helper.read_text().replace("else 'gateway-fixture-owner'", "else 'gateway-wrong-owner'"))
    elif mutation == "foreign-process":
        (tmp_path / "runtime" / "edge-slot1-process.json").write_text("foreign")
    else:
        guard_path = tmp_path / "guard.json"
        guard = json.loads(guard_path.read_text())
        guard[mutation] = 0 if mutation == "guard_generation" else "wrong"
        guard_path.write_text(json.dumps(guard))
    result = _run(tmp_path, sha)
    assert result.returncode != 0
    assert not _events(tmp_path)
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {sha} closed"


@pytest.mark.parametrize("mode,backend,leak", [("off", "closed", "leak"), ("live", "ready", "leak"), ("live", "closed", "clean")])
def test_unaffected_flows_do_not_handoff(tmp_path: Path, mode: str, backend: str, leak: str):
    sha = _setup(tmp_path, mode=mode, backend=backend, leak=leak)
    result = _run(tmp_path, sha)
    assert result.returncode == 0, result.stderr
    assert _events(tmp_path) == ["normal-docker-gates"]


def test_repair_runs_after_exact_epoch_proof_before_first_docker_artifact_child():
    call = SOURCE.index("\nrecover_arena_inherited_docker_locks\n")
    assert SOURCE.index("--phase prepared_archive") < SOURCE.index('printf \'%s\\n\' "$GATEWAY_RESTART_EPOCH_REPORT"') < call
    assert call < SOURCE.index("if ! start_gateway_offline_artifact_prepare;")


@pytest.mark.parametrize("held,expected", [((7, 8), 0), ((7,), 0), ((8,), 0), ((), 1)])
def test_actual_owned_descriptor_probe_accepts_any_proven_inherited_lock(tmp_path: Path, held: tuple[int, ...], expected: int):
    assert _probe(tmp_path, held=held) == expected


@pytest.mark.parametrize("poison", ["lock-mode", "missing-lock", "unreadable-info", "invalid-proof", "foreign-process", "replaced-process"])
def test_actual_owned_descriptor_probe_refuses_ambiguous_identity(tmp_path: Path, poison: str):
    assert _probe(tmp_path, held=(7, 8), poison=poison) == 2


def _probe(base: Path, *, held: tuple[int, ...], poison: str = "") -> int:
    proc = base / "proc" / "123"
    (proc / "fd").mkdir(parents=True)
    (proc / "fdinfo").mkdir()
    for fd, name in ((7, "resource.lock"), (8, "resource.lock.admission")):
        lock = base / name
        lock.touch(mode=0o600)
        if fd in held:
            (proc / "fd" / str(fd)).symlink_to(lock)
            (proc / "fdinfo" / str(fd)).write_text(f"lock: 1: FLOCK ADVISORY WRITE 999 08:01:123 0 EOF\n")
    if poison == "lock-mode":
        (base / "resource.lock").chmod(0o666)
    if poison == "missing-lock":
        (base / "resource.lock").unlink()
    if poison == "unreadable-info":
        (proc / "fdinfo" / "7").unlink()
    proof = {"ok": True, "pid": 123}
    if poison == "invalid-proof":
        proof["pid"] = True
    scripts = base / "scripts"
    scripts.mkdir()
    (base / "proof.json").write_text(json.dumps(proof))
    (scripts / "manage_owned_process_group.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        f"base=Path({str(base)!r})\n"
        + ("raise SystemExit(1)\n" if poison == "foreign-process" else
           "print((base/'proof.json').read_text())\n"
           + ("(base/'proof.json').write_text(json.dumps({'ok':True,'pid':124}))\n"
              if poison == "replaced-process" else "")))
    function = _function("arena_component_inherited_docker_locks").replace(
        'f"/proc/{pid}/', f'f"{base}/proc/{{pid}}/')
    script = f'''set -uo pipefail
GATEWAY_PREFLIGHT_TREE={shlex.quote(str(base))}
GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}
LEADPOET_DOCKER_OPERATION_LOCK_FILE={shlex.quote(str(base / "resource.lock"))}
arena_component_command() {{ ARENA_COMPONENT_STATE_FILE=fixture; ARENA_COMPONENT_COMMAND=(fixture); }}
{function}
arena_component_inherited_docker_locks api 1 fixture
'''
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=5).returncode


def test_serving_api_with_other_guard_owner_refuses_before_candidate(tmp_path: Path):
    sha = _setup(tmp_path)
    (tmp_path / "runtime/api-slot1-process.json").write_text("foreign-owner")
    assert _run(tmp_path, sha).returncode != 0
    assert not _events(tmp_path)
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {sha} closed"


def test_retirement_retry_preserves_component_with_other_guard_owner(tmp_path: Path):
    sha = _setup(tmp_path)
    assert _run(tmp_path, sha, failure="stop-api").returncode != 0
    (tmp_path / "runtime/api-slot1-process.json").write_text("foreign-owner")
    before = len(_events(tmp_path))
    assert _run(tmp_path, sha).returncode != 0
    assert _events(tmp_path)[before:] == []
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {sha} closed"


def test_unknown_empty_directory_is_preserved_on_archive_cleanup(tmp_path: Path):
    sha = _setup(tmp_path)
    tree = tmp_path / "runtime" / f"slot1-{sha}"
    tree.chmod(0o700)
    (tree / "unknown-data").mkdir()
    tree.chmod(0o500)
    result = _run(tmp_path, sha)
    assert result.returncode != 0
    assert "unknown paths; preserved" in result.stderr
    assert (tree / "unknown-data").is_dir()
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {sha} closed"
    assert (tmp_path / "runtime/api-slot0-process.json").read_text().strip() == "clean"


def test_preflight_bytecode_for_tracked_source_does_not_block_retirement(tmp_path: Path):
    sha = _setup(tmp_path)
    tree = tmp_path / "runtime" / f"slot1-{sha}"
    tree.chmod(0o700)
    cache = tree / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-311.pyc").write_bytes(b"fixture bytecode")
    (cache / "app.cpython-311.pyc").chmod(0o400)
    cache.chmod(0o500)
    tree.chmod(0o500)
    result = _run(tmp_path, sha)
    assert result.returncode == 0, result.stderr
    assert not tree.exists()
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {sha} closed"


@pytest.mark.parametrize("driver", ["serving-state", "alternate-state", "running-without-state"])
def test_driver_presence_refuses_handoff_before_cas(tmp_path: Path, driver: str):
    sha = _setup(tmp_path)
    if driver == "running-without-state":
        (tmp_path / "running-driver").touch()
    else:
        slot = 1 if driver == "serving-state" else 0
        (tmp_path / "runtime" / f"driver-slot{slot}-process.json").write_text("owned driver")
    result = _run(tmp_path, sha)
    assert result.returncode != 0
    assert "driver" in result.stderr
    assert not _events(tmp_path)
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {sha} closed"


@pytest.mark.parametrize('mutation', ['inactive', 'not-preserved', 'current-lease',
    'captured-lease', 'lost-work', 'malformed-drain', 'boolean-generation', 'paid-review'])
def test_recovery_requires_active_guard_and_idle_work_before_any_handoff(tmp_path, mutation):
    sha = _setup(tmp_path)
    path = tmp_path / 'guard.json'
    guard = json.loads(path.read_text())
    if mutation == 'inactive':
        guard['guard_active'] = False
    elif mutation == 'not-preserved':
        guard['drain']['preserved'] = False
    elif mutation == 'current-lease':
        guard['drain']['current_leased_count'] = 1
    elif mutation == 'captured-lease':
        guard['drain']['still_leased_count'] = 1
    elif mutation == 'lost-work':
        guard['drain']['lost_or_mutated_count'] = 1
    elif mutation == 'malformed-drain':
        guard['drain'] = {}
    elif mutation == 'boolean-generation':
        guard['guard_generation'] = True
    else:
        (tmp_path / 'paid-review-active').touch()
    path.write_text(json.dumps(guard))
    result = _run(tmp_path, sha)
    assert result.returncode != 0
    assert not _events(tmp_path)
    assert (tmp_path / 'ingress').read_text().strip() == f'dual 1 {sha} closed'
    assert (tmp_path / 'runtime/api-slot1-process.json').read_text() == 'leak'
    assert (tmp_path / 'runtime/edge-slot1-process.json').read_text() == 'leak'


def test_guard_expiry_during_readiness_prevents_ingress_handoff(tmp_path):
    sha = _setup(tmp_path)
    result = _run(tmp_path, sha, failure='guard-before-cas')
    assert result.returncode != 0
    assert (tmp_path / 'ingress').read_text().strip() == f'dual 1 {sha} closed'
    assert not any(event.startswith('cas:') or event in ('stop:edge:1', 'stop:api:1') for event in _events(tmp_path))
    assert (tmp_path / 'runtime/api-slot1-process.json').read_text() == 'leak'
    assert (tmp_path / 'runtime/edge-slot1-process.json').read_text() == 'leak'


def test_guard_expiry_after_handoff_prevents_retiring_either_old_component(tmp_path):
    sha = _setup(tmp_path)
    result = _run(tmp_path, sha, failure='guard-before-retirement')
    assert result.returncode != 0
    assert (tmp_path / 'ingress').read_text().strip() == f'dual 0 {sha} closed'
    assert _events(tmp_path) == ['start:api:0', 'start:edge:0', 'cas:1:0']
    for component in ('api', 'edge'):
        assert (tmp_path / f'runtime/{component}-slot1-process.json').read_text() == 'leak'
        assert (tmp_path / f'runtime/{component}-slot0-process.json').read_text().strip() == 'clean'


def test_atomic_authorization_failure_refuses_handoff(tmp_path):
    sha = _setup(tmp_path)
    (tmp_path / 'authorize-reject').touch()
    result = _run(tmp_path, sha)
    assert result.returncode != 0
    assert 'atomic authority changed' in result.stderr
    assert not _events(tmp_path)
    assert (tmp_path / 'ingress').read_text().strip() == f'dual 1 {sha} closed'


def test_guard_expiry_between_retirements_preserves_remaining_old_api(tmp_path):
    sha = _setup(tmp_path)
    result = _run(tmp_path, sha, failure='guard-before-api')
    assert result.returncode != 0
    assert (tmp_path / 'ingress').read_text().strip() == f'dual 0 {sha} closed'
    assert _events(tmp_path) == ['start:api:0', 'start:edge:0', 'cas:1:0', 'stop:edge:1']
    assert (tmp_path / 'runtime/api-slot1-process.json').read_text() == 'leak'
    assert not (tmp_path / 'runtime/edge-slot1-process.json').exists()
    assert (tmp_path / 'runtime/api-slot0-process.json').read_text().strip() == 'clean'
    assert (tmp_path / 'runtime/edge-slot0-process.json').read_text().strip() == 'clean'
