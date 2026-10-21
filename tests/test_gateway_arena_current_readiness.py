"""Final readiness accepts a slow healthy read and bounds failed reads."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shlex
import sys
import subprocess
import threading
import time

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "gw_restart.sh").read_text()
HELPER = SOURCE.split("wait_for_arena_current_readiness() {", 1)[1].split("\n}\n", 1)[0]
HELPER = "wait_for_arena_current_readiness() {" + HELPER + "\n}\n"


def test_slow_healthy_current_read_succeeds(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(6)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    deadline = tmp_path / "timeout.py"
    deadline.write_text("import subprocess,sys\ntry:\n sys.exit(subprocess.run(sys.argv[2:],timeout=float(sys.argv[1])).returncode)\nexcept subprocess.TimeoutExpired:\n sys.exit(124)\n")
    script = f'timeout() {{ {shlex.quote(sys.executable)} {shlex.quote(str(deadline))} "$@"; }}\n' + HELPER
    script += f'\nwait_for_arena_current_readiness http://127.0.0.1:{server.server_port}/arena/v1/current\n'
    try:
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize("failures,expected_calls,expected_status", [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 3, 1)])
@pytest.mark.parametrize("failure_status", [22, 28, 124])
def test_failed_reads_retry_only_within_fixed_bound(tmp_path, failures, expected_calls, expected_status, failure_status):
    script = '''set -eu
calls=0
sleeps=0
timeout() { test "$1" = 15; shift; "$@"; }
curl() {
 test "$1" = --connect-timeout
 test "$2" = 5
 test "$3" = --max-time
 test "$4" = 15
 test "$5" = -fsS
 calls=$((calls + 1))
 if [ "$calls" -le "''' + str(failures) + '''" ]; then return ''' + str(failure_status) + '''; fi
 return 0
}
sleep() { test "$1" = 2; sleeps=$((sleeps + 1)); }
''' + HELPER + '''
status=0
wait_for_arena_current_readiness http://fixture/arena/v1/current || status=$?
printf '%s %s %s' "$status" "$calls" "$sleeps"
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0
    assert result.stdout == f"{expected_status} {expected_calls} {min(failures, 2)}"
