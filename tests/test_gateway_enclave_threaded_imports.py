from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_worker_import_probe(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_all_gateway_rpc_runtime_imports_are_worker_thread_safe_across_restarts():
    script = r'''
import ast
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

source = Path("gateway/tee/tee_service.py").read_text(encoding="utf-8")
tree = ast.parse(source, filename="gateway/tee/tee_service.py")
modules = set()
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
        if node.module.startswith(("gateway.", "leadpoet_")):
            modules.add(node.module)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith(("gateway.", "leadpoet_")):
                modules.add(alias.name)

def import_runtime_modules():
    for module_name in sorted(modules):
        importlib.import_module(module_name)
    return len(modules)

with ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="gateway-vsock-rpc",
) as executor:
    count = executor.submit(import_runtime_modules).result(timeout=25)
print(f"threaded_imports_ok={count}")
'''

    for _restart_index in range(2):
        result = _run_worker_import_probe(script)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "threaded_imports_ok=" in result.stdout


def test_enclave_tls_does_not_import_host_tee_clients():
    result = _run_worker_import_probe(
        r'''
import sys
from concurrent.futures import ThreadPoolExecutor

def import_tls():
    import gateway.tee.inter_enclave_tls
    assert "gateway.utils.tee_client" not in sys.modules

with ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="gateway-vsock-rpc",
) as executor:
    executor.submit(import_tls).result(timeout=10)
'''
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_kms_envelope_validation_does_not_import_host_tee_clients():
    result = _run_worker_import_probe(
        r'''
import sys
from concurrent.futures import ThreadPoolExecutor

def import_kms_validation():
    import gateway.utils.tee_kms_provision_v2
    assert "gateway.utils.tee_client" not in sys.modules

with ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="gateway-vsock-rpc",
) as executor:
    executor.submit(import_kms_validation).result(timeout=10)
'''
    )

    assert result.returncode == 0, result.stdout + result.stderr
