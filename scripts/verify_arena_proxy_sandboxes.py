#!/usr/bin/env python3
"""Probe real isolated sandboxes with configured exits; no Arena rows or paid calls.

The native coordinator plus every supplied proxy runs the same public HTTPS
check in parallel. Repeat batches exercise slot reuse and independent cleanup.
Only hashes of observed exit addresses are printed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab_arena import runtime
from lab_arena.proxy_workers import ProxyWorkerPool, preflight_proxy_workers, proxy_workers_from_environment
from lab_arena.runner import RunState, WorkerSocketServer, _stage_agent_entrypoint, _attempt_web_egress
from lab_arena.validator_proxy_environment import validator_proxy_environment
from lab_arena.web_egress import WebEgressServer
from scripts._lab_arena_runsc_probe_ci import ProbeApi, make_agent_spec, probe_work_dir

HARNESS = '''
import hashlib,json,os,socket,urllib.request
def run_icp(icp):
    assert not any(name in os.environ for name in ('OPENROUTER_API_KEY','DEEPLINE_API_KEY'))
    assert not any('WEBSHARE_PROXY_' in name or name.startswith('RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_') for name in os.environ)
    assert os.environ['HTTPS_PROXY'].startswith('http://127.0.0.1:')
    try:
        socket.create_connection(('1.1.1.1',443),timeout=1).close()
    except OSError: pass
    else: raise AssertionError('direct network is reachable')
    for url in ('http://169.254.169.254/latest/meta-data/','https://openrouter.ai/api/v1/models'):
        try: urllib.request.urlopen(url,timeout=10).close()
        except Exception: pass
        else: raise AssertionError('blocked destination is reachable')
    with urllib.request.urlopen('https://api.ipify.org?format=json',timeout=40) as response:
        exit_ip=json.load(response)['ip']
    open('/tmp/attempt_marker','x').write('private')
    try: open('/agent/source/shared','w')
    except OSError: pass
    else: raise AssertionError('source is writable')
    return [{'exit_fingerprint':hashlib.sha256(exit_ip.encode()).hexdigest()[:16]}]
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy-env-file', type=Path, required=True)
    parser.add_argument('--runsc-path', type=Path, default=Path('/usr/local/bin/runsc'))
    parser.add_argument('--batches', type=int, default=2)
    args = parser.parse_args()
    if os.geteuid() != 0 or not 1 <= args.batches <= 3:
        raise RuntimeError('probe requires root and one to three batches')
    environment = validator_proxy_environment({'LAB_ARENA_PROXY_ENV_FILE': str(args.proxy_env_file)})
    verified = preflight_proxy_workers(proxy_workers_from_environment(environment))
    pool = ProxyWorkerPool(verified)
    count = min(20, verified.total_process_capacity)
    expected = {verified.native_exit_ip_fingerprint, *(w.exit_ip_fingerprint for w in verified.workers)}
    with probe_work_dir(dry_run=False) as work:
        work.chmod(0o755)
        rootfs = work / 'rootfs'
        for name in ('etc','usr','input','output','run/lab_arena','agent','model'):
            (rootfs / name).mkdir(parents=True, exist_ok=True)
        for name, target in (('bin','usr/bin'),('lib','usr/lib'),('lib64','usr/lib64')):
            (rootfs / name).symlink_to(target)
        cert = ssl.get_default_verify_paths().cafile
        if not cert:
            raise RuntimeError('host TLS certificate bundle is unavailable')
        target = rootfs / cert.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cert, target)
        subprocess.run(['mount','--bind','/usr',str(rootfs/'usr')],check=True)
        try:
            subprocess.run(['mount','-o','remount,bind,ro',str(rootfs/'usr')],check=True)
            (work / 'sandboxes').mkdir(mode=0o700)
            engine = runtime.RunscRuntime(runtime.RuntimeConfig(args.runsc_path,work/'sandboxes'))
            for batch in range(args.batches):
                barrier = threading.Barrier(count)
                def one(index):
                    attempt = work / ('batch%d-%d' % (batch,index))
                    spec = make_agent_spec(attempt,rootfs_path=rootfs)
                    (spec.source_dir/'harness.py').write_text(HARNESS)
                    bridge = _stage_agent_entrypoint(ROOT/'lab_arena/web_egress_bridge.py',attempt/'agent/run',filename='web-egress-bridge.py')
                    spec = replace(spec, sandbox_id='arena-proxy-probe-%d-%d' % (batch,index),web_bridge_path=bridge,wall_clock_seconds=120)
                    state = RunState(lease={'run_id':spec.sandbox_id},lease_token='probe')
                    broker = WorkerSocketServer(spec.socket_path,ProbeApi(),state)
                    with pool.acquire(timeout=5) as slot:
                        with _attempt_web_egress(WebEgressServer(spec.socket_dir/runtime.SANDBOX_WEB_SOCKET_NAME,proxy_url=slot.proxy_url), slot):
                            broker.start()
                            try:
                                barrier.wait(timeout=30)
                                result = engine.run_icp(spec)
                            finally:
                                broker.stop()
                    if result.exit_code != 0 or result.timed_out or result.output_error:
                        # This source is repository-owned and contains no secrets.
                        raise RuntimeError('sandbox probe failed: '+result.stderr.decode(errors='replace')[-1600:])
                    output = json.loads(result.output_bytes)['companies'][0]['exit_fingerprint']
                    assert output == slot.exit_ip_fingerprint, 'exit changed during the attempt'
                    return output
                with ThreadPoolExecutor(max_workers=count) as executor:
                    exits = list(executor.map(one,range(count)))
                assert len(set(exits)) == count and set(exits).issubset(expected)
                print(json.dumps({'batch':batch+1,'concurrent_sandboxes':count,'unique_verified_exits':len(set(exits)),'checks':'passed'}),flush=True)
        finally:
            subprocess.run(['umount','--',str(rootfs/'usr')],check=True)
    print('ARENA_PROXY_SANDBOX_PROBE_PASSED',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
