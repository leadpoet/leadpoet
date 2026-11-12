#!/usr/bin/env python3
"""Check three Arena SSH restrictions; --apply installs and reloads sshd."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile

OWNER = 0
MAIN = Path('/etc/ssh/sshd_config')
TARGET = Path('/etc/ssh/sshd_config.d/00-arena-hardening.conf')
PAYLOAD = (b'# Arena: use ec2-user with sudo; retain SSH TCP forwarding.\n'
           b'PermitRootLogin no\nAllowAgentForwarding no\nX11Forwarding no\n')
SETTINGS = {'permitrootlogin': 'no', 'allowagentforwarding': 'no', 'x11forwarding': 'no'}


def require(ok, code):
    if not ok:
        raise RuntimeError(code)


def secure(path, directory=False):
    info = path.lstat()
    require(not stat.S_ISLNK(info.st_mode) and info.st_uid == OWNER
            and not info.st_mode & 0o022
            and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)),
            'unsafe_config_path')


def fingerprint(path):
    secure(path)
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns,
            info.st_uid, stat.S_IMODE(info.st_mode), hashlib.sha256(path.read_bytes()).hexdigest())


def snapshot(main, target):
    return {str(path): fingerprint(path) for path in [main, *sorted(target.parent.glob('*.conf'))]}


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, 'native_command_failed')
    return result.stdout


def effective(sshd, config, user, run, address='127.0.0.1'):
    text = run([sshd, '-T', '-f', str(config), '-C', 'user='+user+',host=localhost,addr='+address])
    return dict(line.split(' ', 1) for line in text.splitlines() if line)


def contexts(sshd, config, run):
    return {(user, address): effective(sshd, config, user, run, address)
            for user in ('root', 'ec2-user') for address in ('127.0.0.1', '::1')}


def verify(before, after):
    for user in before:
        require(all(after[user].get(key) == value for key, value in SETTINGS.items()),
                'effective_hardening_conflict')
        require({k: v for k, v in before[user].items() if k not in SETTINGS}
                == {k: v for k, v in after[user].items() if k not in SETTINGS},
                'unrelated_effective_setting_changed')
        require(after[user].get('allowtcpforwarding') in ('yes', 'all', 'local'),
                'tcp_forwarding_unavailable')


def configure(main=MAIN, target=TARGET, payload=PAYLOAD, *, apply=False,
              sshd='/usr/sbin/sshd', run=command):
    require(payload == PAYLOAD, 'unreviewed_payload')
    secure(main.parent, True)
    secure(target.parent, True)
    before_files = snapshot(main, target)
    existed = target.exists()
    if existed:
        require(target.read_bytes() == payload, 'unknown_existing_dropin')
    text = main.read_text()
    pattern = r'(?m)^[ \t]*Include[ \t]+' + re.escape(str(target.parent/'*.conf')) + r'[ \t]*(?:#.*)?$'
    matches = list(re.finditer(pattern, text))
    require(len(matches) == 1 and not re.search(r'(?mi)^\s*Match\s', text[:matches[0].start()]),
            'global_dropin_include_required')
    before = contexts(sshd, main, run)
    # Expand the actual glob in lexical order, including our prospective 00
    # file in its real position. Prepending it would hide first-value conflicts.
    with tempfile.TemporaryDirectory(prefix='.arena-sshd-', dir=main.parent) as temporary:
        candidate = Path(temporary)/'hardening'
        candidate.write_bytes(payload)
        candidate.chmod(0o600)
        paths = sorted(set([*target.parent.glob('*.conf'), target]))
        includes = '\n'.join('Include '+str(candidate if path == target else path) for path in paths)
        config = Path(temporary)/'sshd_config'
        config.write_text(text[:matches[0].start()] + includes + text[matches[0].end():])
        config.chmod(0o600)
        run([sshd, '-t', '-f', str(config)])
        proposed = contexts(sshd, config, run)
        verify(before, proposed)
        require(snapshot(main, target) == before_files, 'configuration_changed_during_preflight')
        if not apply or existed:
            return {'outcome': 'already_hardened' if existed else 'check_passed', 'changed': False}
        staged = Path(temporary)/'install'
        staged.write_bytes(payload)
        staged.chmod(0o644)
        with staged.open('rb') as stream:
            os.fsync(stream.fileno())
        expected = fingerprint(staged)
        # Link changes ctime. Adopt only the original staged inode and exact
        # reviewed bytes, never a replacement observed after the link.
        os.link(staged, target)
        owned = fingerprint(target)
        staged_after = fingerprint(staged)
        require(owned[:2] == staged_after[:2] == expected[:2]
                and owned[-1] == staged_after[-1] == expected[-1]
                and expected[-1] == hashlib.sha256(payload).hexdigest(),
                'installed_candidate_changed_ownership_unproven')
        try:
            actual = snapshot(main, target)
            require({k: v for k, v in actual.items() if k != str(target)} == before_files,
                    'configuration_changed_during_install')
            run([sshd, '-t', '-f', str(main)])
            verify(before, contexts(sshd, main, run))
            run(['systemctl', 'reload', 'sshd'])
            require(snapshot(main, target) == actual, 'configuration_changed_during_reload')
            verify(before, contexts(sshd, main, run))
        except Exception:
            require(target.exists() and fingerprint(target) == owned, 'rollback_refused_candidate_changed')
            target.unlink()
            # Restore only this file's absence. Never rewrite concurrent files.
            run([sshd, '-t', '-f', str(main)])
            run(['systemctl', 'reload', 'sshd'])
            raise
    return {'outcome': 'installed_reloaded_verified', 'changed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    require(not (args.apply and args.check), 'conflicting_modes')
    require(os.geteuid() == 0, 'root_required')
    require(MAIN == Path('/etc/ssh/sshd_config') and TARGET == Path('/etc/ssh/sshd_config.d/00-arena-hardening.conf'), 'path_differs')
    secure(Path('/run/lock'), True)
    lock_fd = os.open('/run/lock/arena-sshd-hardening.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(lock_fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o077, 'unsafe_lock')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(configure(apply=args.apply)))
    finally:
        os.close(lock_fd)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_class': type(error).__name__,
                          'code': str(error) if type(error) is RuntimeError else 'native_operation_failed'}))
        raise SystemExit(1) from None
