"""Use native OpenSSH parsing on disposable configs; never reload host sshd."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from scripts import configure_arena_sshd as subject


@pytest.fixture
def installation(tmp_path, monkeypatch):
    sshd = shutil.which('sshd') or '/usr/sbin/sshd'
    if not Path(sshd).is_file() or not shutil.which('ssh-keygen'):
        pytest.skip('native OpenSSH server and key generator required')
    monkeypatch.setattr(subject, 'OWNER', os.getuid())
    directory = tmp_path/'sshd_config.d'
    directory.mkdir()
    key = tmp_path/'hostkey'
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key)],
                   check=True, capture_output=True)
    main = tmp_path/'sshd_config'
    main.write_text(f'HostKey {key}\nInclude {directory}/*.conf\n'
                    'PasswordAuthentication no\nAllowTcpForwarding yes\nUsePAM no\n')
    target = directory/'00-arena-hardening.conf'
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[0] == 'systemctl':
            assert argv == ['systemctl', 'reload', 'sshd']
            return ''  # Never touch the real host service.
        return subject.command(argv)

    return main, target, sshd, run, calls


def configure(installation, **kwargs):
    main, target, sshd, run, _ = installation
    return subject.configure(main, target, sshd=sshd, run=run, **kwargs)


def test_default_check_uses_native_validation_without_install_or_reload(installation):
    main, target, _, _, calls = installation
    before = main.read_bytes()
    assert configure(installation) == {'outcome': 'check_passed', 'changed': False}
    assert not target.exists() and main.read_bytes() == before
    assert not any(argv[0] == 'systemctl' for argv in calls)


def test_apply_preserves_other_config_and_tcp_forwarding_and_is_idempotent(installation):
    main, target, sshd, run, calls = installation
    other = target.parent/'50-existing.conf'
    other.write_text('PermitTTY yes\n')
    before = {p: p.read_bytes() for p in (main, other)}
    assert configure(installation, apply=True)['changed']
    assert target.read_bytes() == subject.PAYLOAD
    assert target.stat().st_mode & 0o777 == 0o644
    assert all(p.read_bytes() == value for p, value in before.items())
    for user in ('root', 'ec2-user'):
        values = subject.effective(sshd, main, user, run)
        assert all(values[key] == value for key, value in subject.SETTINGS.items())
        assert values['allowtcpforwarding'] == 'yes'
        assert values['passwordauthentication'] == 'no'
    reloads = sum(argv[0] == 'systemctl' for argv in calls)
    assert configure(installation, apply=True)['outcome'] == 'already_hardened'
    assert sum(argv[0] == 'systemctl' for argv in calls) == reloads == 1


@pytest.mark.parametrize('location', ['main', 'earlier-dropin'])
def test_native_first_value_conflict_is_caught_before_install(installation, location):
    main, target, _, _, _ = installation
    if location == 'main':
        main.write_text('AllowAgentForwarding yes\n' + main.read_text())
    else:
        (target.parent/'00-aaa.conf').write_text('AllowAgentForwarding yes\n')
    with pytest.raises(RuntimeError, match='effective_hardening_conflict'):
        configure(installation, apply=True)
    assert not target.exists()


def test_unknown_existing_target_is_never_overwritten(installation):
    target = installation[1]
    target.write_text('PermitRootLogin yes\n')
    with pytest.raises(RuntimeError, match='unknown_existing_dropin'):
        configure(installation, apply=True)
    assert target.read_text() == 'PermitRootLogin yes\n'


def test_invalid_directive_and_unreviewed_payload_do_not_install(installation):
    target = installation[1]
    with pytest.raises(RuntimeError, match='unreviewed_payload'):
        configure(installation, apply=True, payload=b'InvalidDirective yes\n')
    (target.parent/'50-invalid.conf').write_text('InvalidDirective yes\n')
    with pytest.raises(RuntimeError, match='native_command_failed'):
        configure(installation, apply=True)
    assert not target.exists()


@pytest.mark.parametrize('failure', ['validation', 'reload'])
def test_owned_install_is_rolled_back_on_native_validation_or_reload_failure(installation, failure):
    main, target, sshd, native, calls = installation
    failed = False

    def run(argv):
        nonlocal failed
        trigger = (argv[0] == 'systemctl' if failure == 'reload'
                   else argv == [sshd, '-t', '-f', str(main)] and target.exists())
        if trigger and not failed:
            failed = True
            raise RuntimeError('injected_native_failure')
        return native(argv)

    with pytest.raises(RuntimeError, match='injected_native_failure'):
        subject.configure(main, target, sshd=sshd, run=run, apply=True)
    assert failed and not target.exists()
    assert any(argv == ['systemctl', 'reload', 'sshd'] for argv in calls)


def test_concurrent_preflight_change_is_preserved_without_install(installation):
    main, target, sshd, native, _ = installation
    other = target.parent/'50-existing.conf'
    other.write_text('PermitTTY yes\n')

    def run(argv):
        result = native(argv)
        if '-t' in argv:
            other.write_text('# concurrent operator work\nPermitTTY yes\n')
        return result

    with pytest.raises(RuntimeError, match='configuration_changed_during_preflight'):
        subject.configure(main, target, sshd=sshd, run=run, apply=True)
    assert not target.exists() and other.read_text() == '# concurrent operator work\nPermitTTY yes\n'


def test_rollback_refuses_to_delete_a_concurrently_changed_target(installation):
    main, target, sshd, native, _ = installation

    def run(argv):
        if argv[0] == 'systemctl':
            target.write_bytes(b'# concurrent operator work\n')
            raise RuntimeError('reload_failed')
        return native(argv)

    with pytest.raises(RuntimeError, match='rollback_refused_candidate_changed'):
        subject.configure(main, target, sshd=sshd, run=run, apply=True)
    assert target.read_bytes() == b'# concurrent operator work\n'


@pytest.mark.parametrize('unsafe', ['symlink', 'writable'])
def test_unsafe_config_paths_are_refused(installation, unsafe):
    main, target, _, _, _ = installation
    if unsafe == 'symlink':
        target.symlink_to(main)
    else:
        main.chmod(0o666)
    with pytest.raises(RuntimeError, match='unsafe_config_path'):
        configure(installation, apply=True)


def test_checked_in_dropin_matches_exact_installer_payload():
    assert (Path(__file__).resolve().parents[1]/'deploy/ssh/arena-hardening.conf').read_bytes() == subject.PAYLOAD


def test_postlink_replacement_cannot_be_adopted_and_deleted_by_rollback(installation, monkeypatch):
    main, target, sshd, native, _ = installation
    real_link = subject.os.link
    replacement = b'# concurrent replacement\nPermitRootLogin yes\n'

    def replace_immediately(source, destination):
        real_link(source, destination)
        Path(destination).unlink()
        Path(destination).write_bytes(replacement)

    def validation_failure(argv):
        if argv == [sshd, '-t', '-f', str(main)] and target.exists():
            raise RuntimeError('injected_validation_failure')
        return native(argv)

    monkeypatch.setattr(subject.os, 'link', replace_immediately)
    with pytest.raises(RuntimeError, match='installed_candidate_changed_ownership_unproven'):
        subject.configure(main, target, sshd=sshd, run=validation_failure, apply=True)
    assert target.read_bytes() == replacement


def test_ipv6_match_conflict_is_detected_before_install(installation):
    main, target, _, _, _ = installation
    main.write_text(main.read_text() + 'Match Address ::1\n    AllowAgentForwarding yes\n')
    with pytest.raises(RuntimeError, match='effective_hardening_conflict'):
        configure(installation, apply=True)
    assert not target.exists()


def test_both_users_and_both_loopback_contexts_are_checked(installation):
    _, _, _, _, calls = installation
    configure(installation)
    contexts = {argv[argv.index('-C')+1] for argv in calls if '-C' in argv}
    assert contexts == {f'user={user},host=localhost,addr={address}'
                        for user in ('root', 'ec2-user') for address in ('127.0.0.1', '::1')}
