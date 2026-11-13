"""Offline checks for the dedicated Arena migration workflow and authority guards."""
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import arena_rds_migration as subject


@pytest.fixture
def repository(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)
    git('init', '-b', 'main')
    (repo / 'scripts').mkdir()
    path = repo / 'scripts' / '700-routine-arena.sql'
    path.write_text('BEGIN; SELECT 1; COMMIT;\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'reviewed')
    commit = git('rev-parse', 'HEAD').decode().strip()
    real_git = subject.git
    def fixed_fetch(root, *args):
        if args[0] == 'fetch':
            assert args == ('fetch', '--no-tags', subject.REMOTE, 'refs/heads/main')
            (repo / '.git' / 'FETCH_HEAD').write_text(commit + '\n')
            return b''
        return real_git(root, *args)
    monkeypatch.setattr(subject, 'git', fixed_fetch)
    return SimpleNamespace(repo=repo, path=path, commit=commit, git=git,
                           digest=hashlib.sha256(path.read_bytes()).hexdigest())


def test_merged_exact_sql_fetches_fixed_remote(repository):
    r = repository
    assert subject.verified_sql(r.repo, r.path, r.commit, r.digest) == r.path.read_bytes()


@pytest.mark.parametrize('case', ['dirty', 'untracked', 'unmerged', 'wrong_hash', 'outside', 'invalid_commit'])
def test_refuses_untrusted_sql(repository, case):
    r = repository
    path, commit, digest = r.path, r.commit, r.digest
    if case == 'dirty':
        path.write_text('SELECT 2;')
    elif case == 'untracked':
        (r.repo / 'extra').write_text('untracked')
    elif case == 'unmerged':
        path.write_text('SELECT 2;')
        r.git('add', '.')
        r.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'unmerged')
        commit = r.git('rev-parse', 'HEAD').decode().strip()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    elif case == 'wrong_hash':
        digest = '0' * 64
    elif case == 'outside':
        path = r.repo / '700-routine-arena.sql'
    else:
        commit = 'main'
    with pytest.raises(subject.MigrationError):
        subject.verified_sql(r.repo, path, commit, digest)


def state():
    return dict(database='postgres', user=subject.USER, session_user=subject.USER, ssl=True,
                version=170009, recovery=False, flags=[True, False, False, False, False, False, False],
                memberships=[['lab_arena_owner', False, True, True]], schema_ready=True,
                temporary_ready=True, container_owner=False, external_definers=False)


class Connection:
    def __init__(self, authority=None):
        self.authority = state() if authority is None else authority
        self.closed, self.autocommit, self.statements = False, False, []
    def cursor(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def execute(self, sql):
        self.statements.append(sql)
    def fetchone(self):
        return (self.authority,)
    def close(self):
        self.closed = True


@pytest.fixture
def access(tmp_path, monkeypatch):
    ca = tmp_path / 'ca.pem'
    ca.write_text('synthetic CA')
    db = dict(DBInstanceArn=f'arn:aws:rds:{subject.REGION}:{subject.ACCOUNT}:db:{subject.INSTANCE}',
              Endpoint=dict(Address=subject.HOST, Port=5432, HostedZoneId='test'),
              DBName='postgres', Engine='postgres', DBInstanceStatus='available')
    secret = dict(username=subject.USER, password='synthetic-private-value', host=subject.HOST,
                  port=5432, dbname='postgres')
    response = dict(Name=subject.SECRET,
                    ARN=f'arn:aws:secretsmanager:{subject.REGION}:{subject.ACCOUNT}:secret:{subject.SECRET}-ABC123',
                    SecretString=json.dumps(secret))
    calls, connection, options = [], Connection(), {}
    def client(service, **kwargs):
        assert kwargs['config'] is subject.AWS_CLIENT_CONFIG
        if service == 'rds':
            return SimpleNamespace(describe_db_instances=lambda **kw: {'DBInstances': [db]})
        assert service == 'secretsmanager'
        def get(**kw):
            calls.append(kw)
            return response
        return SimpleNamespace(get_secret_value=get)
    monkeypatch.setattr(subject, 'gateway_iam_session', lambda **kw: (
        SimpleNamespace(client=client) if kw == {'profile': 'leadpoet-lab'} else pytest.fail('wrong profile')))
    monkeypatch.setattr(subject.psycopg2, 'connect', lambda **kw: options.update(kw) or connection)
    return SimpleNamespace(ca=ca, db=db, secret=secret, response=response, calls=calls,
                           connection=connection, options=options)


def test_dedicated_secret_and_tls_only(access):
    a = access
    assert subject.connect(25551, a.ca) is a.connection
    assert a.calls == [{'SecretId': subject.SECRET}]
    assert a.options['host'] == subject.HOST and a.options['hostaddr'] == '127.0.0.1'
    assert a.options['user'] == subject.USER and a.options['dbname'] == 'postgres'
    assert a.options['sslmode'] == 'verify-full' and a.connection.autocommit is True


@pytest.mark.parametrize('case', ['host', 'account', 'db_name', 'secret_name', 'secret_arn',
                                  'user', 'secret_host', 'password', 'malformed', 'role', 'tls'])
def test_wrong_target_secret_or_authority_refused(access, case, capsys):
    a = access
    if case == 'host': a.db['Endpoint']['Address'] = 'wrong.invalid'
    elif case == 'account': a.db['DBInstanceArn'] = a.db['DBInstanceArn'].replace(subject.ACCOUNT, '111111111111')
    elif case == 'db_name': a.db['DBName'] = 'wrong'
    elif case == 'secret_name': a.response['Name'] = 'master-secret'
    elif case == 'secret_arn': a.response['ARN'] = a.response['ARN'].replace(subject.ACCOUNT, '111111111111')
    elif case == 'role': a.connection.authority['memberships'].append(['rds_superuser', False, True, True])
    elif case == 'tls': a.connection.authority['ssl'] = False
    else:
        if case == 'user': a.secret['username'] = 'arena_admin'
        elif case == 'secret_host': a.secret['host'] = 'wrong.invalid'
        elif case == 'password': a.secret['password'] = ''
        a.response['SecretString'] = 'bad-json synthetic-private-value' if case == 'malformed' else json.dumps(a.secret)
    assert subject.main(['--check-access', '--local-port', '25551', '--ca-file', str(a.ca)]) == 1
    output = capsys.readouterr()
    assert 'synthetic-private-value' not in output.out + output.err
    assert all(call == {'SecretId': subject.SECRET} for call in a.calls)
    if case in {'role', 'tls'}: assert a.connection.closed


@pytest.mark.parametrize('sql', [b'BEGIN; SELECT 1; COMMIT;',
                               b'CREATE INDEX CONCURRENTLY test_idx ON public.lab_arena_runs(run_id);'])
def test_apply_keeps_exact_sql_and_autocommit(access, monkeypatch, sql):
    a = access
    monkeypatch.setattr(subject, 'verified_sql', lambda *_: sql)
    assert subject.main(['--apply', 'scripts/700-routine.sql', '--commit', 'a'*40,
                         '--expected-sha256', 'b'*64, '--local-port', '25551',
                         '--ca-file', str(a.ca)]) == 0
    assert a.connection.statements == [subject.ROLE_QUERY, sql.decode()]
    assert a.connection.autocommit is True and a.connection.closed


def test_provider_errors_redacted_without_master_fallback(access, monkeypatch, capsys):
    def fail(**_):
        raise RuntimeError('synthetic-private-value provider details')
    monkeypatch.setattr(subject, 'gateway_iam_session', fail)
    assert subject.main(['--check-access', '--local-port', '25551', '--ca-file', str(access.ca)]) == 1
    assert 'synthetic-private-value' not in capsys.readouterr().err
    assert access.calls == []


def test_privilege_failure_is_clear_and_redacted(access, monkeypatch, capsys):
    monkeypatch.setattr(subject, 'verified_sql', lambda *_: b'SELECT 1')
    def execute(sql):
        if sql != subject.ROLE_QUERY:
            error = type('PrivilegeError', (Exception,), {'pgcode': '42501'})('synthetic-private-value')
            raise error
    monkeypatch.setattr(access.connection, 'execute', execute)
    assert subject.main(['--apply', 'scripts/700-routine.sql', '--local-port', '25551',
                         '--ca-file', str(access.ca)]) == 1
    result = capsys.readouterr().err
    assert 'privileged channel' in result and 'synthetic-private-value' not in result
    assert access.connection.closed


def test_clean_mutated_file_still_fails_original_binding(repository):
    r = repository
    r.path.write_text('SELECT 99;')
    r.git('add', '.')
    r.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'changed')
    with pytest.raises(subject.MigrationError, match='bytes or hash'):
        subject.verified_sql(r.repo, r.path, r.commit, r.digest)


def test_checkout_must_include_latest_fetched_main(repository, monkeypatch):
    r = repository
    r.path.write_text('SELECT 2;')
    r.git('add', '.')
    r.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'new main')
    latest = r.git('rev-parse', 'HEAD').decode().strip()
    r.git('checkout', '--detach', r.commit)
    previous = subject.git
    def newer_fetch(repo, *args):
        if args[0] == 'fetch':
            (r.repo / '.git' / 'FETCH_HEAD').write_text(latest + '\n')
            return b''
        return previous(repo, *args)
    monkeypatch.setattr(subject, 'git', newer_fetch)
    with pytest.raises(subject.MigrationError):
        subject.verified_sql(r.repo, r.path, r.commit, r.digest)


def test_verify_only_never_reads_credentials(repository, monkeypatch):
    r = repository
    monkeypatch.setattr(subject, 'gateway_iam_session', lambda **_: pytest.fail('credential access'))
    assert subject.main(['--verify-only', str(r.path), '--repo', str(r.repo), '--commit', r.commit,
                         '--expected-sha256', r.digest]) == 0


def test_bad_authority_never_executes_migration(access, monkeypatch):
    a = access
    a.connection.authority['flags'][3] = True
    monkeypatch.setattr(subject, 'verified_sql', lambda *_: b'SELECT forbidden_migration()')
    assert subject.main(['--apply', 'scripts/700-routine.sql', '--local-port', '25551',
                         '--ca-file', str(a.ca)]) == 1
    assert a.connection.statements == [subject.ROLE_QUERY]
    assert a.connection.closed


def test_failed_sql_is_never_replayed(access, monkeypatch):
    a = access
    migration = 'COMMIT; SELECT fail_after_commit();'
    monkeypatch.setattr(subject, 'verified_sql', lambda *_: migration.encode())
    previous = a.connection.execute
    def execute(sql):
        previous(sql)
        if sql == migration:
            raise RuntimeError('private database error')
    monkeypatch.setattr(a.connection, 'execute', execute)
    assert subject.main(['--apply', 'scripts/700-routine.sql', '--local-port', '25551',
                         '--ca-file', str(a.ca)]) == 1
    assert a.connection.statements == [subject.ROLE_QUERY, migration]
    assert a.connection.closed


def test_other_owner_definer_access_blocks_migration(access, monkeypatch):
    access.connection.authority['external_definers'] = True
    monkeypatch.setattr(subject, 'verified_sql', lambda *_: b'SELECT forbidden_migration()')
    assert subject.main(['--apply', 'scripts/700-routine.sql', '--local-port', '25551',
                         '--ca-file', str(access.ca)]) == 1
    assert access.connection.statements == [subject.ROLE_QUERY]
    assert access.connection.closed
