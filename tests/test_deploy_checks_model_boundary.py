from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-checks.yml"
PARITY_WORKFLOW = (
    ROOT / ".github" / "workflows" / "production-parity-fast.yml"
)


def test_deploy_checks_has_no_closed_model_checkout() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "closed-model" not in workflow.lower()
    assert "private model" not in workflow.lower()


def test_all_test_partitions_gate_the_stable_required_check_without_push_coverage():
    import yaml
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert workflow['on']['schedule'] == [{'cron': '17 8 * * *'}]
    assert "github.event_name == 'schedule' && 'nightly'" in workflow['concurrency']['group']
    jobs = workflow['jobs']
    assert all(job['runs-on'] == 'ubuntu-latest' for job in jobs.values())
    shards = jobs['test-shards']
    assert shards['strategy']['matrix']['shard'] == [str(i) for i in range(8)]
    assert shards['strategy']['max-parallel'] == '4'
    assert shards['strategy']['fail-fast'] == 'false'
    partition = next(step['run'] for step in shards['steps']
                     if step['name'] == 'Run complete pytest partition')
    assert 'if [ "${{ github.event_name }}" = schedule ]; then' in partition
    assert '--ci-shard-count=8' in partition
    assert '--ci-shard-manifest=' in partition
    assert '"${coverage_args[@]}"' in partition
    assert jobs['test']['name'] == 'Unit and Workflow Tests'
    assert jobs['test']['if'] == 'always()'
    assert jobs['test']['needs'] == ['preflight', 'test-shards']
    assert jobs['test']['strategy']['matrix']['python-version'] == ['3.11']
    aggregate = jobs['test']['steps']
    gate = aggregate[0]['run']
    assert 'test "$PREFLIGHT_RESULT" = success' in gate
    assert 'test "$SHARD_RESULT" = success' in gate
    assert any('scripts.pytest_shard ci-results --count=8' in step.get('run', '')
               for step in aggregate)
    coverage = [step for step in aggregate if 'coverage' in step['name'].lower()]
    assert coverage and all("github.event_name == 'schedule'" in step['if']
                            for step in coverage)



def test_every_checkout_uses_the_same_immutable_event_commit():
    import yaml
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    checkouts = {
        name: [step for step in job['steps']
               if step.get('uses', '').startswith('actions/checkout@')]
        for name, job in workflow['jobs'].items()
    }
    assert set(checkouts) == {'preflight', 'test-shards', 'test', 'docker-build'}
    for name, steps in checkouts.items():
        assert len(steps) == 1, name
        assert steps[0]['with']['ref'] == '${{ github.event.deployment.sha || github.sha }}', name



def test_sentry_receives_only_trusted_main_workflow_events_without_checkout():
    import yaml
    workflow = yaml.load((WORKFLOW.parent / 'deploy-checks-sentry.yml').read_text(),
                         Loader=yaml.BaseLoader)
    assert set(workflow['on']) == {'workflow_run'}
    event = workflow['on']['workflow_run']
    assert event == {'workflows': ['Deploy Checks'],
                     'types': ['in_progress', 'completed'], 'branches': ['main']}
    assert workflow['permissions'] == {'contents': 'read'}
    job = workflow['jobs']['check-in']
    assert "head_repository.full_name == github.repository" in job['if']
    assert "head_branch == 'main'" in job['if']
    assert "workflow_run.event == 'pull_request'" not in job['if']
    assert job['runs-on'] == 'ubuntu-latest'
    assert all('uses' not in step for step in job['steps'])
    assert len(job['steps']) == 1


import pytest


@pytest.mark.parametrize('action,conclusion,expected', [
    ('in_progress', None, 'in_progress'), ('completed', 'success', 'ok'),
    ('completed', 'failure', 'error'), ('completed', 'timed_out', 'error'),
    ('completed', 'cancelled', 'error'),
])
def test_sentry_check_in_correlates_start_and_completion_without_logging_key(
        tmp_path, monkeypatch, capsys, action, conclusion, expected):
    import json
    import urllib.parse
    import urllib.request
    import yaml
    path = tmp_path / 'event.json'
    path.write_text(json.dumps({'action': action, 'workflow_run': {
        'id': 123, 'run_attempt': 2, 'conclusion': conclusion,
        'run_started_at': '2026-10-29T01:00:00Z',
        'updated_at': '2026-10-29T01:42:00Z',
    }}))
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(path))
    monkeypatch.setenv('GITHUB_REPOSITORY', 'leadpoet/leadpoet')
    monkeypatch.setenv('SENTRY_CRON_URL', 'https://sentry.io/api/123/cron/deploy-checks/test-only-key/')
    observed = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
    def receive(request, timeout):
        observed.append(urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query))
        assert timeout == 30
        return Response()
    monkeypatch.setattr(urllib.request, 'urlopen', receive)
    workflow = yaml.load((WORKFLOW.parent / 'deploy-checks-sentry.yml').read_text(),
                         Loader=yaml.BaseLoader)
    script = workflow['jobs']['check-in']['steps'][0]['run']
    exec(compile(script, 'sentry-check-in', 'exec'), {})
    assert observed[0]['status'] == [expected]
    assert observed[0]['check_in_id'] == ['a2b75d001d3650aba724e195082cbcaa']
    if action == 'completed':
        assert observed[0]['duration'] == ['2520.0']
    assert 'test-only-key' not in capsys.readouterr().out


def test_sentry_check_in_transport_failure_does_not_print_the_authenticated_url(
        tmp_path, monkeypatch, capsys):
    import json
    import urllib.request
    import yaml
    path = tmp_path / 'event.json'
    path.write_text(json.dumps({'action': 'in_progress', 'workflow_run': {
        'id': 123, 'run_attempt': 2,
    }}))
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(path))
    monkeypatch.setenv('GITHUB_REPOSITORY', 'leadpoet/leadpoet')
    monkeypatch.setenv('SENTRY_CRON_URL', 'https://sentry.io/api/123/cron/deploy-checks/test-only-key/')
    def fail(request, timeout):
        raise ValueError(request.full_url)
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    workflow = yaml.load((WORKFLOW.parent / 'deploy-checks-sentry.yml').read_text(),
                         Loader=yaml.BaseLoader)
    script = workflow['jobs']['check-in']['steps'][0]['run']
    with pytest.raises(SystemExit, match='^Sentry Cron check-in failed$'):
        exec(compile(script, 'sentry-check-in', 'exec'), {})
    captured = capsys.readouterr()
    assert 'test-only-key' not in captured.out + captured.err
