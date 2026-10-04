"""The production-style probe must stay bounded and credential-free."""

import ast

import pytest

from scripts import probe_arena_parallel_runsc as probe


@pytest.mark.parametrize("workers", (9, 20))
def test_parallel_probe_plan(workers, capsys):
    assert probe.main(["--dry-run", "--workers", str(workers)]) == 0
    assert '"sandboxes": %d' % (2 * workers) in capsys.readouterr().out
    for index in range(workers):
        reference = probe._source(index, parallel=False)
        measured = probe._source(index, parallel=True)
        ast.parse(reference)
        ast.parse(measured)
        assert '"query": "probe-%d"' % index in reference
        assert '"Probe %d"' % index in measured
    assert "INTENTIONAL_FAILURE" in probe._source(0, parallel=True)
    assert "time.sleep(600)" in probe._source(1, parallel=True)
