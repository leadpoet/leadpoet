"""Judge evidence uses its scoring contract; miner output keeps its own cap."""
from dataclasses import replace
import signal

import pytest

from lab_arena import contracts, runtime, scoring
from tests.lab_arena.test_lab_arena_runtime import (
    FakeClock, FakeProcess, FakeRunner, make_config, make_spec,
)


def evidence_output():
    # Five bounded source-bearing rows exceed the miner output cap. This is a
    # serialization fixture only, not a claim that a synthetic company fits.
    output = scoring.build_scoring_output('five-company-output-bound', [
        {'company_index': i, 'final_score': 0.0,
         'source_text': ['Bounded source evidence. ' * 75] * 64} for i in range(5)
    ])
    raw = contracts.canonical_json(output).encode()
    assert runtime.MAX_OUTPUT_BYTES < len(raw) < scoring.MAX_SCORING_OUTPUT_BYTES
    assert scoring.scoring_output_from_bytes(raw) == output
    return raw


@pytest.mark.parametrize('termination', ['normal', 'timeout', 'signal'])
def test_runtime_preserves_large_bounded_judge_output_on_each_exit(tmp_path, termination):
    config = make_config(tmp_path)
    spec = replace(make_spec(tmp_path), entry_command=runtime.SCORER_ENTRY_COMMAND,
                   working_dir=runtime.SCORER_WORKING_DIR, wall_clock_seconds=2)
    raw = evidence_output()
    clock = FakeClock()

    def launch(argv):
        spec.output_path.write_bytes(raw)
        return FakeProcess(argv, clock=clock,
                           finish_at=None if termination == 'timeout' else clock(),
                           returncode=-signal.SIGINT if termination == 'signal' else 0)

    args = dict(process_runner=FakeRunner(clock, run_process=launch), clock=clock,
                sleep=clock.sleep, rusage=lambda: (0.0, 0))
    if termination == 'signal':
        with pytest.raises(runtime.RuntimeHostError) as failed:
            runtime.run_sandbox(config, spec, **args)
        assert failed.value.reason == 'sandbox_launcher_signaled'
        assert failed.value.checkpoint_bytes == raw
    else:
        result = runtime.run_sandbox(config, spec, **args)
        assert result.output_bytes == raw and result.output_error is None
        assert result.timed_out is (termination == 'timeout')


def test_scorer_uses_existing_exact_cap_and_explicit_smaller_cap(tmp_path):
    spec = replace(make_spec(tmp_path), entry_command=runtime.SCORER_ENTRY_COMMAND)
    spec.output_dir.mkdir()
    spec.output_path.write_bytes(b' ' * scoring.MAX_SCORING_OUTPUT_BYTES)
    assert len(runtime.read_output(spec)) == scoring.MAX_SCORING_OUTPUT_BYTES
    with pytest.raises(runtime.SandboxOutputError):
        runtime.read_output(spec, max_bytes=runtime.MAX_OUTPUT_BYTES)
    spec.output_path.write_bytes(b' ' * (scoring.MAX_SCORING_OUTPUT_BYTES + 1))
    with pytest.raises(runtime.SandboxOutputError):
        runtime.read_output(spec)


def test_miner_cannot_gain_judge_cap_from_output_contents(tmp_path):
    spec = make_spec(tmp_path)
    spec.output_dir.mkdir()
    spec.output_path.write_bytes(evidence_output())
    with pytest.raises(runtime.SandboxOutputError):
        runtime.read_output(spec)


@pytest.mark.parametrize('kind', ['symlink', 'directory'])
def test_scorer_keeps_regular_file_boundary(tmp_path, kind):
    spec = replace(make_spec(tmp_path), entry_command=runtime.SCORER_ENTRY_COMMAND)
    spec.output_dir.mkdir()
    if kind == 'symlink':
        spec.output_path.symlink_to(spec.input_dir / runtime.INPUT_FILE_NAME)
    else:
        spec.output_path.mkdir()
    with pytest.raises(runtime.SandboxOutputError):
        runtime.read_output(spec)
