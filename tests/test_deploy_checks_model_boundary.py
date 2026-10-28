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


def test_all_shards_and_preflight_gate_the_existing_required_check() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    preflight, rest = workflow.split("  test-shards:\n", 1)
    shards, rest = rest.split("\n  test:\n", 1)
    aggregate, docker = rest.split("\n  docker-build:\n", 1)
    assert preflight.count("name: Verify native Codex protocol and broker boundaries") == 1
    assert preflight.count("name: Verify trace integrity and reviewer repair seams") == 1
    assert "needs: preflight" in shards
    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7]" in shards
    assert "fail-fast: false" in shards
    assert "--ci-shard-count=8" in shards
    assert "--ci-shard-manifest=ci-results/shard-${{ matrix.shard }}.json" in shards
    assert "include-hidden-files: true" in shards
    assert "overwrite: true" in shards
    assert "name: Unit and Workflow Tests\n" in aggregate
    assert 'python-version: ["3.11"]' in aggregate
    assert "if: always()" in aggregate
    assert "needs: [preflight, test-shards]" in aggregate
    assert 'test "$PREFLIGHT_RESULT" = success' in aggregate
    assert 'test "$SHARD_RESULT" = success' in aggregate
    assert "python -m scripts.pytest_shard ci-results --count=8" in aggregate
    assert "for shard in 0 1 2 3 4 5 6 7; do" in aggregate
    assert "python -m coverage combine ci-results" in aggregate
    assert "overwrite: true" in aggregate
    assert "needs: test" in docker
