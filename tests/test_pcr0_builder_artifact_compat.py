import ast
from pathlib import Path

from gateway.utils import pcr0_builder


ROOT = Path(__file__).resolve().parents[1]


def test_pcr0_builder_only_stages_v2_artifacts_when_dockerfile_requires_them(
    tmp_path,
):
    tee_dir = tmp_path / "validator_tee"
    tee_dir.mkdir()
    dockerfile = tee_dir / "Dockerfile.enclave"

    dockerfile.write_text("FROM validator-base:v1\n", encoding="utf-8")
    assert not pcr0_builder.validator_v2_artifacts_required(str(tmp_path))

    dockerfile.write_text(
        "COPY .validator-tee-artifacts/runtime.whl /tmp/runtime.whl\n",
        encoding="utf-8",
    )
    assert pcr0_builder.validator_v2_artifacts_required(str(tmp_path))


def test_validator_v2_wheel_keeps_a_valid_distribution_filename():
    source = (ROOT / "validator_tee" / "Dockerfile.enclave").read_text(
        encoding="utf-8"
    )

    wheel = (
        "py_sr25519_bindings-0.2.2-cp37-cp37m-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert f"/tmp/{wheel}" in source
    assert "/tmp/sr25519.whl" not in source


def test_artifact_dir_presence_check_only_applies_to_v2_commits():
    # Legacy commits rmtree the artifact dir before the Docker build, so an
    # unconditional presence check makes every legacy PCR0 rebuild fail.
    tree = ast.parse(
        (ROOT / "gateway" / "utils" / "pcr0_builder.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "build_enclave_and_extract_pcr0"
    )

    parents = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    missing_dir_error = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "artifact directory is missing" in node.value
    )

    guard_names = []
    cursor = missing_dir_error
    while cursor in parents:
        cursor = parents[cursor]
        if isinstance(cursor, ast.If) and isinstance(cursor.test, ast.Name):
            guard_names.append(cursor.test.id)
    assert "requires_v2_artifacts" in guard_names


def test_pcr0_build_uses_module_shutil_without_shadowing_it_locally():
    tree = ast.parse(
        (ROOT / "gateway" / "utils" / "pcr0_builder.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "build_enclave_and_extract_pcr0"
    )

    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "shutil" for alias in node.names)
        for node in ast.walk(function)
    )


def test_drand_builder_restores_runner_ownership_before_cleanup():
    source = (
        ROOT / "validator_tee" / "scripts" / "build_drand_cabi_v2.sh"
    ).read_text(encoding="utf-8")

    assert '-e "HOST_UID=$(id -u)"' in source
    assert '-e "HOST_GID=$(id -g)"' in source
    assert 'chown -R "$HOST_UID:$HOST_GID" /work' in source
