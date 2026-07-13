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
