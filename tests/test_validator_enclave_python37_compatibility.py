import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
ENCLAVE_ROOT = ROOT / "validator_tee" / "enclave"
ARENA_SOURCES = (
    ENCLAVE_ROOT / "arena_hotkey.py",
    ENCLAVE_ROOT / "arena_weight_signer.py",
    ENCLAVE_ROOT / "chain_source_v2.py",
    ENCLAVE_ROOT / "tee_service.py",
    ROOT / "leadpoet_canonical" / "arena_weights.py",
)


def test_measured_arena_signer_sources_are_python37_compatible():
    for path in ARENA_SOURCES:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 7))
        assert ".removeprefix(" not in source
        assert ".removesuffix(" not in source
