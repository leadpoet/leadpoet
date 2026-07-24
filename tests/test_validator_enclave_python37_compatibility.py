import ast
from pathlib import Path


ENCLAVE_ROOT = Path(__file__).parents[1] / "validator_tee" / "enclave"


def test_validator_enclave_sources_are_python37_compatible() -> None:
    for path in sorted(ENCLAVE_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 7))

        # The measured enclave is pinned to CPython 3.7. These methods were
        # added in Python 3.9 and fail only after the signed extrinsic exists.
        assert ".removeprefix(" not in source
        assert ".removesuffix(" not in source
