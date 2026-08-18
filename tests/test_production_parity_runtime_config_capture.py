from __future__ import annotations

from pathlib import Path

import pytest

from scripts.capture_production_parity_runtime_config import (
    RuntimeConfigCaptureError,
    capture,
)
from scripts.materialize_production_parity_secrets import (
    SecretMaterializationError,
    _parse_environment_document,
)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("GIT_SSH_COMMAND", "ssh -i /run/key -o IdentitiesOnly=yes"),
        ("LESSOPEN", "| /usr/bin/lesspipe %s"),
        ("SSH_CLIENT", "192.0.2.10 12345 22"),
        ("SSH_CONNECTION", "192.0.2.10 12345 198.51.100.20 22"),
        ("which_declare", "declare -f"),
    ),
)
def test_environment_document_preserves_live_shaped_multi_token_values(
    name: str,
    value: str,
):
    assert _parse_environment_document(
        f"{name}={value}\n", field="production gateway environment"
    )[name] == value


def test_environment_document_preserves_unmatched_quote_as_data():
    value = "ssh -o ProxyCommand='proxy --safe"
    assert _parse_environment_document(
        f"GIT_SSH_COMMAND={value}\n", field="production gateway environment"
    )["GIT_SSH_COMMAND"] == value


def test_environment_document_accepts_only_identical_duplicate_assignments():
    name = "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED"
    assert _parse_environment_document(
        f"{name}=false\n{name}=false\n",
        field="production gateway environment",
    ) == {name: "false"}

    with pytest.raises(SecretMaterializationError, match="conflicting duplicate"):
        _parse_environment_document(
            f"{name}=false\n{name}=true\n",
            field="production gateway environment",
        )


def test_capture_classifies_conflicting_duplicate_as_parse_failure(tmp_path: Path):
    class Client:
        @staticmethod
        def get_secret_value(*, SecretId: str):
            assert SecretId == "production-secret"
            return {"SecretString": "DUPLICATE=value-a\nDUPLICATE=value-b\n"}

    output = tmp_path / "runtime.json"
    with pytest.raises(RuntimeConfigCaptureError, match="could not be parsed"):
        capture(client=Client(), secret_id="production-secret", output=output)
    assert not output.exists()
