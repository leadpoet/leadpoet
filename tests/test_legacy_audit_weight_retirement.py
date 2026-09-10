import pytest
from fastapi import HTTPException

from gateway.api import weights


def test_legacy_weight_retirement_is_an_immediate_file_flip(tmp_path, monkeypatch):
    flag = tmp_path / "legacy-audit-weights.state"
    monkeypatch.setenv(weights.LEGACY_AUDIT_WEIGHT_RETIREMENT_FILE_ENV, str(flag))
    weights._require_legacy_audit_weight_path()
    flag.write_text("enabled\n", encoding="utf-8")
    weights._require_legacy_audit_weight_path()
    flag.write_text("retired\n", encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        weights._require_legacy_audit_weight_path()
    assert error.value.status_code == 410


def test_unconfigured_retirement_control_preserves_legacy_routes(monkeypatch):
    monkeypatch.delenv(weights.LEGACY_AUDIT_WEIGHT_RETIREMENT_FILE_ENV, raising=False)
    weights._require_legacy_audit_weight_path()


def test_retirement_rejects_symlink_and_writable_flag(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.write_text("retired\n", encoding="utf-8")
    flag = tmp_path / "flag"
    flag.symlink_to(target)
    monkeypatch.setenv(weights.LEGACY_AUDIT_WEIGHT_RETIREMENT_FILE_ENV, str(flag))
    weights._require_legacy_audit_weight_path()
    monkeypatch.setenv(weights.LEGACY_AUDIT_WEIGHT_RETIREMENT_FILE_ENV, str(target))
    target.chmod(0o666)
    weights._require_legacy_audit_weight_path()
    target.chmod(0o644)
    with pytest.raises(HTTPException) as error:
        weights._require_legacy_audit_weight_path()
    assert error.value.status_code == 410
