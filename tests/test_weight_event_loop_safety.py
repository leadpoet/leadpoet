def test_validation_nonce_check_uses_async_path():
    from pathlib import Path
    source = Path("gateway/api/validate.py").read_text(encoding="utf-8")
    assert "await check_and_store_nonce_async(" in source
    assert "check_and_store_nonce," not in source
