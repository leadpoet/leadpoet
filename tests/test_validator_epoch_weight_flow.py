from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_SOURCE = ROOT / "neurons" / "validator.py"


def _between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_already_processed_epoch_still_checks_weight_submission():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "if current_epoch <= self._last_processed_epoch:",
        'print(f"[DEBUG] Processing epoch {current_epoch} for the FIRST TIME")',
    )

    assert "await self._check_weight_submission_for_processed_epoch(" in snippet
    assert '"already_processed"' in snippet


def test_legacy_sourcing_disabled_path_checks_weight_submission_before_return():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        'if not _env_flag("ENABLE_LEGACY_SOURCING"):',
        "# Fetch assigned leads from gateway",
    )

    assert "await self._check_weight_submission_for_processed_epoch(" in snippet
    assert '"legacy_sourcing_disabled"' in snippet
    assert snippet.index("await self._check_weight_submission_for_processed_epoch(") < snippet.index(
        "self._last_processed_epoch = current_epoch"
    )


def test_gateway_already_submitted_path_checks_weight_submission_before_return():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "if leads is None:",
        'print(f"[DEBUG] Received {len(leads)} leads from gateway',
    )

    assert "await self._check_weight_submission_for_processed_epoch(" in snippet
    assert '"gateway_already_submitted_or_queue_empty"' in snippet
    assert snippet.index("await self._check_weight_submission_for_processed_epoch(") < snippet.index(
        "self._last_processed_epoch = current_epoch"
    )
