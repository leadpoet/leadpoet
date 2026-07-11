from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_SOURCE = ROOT / "neurons" / "validator.py"


def _between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_research_lab_fallback_uses_shared_allocation_default():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "RESEARCH_LAB_FALLBACK_SHARE = _env_percent_share(",
        "RESEARCH_LAB_SHARE = _doc_percent_share(",
    )

    assert "DEFAULT_RESEARCH_LAB_EMISSION_PERCENT" in snippet
    assert "20.0" not in snippet


def test_epoch_debug_line_includes_absolute_and_within_epoch_blocks():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "# DEBUG: Always log epoch status",
        "# Check if we've already processed this epoch",
    )

    assert "Block: {current_block}" in snippet
    assert "Epoch block: {blocks_into_epoch}/360" in snippet


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
