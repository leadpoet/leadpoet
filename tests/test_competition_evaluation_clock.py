from __future__ import annotations

import os

import pytest

from qualification.scoring.evaluation_clock import (
    EVALUATION_DATE_ENV,
    evaluation_date,
    use_evaluation_date,
)


def test_evaluation_date_context_sets_and_restores_environment(monkeypatch):
    monkeypatch.setenv(EVALUATION_DATE_ENV, "2026-09-01")

    with use_evaluation_date("2026-09-04"):
        assert evaluation_date().isoformat() == "2026-09-04"
        assert os.environ[EVALUATION_DATE_ENV] == "2026-09-04"

    assert os.environ[EVALUATION_DATE_ENV] == "2026-09-01"


def test_evaluation_date_context_rejects_invalid_date(monkeypatch):
    monkeypatch.delenv(EVALUATION_DATE_ENV, raising=False)

    with pytest.raises(ValueError, match="evaluation date"):
        with use_evaluation_date("not-a-date"):
            pass

    assert EVALUATION_DATE_ENV not in os.environ
