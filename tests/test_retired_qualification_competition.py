from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retired_qualification_routes_are_not_mounted() -> None:
    gateway = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")

    assert "qualification_router" not in gateway
    assert "gateway.qualification.api.router" not in gateway
    assert not (ROOT / "qualification" / "main.py").exists()


def test_retired_qualification_runtime_is_removed() -> None:
    for path in (
        ROOT / "miner_models",
        ROOT / "gateway" / "qualification" / "api",
        ROOT / "qualification" / "validator",
    ):
        assert not any(path.glob("*.py"))

    # The normal Arena validator entry point was restored at9c39c3b5a.
    # It must remain a thin wrapper, without legacy qualification runtime.
    assert {path.name for path in (ROOT / "neurons").glob("*.py")} == {"__init__.py", "validator.py"}
    validator = (ROOT / "neurons" / "validator.py").read_text()
    assert "from lab_arena.validator import main" in validator
    assert "raise SystemExit(main())" in validator
    assert "qualification.validator" not in validator

    for name in (
        "baseline.py",
        "baseline_arms.py",
        "champion.py",
        "emissions.py",
    ):
        assert not (ROOT / "qualification" / "scoring" / name).exists()


def test_public_validator_entrypoint_delegates_to_current_arena_runtime() -> None:
    # The normal validator was restored as a small process entrypoint. Its
    # legacy qualification implementation remains removed.
    assert {path.name for path in (ROOT / "neurons").glob("*.py")} == {
        "__init__.py", "validator.py",
    }
    source = (ROOT / "neurons/validator.py").read_text(encoding="utf-8")
    assert "from lab_arena.validator import main" in source
    assert "raise SystemExit(main())" in source
    assert "qualification" not in source
