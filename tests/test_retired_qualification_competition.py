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

    neurons = ROOT / "neurons"
    assert {p.name for p in neurons.glob("*.py")} == {"__init__.py", "validator.py"}
    assert "from lab_arena.validator import main" in (neurons / "validator.py").read_text()

    for name in (
        "baseline.py",
        "baseline_arms.py",
        "champion.py",
        "emissions.py",
    ):
        assert not (ROOT / "qualification" / "scoring" / name).exists()
