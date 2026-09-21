"""Daily environment values agree between round creation and bank generation."""
import pytest
from gateway.tasks.icp_generator import configured_benchmark_icp_count
from lab_arena import contracts, wiring
from lab_arena.service import RoundDefaults, ServiceError

@pytest.mark.parametrize("raw,expected", [(None,10),("",10),("  ",10),("10",10),("15",15),("30",30),("100",100)])
def test_round_and_generator_share_count(monkeypatch, raw, expected):
    monkeypatch.delenv("LAB_ARENA_BENCHMARK_ICP_COUNT", raising=False)
    if raw is not None:
        monkeypatch.setenv("LAB_ARENA_BENCHMARK_ICP_COUNT", raw)
    assert wiring._benchmark_icp_count_from_environment() == expected
    assert configured_benchmark_icp_count() == expected

@pytest.mark.parametrize("raw", ["0","1","101","10.5","true"])
def test_invalid_count_fails_before_round_creation_or_generation(monkeypatch, raw):
    monkeypatch.setenv("LAB_ARENA_BENCHMARK_ICP_COUNT", raw)
    with pytest.raises(ServiceError): wiring._benchmark_icp_count_from_environment()
    with pytest.raises(ValueError): configured_benchmark_icp_count()

@pytest.mark.parametrize("raw,expected", [("",.5),("0.5",.5),("1",1),("0",0)])
def test_margin_override_and_new_defaults(monkeypatch, raw, expected):
    monkeypatch.setenv("LAB_ARENA_PROMOTION_MARGIN", raw)
    assert wiring._promotion_margin_from_environment() == expected
    assert RoundDefaults().benchmark_icp_count == 10
    assert RoundDefaults().promotion_margin == .5
    assert contracts.promotion_margin({}) == 1

@pytest.mark.parametrize("raw", ["NaN","inf","-1","101","true"])
def test_invalid_margin_fails_before_round_creation(monkeypatch, raw):
    monkeypatch.setenv("LAB_ARENA_PROMOTION_MARGIN", raw)
    with pytest.raises(ServiceError): wiring._promotion_margin_from_environment()
