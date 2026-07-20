from pathlib import Path

from gateway.research_lab.attested_v2_store import (
    replayable_execution_result_v2,
)
from leadpoet_canonical.weight_authority_v2 import WEIGHT_INPUT_PURPOSES


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "104-research-lab-attested-result-replay-v2.sql"
).read_text(encoding="utf-8")


def test_result_replay_table_is_append_only_and_receipt_bound():
    assert "research_lab_attested_execution_results_v2" in SQL
    assert (
        "REFERENCES public.research_lab_attested_execution_receipts_v2"
        in SQL
    )
    assert "BEFORE UPDATE OR DELETE" in SQL
    assert "prevent_research_lab_attested_v2_mutation()" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "GRANT SELECT, INSERT" in SQL
    assert "GRANT UPDATE" not in SQL
    assert "GRANT DELETE" not in SQL


def test_result_replay_table_is_limited_to_public_weight_authority_results():
    assert "'research_lab_allocation'" in SQL
    assert "'attest_weight_input'" in SQL
    assert "'research_lab.allocation.v2'" in SQL
    assert "'research_lab.champion_input.v2'" in SQL
    assert "'research_lab.anomaly_adjustment_input.v2'" in SQL
    assert "result_doc::TEXT !~*" in SQL
    assert "service_role" in SQL
    assert "openrouter_api_key" in SQL
    assert "proxy-authorization" in SQL
    for role, purpose in WEIGHT_INPUT_PURPOSES.values():
        if role != "gateway_coordinator":
            assert not replayable_execution_result_v2(
                operation="attest_weight_input",
                purpose=purpose,
            )
            continue
        assert repr(purpose) in SQL
        assert replayable_execution_result_v2(
            operation="attest_weight_input",
            purpose=purpose,
        )
