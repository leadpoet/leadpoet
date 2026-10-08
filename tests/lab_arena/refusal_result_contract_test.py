"""Optional provider-failure audit counts preserve the signed result contract."""
import json
import pytest
from lab_arena import contracts

FIELDS = ("provider_refusal_count", "provider_local_refusal_count", "provider_unresolved_failure_count")

def result():
    return {"schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
            "resource_summary": {"wall_seconds": 1.0, "cpu_seconds": 0.0,
                                 "max_rss_bytes": 0, "stdout_bytes": 0,
                                 "stderr_bytes": 0, "provider_call_count": 26},
            "started_at": "2026-10-08T00:00:00Z", "finished_at": "2026-10-08T00:00:01Z",
            "terminal_status": "accepted"}

def test_old_and_new_result_round_trip():
    old = result()
    assert contracts.validate_run_result(json.loads(json.dumps(old))) == old
    new = result()
    new["resource_summary"].update(dict(zip(FIELDS, (25, 1, 1))))
    assert contracts.validate_run_result(json.loads(json.dumps(new))) == new

@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [-1, True, "25", 1.5])
def test_failure_counts_require_nonnegative_integers(field, bad):
    doc = result()
    doc["resource_summary"][field] = bad
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_run_result(doc)
