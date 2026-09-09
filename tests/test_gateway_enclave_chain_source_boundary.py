from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from Leadpoet.utils.subnet_epoch import CUTOVER_JSON_ENV, SubnetEpochCutover
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from tests.v2_epoch_test_utils import epoch_test_environment


ROOT = Path(__file__).resolve().parents[1]
TEST_HOST = "test.finney.opentensor.ai"
FINNEY_HOST = "entrypoint-finney.opentensor.ai"


def _testnet401_environment():
    cutover = SubnetEpochCutover(
        network_genesis_hash=(
            "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        ),
        netuid=401,
        cutover_block=7_700_000,
        cutover_block_hash="0x" + "4" * 64,
        first_subnet_epoch_index=1,
        first_settlement_epoch_id=1,
        last_legacy_epoch_id=0,
    )
    return {
        "BITTENSOR_NETWORK": "test",
        "BITTENSOR_NETUID": "401",
        CUTOVER_JSON_ENV: json.dumps(cutover.to_dict()),
    }


def _configuration(environment):
    return {
        "research_lab_execution_config": build_research_lab_execution_config(
            environment=environment
        )
    }


def _fresh_process(configuration, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    program = (
        "import json,sys; "
        f"sys.path.insert(0,{str(ROOT)!r}); "
        f"sys.path.insert(0,{str(ROOT / 'gateway' / 'tee')!r}); "
        "from gateway.tee import tee_service as service; "
        f"config=json.load(open({str(config_path)!r},encoding='utf-8')); "
        "boundary=service._configure_v2_chain_source_boundary(config); "
        "from leadpoet_canonical import weight_authority_v2 as weight; "
        "from leadpoet_canonical import compact_auditor_authority_v2 as compact; "
        "print('BOUNDARY='+json.dumps([boundary['chain_host'],"
        "weight.CHAIN_ENDPOINT_HOST,compact.CHAIN_ENDPOINT_HOST]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    marker = [line for line in completed.stdout.splitlines() if line.startswith("BOUNDARY=")]
    assert len(marker) == 1
    return json.loads(marker[0].split("=", 1)[1])


def test_testnet401_runtime_binds_canonical_consumers_before_import(tmp_path):
    assert _fresh_process(_configuration(_testnet401_environment()), tmp_path) == [
        TEST_HOST,
        TEST_HOST,
        TEST_HOST,
    ]


def test_finney_runtime_keeps_default_canonical_boundary(tmp_path):
    assert _fresh_process(_configuration(epoch_test_environment()), tmp_path) == [
        FINNEY_HOST,
        FINNEY_HOST,
        FINNEY_HOST,
    ]


def test_unapproved_profile_destination_fails_before_consumer_import(tmp_path):
    configuration = _configuration(_testnet401_environment())
    configuration["research_lab_execution_config"]["epoch_authority"][
        "chain_signing_profile"
    ]["chain_endpoint"] = "wss://unapproved.example:443"
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    program = (
        "import json,sys; "
        f"sys.path.insert(0,{str(ROOT)!r}); "
        f"sys.path.insert(0,{str(ROOT / 'gateway' / 'tee')!r}); "
        "from gateway.tee import tee_service as service; "
        f"config=json.load(open({str(config_path)!r},encoding='utf-8'))\n"
        "try:\n"
        " service._configure_v2_chain_source_boundary(config)\n"
        "except ValueError:\n"
        " print('REJECTED='+json.dumps(["
        "'leadpoet_canonical.weight_authority_v2' in sys.modules,"
        "'leadpoet_canonical.compact_auditor_authority_v2' in sys.modules]))\n"
        "else:\n"
        " raise AssertionError('unapproved profile accepted')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    marker = [
        line for line in completed.stdout.splitlines() if line.startswith("REJECTED=")
    ]
    assert len(marker) == 1
    assert json.loads(marker[0].split("=", 1)[1]) == [False, False]


def test_coordinator_configures_boundary_before_canonical_imports():
    source = (ROOT / "gateway/tee/tee_service.py").read_text(encoding="utf-8")
    function = source[source.index("def get_v2_coordinator_job_manager():") :]
    assert function.index("_configure_v2_chain_source_boundary(configuration)") < (
        function.index("from gateway.tee.coordinator_executor_v2 import")
    )


def test_late_chain_consumer_import_is_rejected(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gateway" / "tee"))
    from gateway.tee import tee_service

    monkeypatch.setitem(sys.modules, "leadpoet_canonical.weight_authority_v2", object())
    with pytest.raises(RuntimeError, match="loaded before boundary"):
        tee_service._configure_v2_chain_source_boundary(
            _configuration(_testnet401_environment())
        )
