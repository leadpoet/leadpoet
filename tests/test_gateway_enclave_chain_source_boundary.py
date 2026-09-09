from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from Leadpoet.utils.subnet_epoch import CUTOVER_JSON_ENV, SubnetEpochCutover
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
    research_lab_execution_config_hash,
)
from gateway.tee.provider_broker_v2 import provider_registry_hash
from tests.test_gateway_runtime_identity_v2 import _configuration as _release_configuration
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
    configuration = _release_configuration()
    execution_config = build_research_lab_execution_config(environment=environment)
    configuration["research_lab_execution_config"] = execution_config
    configuration["research_lab_execution_config_hash"] = (
        research_lab_execution_config_hash(execution_config)
    )
    configuration["provider_registry_hash"] = provider_registry_hash(
        execution_config=execution_config
    )
    return configuration


def _fresh_process(configuration, tmp_path, *, preload_consumers=False):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    preload = (
        "from leadpoet_canonical import weight_authority_v2 as weight; "
        "from leadpoet_canonical import compact_auditor_authority_v2 as compact; "
        if preload_consumers
        else ""
    )
    prefix = (
        "import json,sys; "
        f"sys.path.insert(0,{str(ROOT)!r}); "
        f"sys.path.insert(0,{str(ROOT / 'gateway' / 'tee')!r}); "
        "from tests.test_gateway_runtime_identity_v2 import "
        "_configuration_hash,_manager; "
        f"config=json.load(open({str(config_path)!r},encoding='utf-8')); "
        f"manager=_manager(__import__('pathlib').Path({str(tmp_path / 'manager')!r}))[0]; "
    )
    program = prefix + preload + (
        "manager.configure(configuration=config,"
        "expected_config_hash=_configuration_hash(config)); "
        "from leadpoet_canonical import chain_source_v2 as source; "
        "from leadpoet_canonical import weight_authority_v2 as weight; "
        "from leadpoet_canonical import compact_auditor_authority_v2 as compact; "
        "print('BOUNDARY='+json.dumps([source.CHAIN_ENDPOINT_HOST,"
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


@pytest.mark.parametrize("preload_consumers", (False, True))
def test_finney_runtime_keeps_default_canonical_boundary(
    tmp_path, preload_consumers,
):
    assert _fresh_process(
        _configuration(epoch_test_environment()),
        tmp_path,
        preload_consumers=preload_consumers,
    ) == [
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
        "from tests.test_gateway_runtime_identity_v2 import "
        "_configuration_hash,_manager; "
        f"config=json.load(open({str(config_path)!r},encoding='utf-8'))\n"
        f"manager=_manager(__import__('pathlib').Path({str(tmp_path / 'manager')!r}))[0]\n"
        "try:\n"
        " manager.configure(configuration=config,"
        "expected_config_hash=_configuration_hash(config))\n"
        "except (RuntimeError,ValueError):\n"
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


def test_runtime_binds_boundary_before_exposing_configuration():
    source = (ROOT / "gateway/tee/runtime_identity_v2.py").read_text(
        encoding="utf-8"
    )
    configure = source[source.index("    def configure(") :]
    assert configure.index("_configure_chain_source_boundary(normalized)") < (
        configure.index("self._runtime_configuration = config_document")
    )


def test_late_chain_consumer_import_is_rejected(monkeypatch):
    from gateway.tee.runtime_identity_v2 import _configure_chain_source_boundary

    monkeypatch.setitem(sys.modules, "leadpoet_canonical.weight_authority_v2", object())
    with pytest.raises(RuntimeError, match="loaded before boundary"):
        _configure_chain_source_boundary(_configuration(_testnet401_environment()))
