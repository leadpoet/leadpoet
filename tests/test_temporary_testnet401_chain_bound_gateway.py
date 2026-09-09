from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.hotkey_authority_v2 import validate_chain_signing_profile
from scripts import run_temporary_testnet401_chain_bound_gateway as launcher
from scripts import run_temporary_testnet401_weight_only_validator as validator_launcher


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HOTKEY = "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"


def _documents(tmp_path, *, genesis=None, wrong_hash=False):
    profile = validate_chain_signing_profile(json.loads(
        (ROOT / "validator_tee/enclave/chain_signing_profile_test_v2.json")
        .read_text(encoding="utf-8")
    ))
    if genesis is not None:
        profile["genesis_hash"] = genesis
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    hotkey_path = tmp_path / "hotkey.json"
    hotkey_path.write_text(json.dumps({
        "schema_version": "leadpoet.validator_hotkey_config.v2",
        "validator_hotkey": EXPECTED_HOTKEY,
        "hotkey_public_key": "4" * 64,
        "chain_signing_profile_hash": (
            "sha256:" + "0" * 64 if wrong_hash else sha256_json(profile)
        ),
        "drand_library_path": "/app/validator_tee/enclave/libbittensor_drand_v2.so",
        "drand_library_sha256": "5" * 64,
    }), encoding="utf-8")
    return profile_path, hotkey_path


def _gateway_environment():
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "ALLOWED_NETUIDS": "401",
        "PRIMARY_VALIDATOR_HOTKEYS": EXPECTED_HOTKEY,
        "BT_SUBTENSOR_NETWORK": "test",
        "BT_SUBTENSOR_CHAIN_ENDPOINT": launcher.__dict__.get(
            "CHAIN_ENDPOINT", "wss://test.finney.opentensor.ai:443"
        ),
    }


def test_real_import_boundary_freezes_both_canonical_consumers_to_testnet(tmp_path):
    profile, hotkey = _documents(tmp_path)
    program = (
        f"import json,sys; sys.path.insert(0,{str(ROOT)!r}); "
        "from scripts.run_temporary_testnet401_chain_bound_gateway "
        "import configure_temporary_testnet401_chain_source as c; "
        f"r=c(profile_path={str(profile)!r},hotkey_config_path={str(hotkey)!r},role='gateway'); "
        "from leadpoet_canonical import weight_authority_v2 as w; "
        "from leadpoet_canonical import compact_auditor_authority_v2 as a; "
        "print(json.dumps({'result':r,'weight':w.CHAIN_ENDPOINT_HOST,'auditor':a.CHAIN_ENDPOINT_HOST},sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program], cwd=ROOT,
        env=_gateway_environment(), capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["weight"] == result["auditor"] == "test.finney.opentensor.ai"
    assert result["result"]["network"] == "test"
    assert result["result"]["netuid"] == 401


def test_default_canonical_consumer_import_remains_finney():
    program = (
        f"import json,sys; sys.path.insert(0,{str(ROOT)!r}); "
        "from leadpoet_canonical import weight_authority_v2 as w; "
        "from leadpoet_canonical import compact_auditor_authority_v2 as a; "
        "print(json.dumps([w.CHAIN_ENDPOINT_HOST,a.CHAIN_ENDPOINT_HOST]))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program], cwd=ROOT,
        env=os.environ, capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout) == [
        "entrypoint-finney.opentensor.ai",
        "entrypoint-finney.opentensor.ai",
    ]


@pytest.mark.parametrize("failure", ("genesis", "profile_hash", "environment"))
def test_chain_bound_startup_rejects_wrong_identity_before_gateway_import(
    tmp_path, failure,
):
    profile, hotkey = _documents(
        tmp_path,
        genesis="0" * 64 if failure == "genesis" else None,
        wrong_hash=failure == "profile_hash",
    )
    environment = _gateway_environment()
    if failure == "environment":
        environment["ALLOWED_NETUIDS"] = "71"
    program = (
        f"import sys; sys.path.insert(0,{str(ROOT)!r}); "
        "from scripts.run_temporary_testnet401_chain_bound_gateway "
        "import configure_temporary_testnet401_chain_source as c; "
        f"c(profile_path={str(profile)!r},hotkey_config_path={str(hotkey)!r},role='gateway')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program], cwd=ROOT,
        env=environment, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "gateway.main" not in sys.modules


def test_both_temporary_entrypoints_configure_before_chain_consumers_import():
    gateway_source = Path(launcher.__file__).read_text(encoding="utf-8")
    validator_source = Path(validator_launcher.__file__).read_text(encoding="utf-8")
    assert gateway_source.index("configure_temporary_testnet401_chain_source(") < (
        gateway_source.index('runpy.run_module("gateway.main"')
    )
    assert validator_source.index("configure_temporary_testnet401_chain_source(") < (
        validator_source.index("from neurons import validator")
    )
    assert "validator.submit_weights_at_epoch_end()" in validator_source
    assert ".set_weights(" not in validator_source
