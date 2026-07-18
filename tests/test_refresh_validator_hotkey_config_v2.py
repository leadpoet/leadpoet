from __future__ import annotations

import json
from pathlib import Path

from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    load_chain_signing_profile,
)
from validator_tee.host.refresh_hotkey_config_v2 import (
    refresh_hotkey_config_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def test_refresh_preserves_identity_and_rebinds_public_measurements(tmp_path):
    config = tmp_path / "validator-hotkey-config-v2.json"
    profile = ROOT / "validator_tee/enclave/chain_signing_profile_v2.json"
    drand_hash = tmp_path / "libbittensor_drand_v2.sha256"
    identity = {
        "validator_hotkey": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
        "hotkey_public_key": "92" * 32,
    }
    config.write_text(
        json.dumps(
            {
                "schema_version": "leadpoet.validator_hotkey_config.v2",
                **identity,
                "chain_signing_profile_hash": "sha256:" + "0" * 64,
                "drand_library_path": (
                    "/app/validator_tee/enclave/libbittensor_drand_v2.so"
                ),
                "drand_library_sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    drand_hash.write_text("2" * 64 + "\n", encoding="ascii")

    result = refresh_hotkey_config_v2(
        config_path=config,
        chain_profile_path=profile,
        drand_hash_path=drand_hash,
    )
    refreshed = json.loads(config.read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert {name: refreshed[name] for name in identity} == identity
    assert refreshed["chain_signing_profile_hash"] == sha256_json(
        load_chain_signing_profile(profile)
    )
    assert refreshed["drand_library_sha256"] == "2" * 64
    assert config.stat().st_mode & 0o777 == 0o600

    second = refresh_hotkey_config_v2(
        config_path=config,
        chain_profile_path=profile,
        drand_hash_path=drand_hash,
    )
    assert second["status"] == "unchanged"
