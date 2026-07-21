"""The repo ships the activated SN71 cutover manifest for auditors.

External audit validators run from a plain repository checkout with no
operator provisioning step. They must be able to construct the official
stateful epoch authority from the repository alone, while explicit operator
configuration always takes precedence and other netuids keep failing closed.
"""

from __future__ import annotations

import json

from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    CUTOVER_PATH_ENV,
    DEFAULT_SN71_CUTOVER_MANIFEST_PATH,
    ensure_cutover_manifest_configured,
    load_subnet_epoch_cutover,
)


# The activated production mapping. Changing the shipped manifest without
# an intentional, receipt-backed re-cutover must fail review loudly.
ACTIVATED_SN71_MAPPING_HASH = (
    "sha256:58d9048c3c96c8f82ccd6e2c2445ed497539022f5e539b2b4af6ddb0f5dd50c2"
)


def test_repo_manifest_exists_and_is_the_activated_sn71_mapping():
    assert DEFAULT_SN71_CUTOVER_MANIFEST_PATH.is_file()
    document = json.loads(
        DEFAULT_SN71_CUTOVER_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert document["netuid"] == 71
    assert document["mapping_hash"] == ACTIVATED_SN71_MAPPING_HASH

    cutover = load_subnet_epoch_cutover(
        {CUTOVER_PATH_ENV: str(DEFAULT_SN71_CUTOVER_MANIFEST_PATH)}
    )
    assert cutover.netuid == 71
    assert cutover.mapping_hash == ACTIVATED_SN71_MAPPING_HASH
    assert cutover.first_settlement_epoch_id == 24073
    assert cutover.last_legacy_epoch_id == 24072
    assert cutover.first_subnet_epoch_index == 24020


def test_default_applies_only_when_unset():
    environ: dict = {}
    ensure_cutover_manifest_configured(environ)
    assert environ[CUTOVER_PATH_ENV] == str(DEFAULT_SN71_CUTOVER_MANIFEST_PATH)

    explicit_path = {CUTOVER_PATH_ENV: "/secure/operator/custom.json"}
    ensure_cutover_manifest_configured(explicit_path)
    assert explicit_path[CUTOVER_PATH_ENV] == "/secure/operator/custom.json"
    assert CUTOVER_JSON_ENV not in explicit_path

    explicit_json = {CUTOVER_JSON_ENV: '{"netuid": 71}'}
    ensure_cutover_manifest_configured(explicit_json)
    assert CUTOVER_PATH_ENV not in explicit_json


def test_defaulted_environment_loads_the_activated_mapping():
    environ: dict = {}
    ensure_cutover_manifest_configured(environ)
    cutover = load_subnet_epoch_cutover(environ)
    assert cutover.mapping_hash == ACTIVATED_SN71_MAPPING_HASH
