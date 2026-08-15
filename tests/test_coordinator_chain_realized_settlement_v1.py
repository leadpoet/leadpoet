from __future__ import annotations

import pytest

from gateway.tee import coordinator_chain_realized_settlement_v1 as authority
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import sha256_json


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


class _Reader:
    def __init__(self, rows_by_policy):
        self.rows_by_policy = rows_by_policy
        self.calls = []

    def read(self, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.rows_by_policy.get(kwargs["policy_id"], []))


class _Chain:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []

    def read_stateful_epoch_close_weights(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.observation)


def _context(*, graphs=()):
    return ExecutionContextV2(
        job_id="chain-realized:101",
        purpose=authority.CHAIN_REALIZED_SETTLEMENT_PURPOSE_V1,
        epoch_id=101,
        external_receipt_graphs=tuple(graphs),
    )


def _chain_state():
    return {
        "official_subnet_epoch_id": 101,
        "cutover_mapping_hash": HASH_A,
        "close_block": 1_359,
        "close_block_hash": "1" * 64,
        "close_header": {"state_root": "2" * 64},
        "next_epoch_block": 1_360,
        "next_epoch_block_hash": "3" * 64,
        "validator_hotkey": "validator-hotkey",
        "validator_uid": 0,
        "metagraph_hotkeys": ["validator-hotkey", "miner-hotkey"],
        "weights": [[0, 65_535], [1, 3_449]],
        "weights_storage_key": "0x1234",
        "last_update_storage_key": "0x5678",
        "last_update_block": 1_345,
        "last_update_block_hash": "4" * 64,
        "last_update_official_subnet_epoch_id": 100,
        "active_source_epoch_id": 100,
    }


def _observation():
    state = _chain_state()
    close_header = state.pop("close_header")
    return {
        "schema_version": authority.CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
        "netuid": 71,
        "epoch_id": 101,
        **state,
        "close_state_root": close_header["state_root"],
        "weights_vector_hash": sha256_json(
            {
                "uids": [item[0] for item in state["weights"]],
                "weights_u16": [item[1] for item in state["weights"]],
            }
        ),
    }


def _authority_summary(
    *,
    bundle_hash=HASH_B,
    validator_hotkey="validator-hotkey",
    finalized_block=1_000,
):
    return {
        "bundle_hash": bundle_hash,
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": validator_hotkey,
        "finalized_block": finalized_block,
        "finalized_block_hash": "5" * 64,
        "finalization_receipt_hash": HASH_C,
    }


def test_observation_uses_latest_finalized_primary_identity(monkeypatch):
    old_summary = _authority_summary(
        bundle_hash=HASH_A,
        validator_hotkey="old",
        finalized_block=900,
    )
    latest_summary = _authority_summary()
    reader = _Reader(
        {
            "latest_finalized_allocation_authority_summaries": [
                latest_summary,
                old_summary,
            ],
            "finalized_allocation_authority_by_bundle_hash": [
                {**latest_summary, "full": True}
            ],
        }
    )
    chain = _Chain(_chain_state())
    monkeypatch.setattr(
        authority,
        "_preliminary_finalized_bundle_authority_v1",
        lambda row: dict(row),
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=chain,
    )

    observed = source.observe(
        payload={
            "schema_version": (
                authority.CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
        },
        context=_context(),
    )

    assert observed == _observation()
    assert chain.calls[0]["validator_hotkey"] == "validator-hotkey"
    assert reader.calls[0]["policy_id"] == (
        "latest_compact_finalized_authority_summaries"
    )
    assert reader.calls[1]["policy_id"] == (
        "latest_finalized_allocation_authority_summaries"
    )
    assert reader.calls[2]["policy_id"] == (
        "finalized_allocation_authority_by_bundle_hash"
    )
    assert reader.calls[2]["parameters"] == {
        "netuid": 71,
        "bundle_hash": HASH_B,
    }


def test_observation_rejects_ambiguous_latest_primary_identity(monkeypatch):
    reader = _Reader(
        {
            "latest_finalized_allocation_authority_summaries": [
                _authority_summary(validator_hotkey="first"),
                _authority_summary(
                    bundle_hash=HASH_A,
                    validator_hotkey="second",
                ),
            ]
        }
    )
    monkeypatch.setattr(
        authority,
        "_preliminary_finalized_bundle_authority_v1",
        lambda row: dict(row),
    )
    chain = _Chain(_chain_state())
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=chain,
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="identity is ambiguous",
    ):
        source.observe(
            payload={
                "schema_version": (
                    authority.CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
            },
            context=_context(),
        )


def test_observation_rejects_full_authority_that_differs_from_summary(
    monkeypatch,
):
    summary = _authority_summary()
    reader = _Reader(
        {
            "latest_finalized_allocation_authority_summaries": [summary],
            "finalized_allocation_authority_by_bundle_hash": [
                {**summary, "full": True}
            ],
        }
    )
    monkeypatch.setattr(
        authority,
        "_preliminary_finalized_bundle_authority_v1",
        lambda row: {**dict(row), "validator_hotkey": "substituted"},
    )
    chain = _Chain(_chain_state())
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=chain,
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="summary differs at validator_hotkey",
    ):
        source.observe(
            payload={
                "schema_version": (
                    authority.CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
            },
            context=_context(),
        )
    assert chain.calls == []


def test_observation_rejects_malformed_authority_summary():
    malformed = _authority_summary()
    malformed["bundle_hash"] = HASH_B + "&select=secret"
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=_Reader(
            {
                "latest_finalized_allocation_authority_summaries": [
                    malformed
                ]
            }
        ),
        chain_source=_Chain(_chain_state()),
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="summary is invalid",
    ):
        source.observe(
            payload={
                "schema_version": (
                    authority.CHAIN_WEIGHT_OBSERVATION_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
            },
            context=_context(),
        )


def test_settlement_rereads_exact_vector_and_rejects_host_substitution(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    finalization_receipt = {"receipt_hash": HASH_C}
    graphs = (
        {"root_receipt_hash": HASH_A, "receipts": [observation_receipt]},
        {"root_receipt_hash": HASH_C, "receipts": [finalization_receipt]},
    )
    reader = _Reader(
        {
            "finalized_authority_by_chain_vector": [
                {"candidate": True, "bundle_hash": HASH_B}
            ]
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, **_kwargs: {
            "bundle_hash": HASH_B,
            "finalization_receipt_hash": HASH_C,
        },
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="host-selected",
    ):
        source.settle(
            payload={
                "schema_version": (
                    authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
                "observation": observation,
                "observation_receipt_hash": HASH_A,
                "authority_mode": "finalized_bundle",
                "bundle_hash": "sha256:" + "d" * 64,
            },
            context=_context(graphs=graphs),
        )

    assert reader.calls[0]["policy_id"] == (
        "compact_finalized_authority_cutover"
    )
    filters = reader.calls[1]["parameters"]
    assert filters["source_epoch_id"] == 100
    assert filters["validator_hotkey"] == "validator-hotkey"
    assert filters["finalized_block"] == 1_345
    assert filters["finalized_block_hash"] == "4" * 64
    assert filters["uids"] == [0, 1]
    assert filters["weights_u16"] == [65_535, 3_449]


def test_settlement_accepts_only_independently_verified_exact_authority(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    finalization_receipt = {"receipt_hash": HASH_C}
    graphs = [
        {"root_receipt_hash": HASH_A, "receipts": [observation_receipt]},
        {"root_receipt_hash": HASH_C, "receipts": [finalization_receipt]},
    ]
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
    }
    package = {
        "settlement_doc": {"epoch_id": 101},
        "settlement_hash": "sha256:" + "d" * 64,
        "credits": [],
    }
    reader = _Reader(
        {
            "finalized_authority_by_chain_vector": [
                {"candidate": True, "bundle_hash": HASH_B}
            ]
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        authority,
        "validate_finalized_allocation_authorities_v2",
        lambda rows, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        authority,
        "build_chain_realized_settlement_package_v1",
        lambda **kwargs: package if kwargs["authority"] == candidate else None,
    )

    result = source.settle(
        payload={
            "schema_version": (
                authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
            "observation": observation,
            "observation_receipt_hash": HASH_A,
            "authority_mode": "finalized_bundle",
            "bundle_hash": HASH_B,
        },
        context=_context(graphs=graphs),
    )

    assert result == package


def test_settlement_records_missing_finalized_authority_without_credit(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    reader = _Reader({"finalized_authority_by_chain_vector": []})
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    result = source.settle(
        payload={
            "schema_version": (
                authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
            "observation": observation,
            "observation_receipt_hash": HASH_A,
            "authority_mode": "unattributed",
            "bundle_hash": None,
        },
        context=_context(
            graphs=(
                {
                    "root_receipt_hash": HASH_A,
                    "receipts": [observation_receipt],
                },
            )
        ),
    )

    assert result["credits"] == []
    assert result["settlement_doc"]["credit_hashes"] == []
    assert result["settlement_doc"]["schema_version"] == (
        "leadpoet.research_lab_chain_realized_epoch_settlement.v2"
    )
    assert result["settlement_doc"]["observation_summary"][
        "authority_mode"
    ] == "unattributed_chain_observation"


def test_unattributed_settlement_rejects_existing_finalized_authority(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    reader = _Reader(
        {
            "finalized_authority_by_chain_vector": [
                {"candidate": True, "bundle_hash": HASH_B}
            ]
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_chain_realized_bundle_candidate_v1",
        lambda _rows, **_kwargs: {
            "bundle_hash": HASH_B,
            "finalization_receipt_hash": HASH_C,
        },
    )

    with pytest.raises(
        authority.CoordinatorChainRealizedSettlementV1Error,
        match="has finalized bundle authority",
    ):
        source.settle(
            payload={
                "schema_version": (
                    authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
                ),
                "netuid": 71,
                "epoch_id": 101,
                "observation": observation,
                "observation_receipt_hash": HASH_A,
                "authority_mode": "unattributed",
                "bundle_hash": None,
            },
            context=_context(
                graphs=(
                    {
                        "root_receipt_hash": HASH_A,
                        "receipts": [observation_receipt],
                    },
                )
            ),
        )


def test_post_cutover_settlement_uses_only_exact_compact_authority(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    finalization_receipt = {"receipt_hash": HASH_C}
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": "validator-hotkey",
        "finalized_block": 1_345,
        "finalized_block_hash": "4" * 64,
    }
    verified = dict(candidate)
    package = {
        "settlement_doc": {"epoch_id": 101},
        "settlement_hash": "sha256:" + "d" * 64,
        "credits": [],
    }
    compact_row = {"bundle_hash": HASH_B, "compact": True}
    reader = _Reader(
        {
            "compact_finalized_authority_cutover": [{"epoch_id": 100}],
            "compact_finalized_authority_by_identity": [compact_row],
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
        expected_lineage_id=HASH_A,
        expected_chain="wss://chain.example:443",
        boot_verifier=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_compact_chain_realized_bundle_candidate_v2",
        lambda rows, **_kwargs: candidate if rows == [compact_row] else None,
    )
    monkeypatch.setattr(
        authority.CoordinatorChainRealizedSettlementV1,
        "_verify_compact_authority",
        lambda _self, row: (
            (candidate, verified) if row == compact_row else (None, None)
        ),
    )
    monkeypatch.setattr(
        authority,
        "build_compact_chain_realized_settlement_package_v2",
        lambda **kwargs: (
            package
            if kwargs == {
                "observation": observation,
                "authority": candidate,
                "verified": verified,
            }
            else None
        ),
    )

    result = source.settle(
        payload={
            "schema_version": (
                authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
            "observation": observation,
            "observation_receipt_hash": HASH_A,
            "authority_mode": "finalized_bundle",
            "bundle_hash": HASH_B,
        },
        context=_context(
            graphs=(
                {
                    "root_receipt_hash": HASH_A,
                    "receipts": [observation_receipt],
                },
                {
                    "root_receipt_hash": HASH_C,
                    "receipts": [finalization_receipt],
                },
            )
        ),
    )

    assert result == package
    assert [call["policy_id"] for call in reader.calls] == [
        "compact_finalized_authority_cutover",
        "compact_finalized_authority_by_identity",
    ]


def test_post_cutover_settlement_resolves_checkpointed_parent_authority(
    monkeypatch,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    finalization_receipt = {"receipt_hash": HASH_C}
    observation_graph = {
        "root_receipt_hash": HASH_A,
        "receipts": [observation_receipt],
    }
    checkpoint_authority_graph = {
        "root_receipt_hash": HASH_C,
        "receipts": [finalization_receipt],
    }
    candidate = {
        "bundle_hash": HASH_B,
        "finalization_receipt_hash": HASH_C,
        "netuid": 71,
        "epoch_id": 100,
        "validator_hotkey": "validator-hotkey",
        "finalized_block": 1_345,
        "finalized_block_hash": "4" * 64,
    }
    package = {
        "settlement_doc": {"epoch_id": 101},
        "settlement_hash": "sha256:" + "d" * 64,
        "credits": [],
    }
    reader = _Reader(
        {
            "compact_finalized_authority_cutover": [{"epoch_id": 100}],
            "compact_finalized_authority_by_identity": [
                {"bundle_hash": HASH_B, "compact": True}
            ],
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
        expected_lineage_id=HASH_A,
        expected_chain="wss://chain.example:443",
        boot_verifier=lambda *_args, **_kwargs: None,
    )
    context = _context(graphs=(observation_graph,))
    context.external_receipt_authority_graphs = lambda: (
        observation_graph,
        checkpoint_authority_graph,
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "select_compact_chain_realized_bundle_candidate_v2",
        lambda _rows, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        authority.CoordinatorChainRealizedSettlementV1,
        "_verify_compact_authority",
        lambda _self, _row: (candidate, dict(candidate)),
    )
    monkeypatch.setattr(
        authority,
        "build_compact_chain_realized_settlement_package_v2",
        lambda **_kwargs: package,
    )

    result = source.settle(
        payload={
            "schema_version": (
                authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
            ),
            "netuid": 71,
            "epoch_id": 101,
            "observation": observation,
            "observation_receipt_hash": HASH_A,
            "authority_mode": "finalized_bundle",
            "bundle_hash": HASH_B,
        },
        context=context,
    )

    assert result == package
    assert context.external_receipt_graphs == (observation_graph,)


def test_compact_authority_verifier_uses_exact_binding_endpoint(monkeypatch):
    observed = {}
    preliminary = {"authority_doc": {"authority": True}}
    verified = {"bundle_hash": HASH_B}
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=_Reader({}),
        chain_source=_Chain(_chain_state()),
        expected_lineage_id=HASH_A,
        expected_chain="wss://entrypoint-finney.opentensor.ai:443",
        boot_verifier=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        authority,
        "_preliminary_compact_finalized_bundle_authority_v2",
        lambda row: preliminary if row == {"row": True} else None,
    )

    def verify(document, **kwargs):
        observed["document"] = document
        observed.update(kwargs)
        return verified

    monkeypatch.setattr(
        authority,
        "verify_compact_published_weight_authority_v2",
        verify,
    )

    assert source._verify_compact_authority({"row": True}) == (
        preliminary,
        verified,
    )
    assert observed["document"] == preliminary["authority_doc"]
    assert observed["expected_lineage_id"] == HASH_A
    assert observed["expected_chain"] == (
        "wss://entrypoint-finney.opentensor.ai:443"
    )
    assert observed["chain_signing_profile"] is None
    assert callable(observed["boot_verifier"])


@pytest.mark.parametrize("authority_mode", ["unattributed", "finalized_bundle"])
def test_post_cutover_missing_compact_authority_never_falls_back_to_legacy(
    monkeypatch,
    authority_mode,
):
    observation = _observation()
    observation_receipt = {
        "receipt_hash": HASH_A,
        "role": "gateway_coordinator",
        "purpose": authority.CHAIN_WEIGHT_OBSERVATION_PURPOSE_V1,
        "status": "succeeded",
        "epoch_id": 101,
        "output_root": sha256_json(observation),
    }
    reader = _Reader(
        {
            "compact_finalized_authority_cutover": [{"epoch_id": 100}],
            "compact_finalized_authority_by_identity": [],
            "finalized_authority_by_chain_vector": [
                {"bundle_hash": HASH_B, "legacy": True}
            ],
        }
    )
    source = authority.CoordinatorChainRealizedSettlementV1(
        reader=reader,
        chain_source=_Chain(_chain_state()),
        expected_lineage_id=HASH_A,
        expected_chain="wss://chain.example:443",
        boot_verifier=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )

    payload = {
        "schema_version": (
            authority.CHAIN_REALIZED_SETTLEMENT_REQUEST_SCHEMA_VERSION_V1
        ),
        "netuid": 71,
        "epoch_id": 101,
        "observation": observation,
        "observation_receipt_hash": HASH_A,
        "authority_mode": authority_mode,
        "bundle_hash": None,
    }
    if authority_mode == "unattributed":
        result = source.settle(
            payload=payload,
            context=_context(
                graphs=(
                    {
                        "root_receipt_hash": HASH_A,
                        "receipts": [observation_receipt],
                    },
                )
            ),
        )
        assert result["credits"] == []
    else:
        with pytest.raises(
            authority.CoordinatorChainRealizedSettlementV1Error,
            match="no finalized canonical bundle",
        ):
            source.settle(
                payload=payload,
                context=_context(
                    graphs=(
                        {
                            "root_receipt_hash": HASH_A,
                            "receipts": [observation_receipt],
                        },
                    )
                ),
            )

    assert [call["policy_id"] for call in reader.calls] == [
        "compact_finalized_authority_cutover",
        "compact_finalized_authority_by_identity",
    ]
