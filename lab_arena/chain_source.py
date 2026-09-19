"""Finalized Bittensor state verification for the local Arena validator."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_MAX_FINALIZATION_SCAN_BLOCKS,
    CHAIN_RPC_METHOD,
    CHAIN_SUBTENSOR_MAX_TEMPO,
    ChainSourceV2Error,
    decode_timestamp_now_storage,
    decode_runtime_metadata_commitment,
    decode_timelocked_weight_commits,
    decode_selective_metagraph_result,
    encode_selective_metagraph_params,
    normalize_raw_hash,
    parse_finalized_block_extrinsics,
    parse_finalized_header,
    parse_runtime_version,
    decode_subnet_epoch_storage,
    decode_system_account_nonce,
    decode_last_update_storage,
    decode_weights_storage,
    subnet_epoch_storage_key,
    system_account_storage_key,
    system_event_count_storage_key,
    system_events_storage_key,
    last_update_storage_key,
    weights_storage_key,
    timestamp_now_storage_key,
    timelocked_weight_commits_storage_key,
)
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2
from leadpoet_canonical.subtensor_events_v2 import (
    RUNTIME_CODE_STORAGE_KEY,
    load_subtensor_events_profile_v2,
    prove_timelocked_weights_reveal_v2,
    validate_subtensor_events_profile_v2,
)


FINALIZATION_RPC_PACING_SECONDS = 1.05


class ValidatorChainSourceV2Error(RuntimeError):
    """The validator could not authenticate a complete chain snapshot."""


def validate_rewarded_uid_ownership(
    hotkeys: Sequence[str], bindings: Sequence[Mapping[str, Any]]
) -> None:
    """Require every signed rewarded UID to retain its finalized owner."""
    normalized = {int(item["uid"]): str(item["hotkey"]) for item in bindings}
    if len(normalized) != len(bindings):
        raise ValidatorChainSourceV2Error("signed recipient UID bindings are not unique")
    for uid, expected_hotkey in normalized.items():
        if uid < 0 or uid >= len(hotkeys) or hotkeys[uid] != expected_hotkey:
            raise ValidatorChainSourceV2Error(
                "rewarded UID ownership changed before reveal"
            )


def _chain_timestamp(timestamp_ms: int) -> str:
    value = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


class ValidatorChainSourceV2:
    def read_finalized_head(self) -> Dict[str, Any]:
        finalized = self._call(method='chain_getFinalizedHead', params=[], request_id=1)
        block_hash = normalize_raw_hash(finalized, "finalized head")
        header = self._call(method='chain_getHeader', params=['0x' + block_hash], request_id=2)
        parsed = parse_finalized_header(header)
        return {"block": int(parsed["block"]), "block_hash": block_hash}

    def __init__(
        self,
        *,
        rpc_call: Callable[..., Any],
        archive_rpc_call: Optional[Callable[..., Any]] = None,
        finalization_sleep: Callable[[float], None] = time.sleep,
        epoch_authority_supplier: Optional[
            Callable[[], Optional[Mapping[str, Any]]]
        ] = None,
    ) -> None:
        if not callable(rpc_call):
            raise ValidatorChainSourceV2Error("chain RPC adapter is required")
        if archive_rpc_call is not None and not callable(archive_rpc_call):
            raise ValidatorChainSourceV2Error("archive RPC adapter is invalid")
        self._rpc_call = rpc_call
        self._archive_rpc_call = archive_rpc_call
        self._finalization_sleep = finalization_sleep
        if epoch_authority_supplier is None:
            raise ValidatorChainSourceV2Error(
                "official SN71 epoch authority supplier is unavailable"
            )
        self._epoch_authority_supplier = epoch_authority_supplier

    def read_chain_signing_runtime(
        self,
        *,
        runtime_block_hash: str,
        max_block_drift: int,
    ) -> Dict[str, Any]:
        """Read one canonical finalized runtime through the configured host RPC."""

        requested_hash = normalize_raw_hash(
            runtime_block_hash, "runtime block hash"
        )
        normalized_drift = int(max_block_drift)
        if not 1 <= normalized_drift <= 1024:
            raise ValidatorChainSourceV2Error(
                "runtime block drift policy is invalid"
            )
        request_id = 1

        def invoke(method: str, params: Sequence[Any]) -> Any:
            nonlocal request_id
            result = self._call(method=method, params=params, request_id=request_id)
            request_id += 1
            return result

        finalized_hash = normalize_raw_hash(
            invoke('chain_getFinalizedHead', []),
            "finalized head",
        )
        finalized_header = parse_finalized_header(
            invoke('chain_getHeader', ['0x' + finalized_hash])
        )
        requested_header = parse_finalized_header(
            invoke('chain_getHeader', ['0x' + requested_hash])
        )
        finalized_block = int(finalized_header["block"])
        runtime_block = int(requested_header["block"])
        if (
            runtime_block > finalized_block
            or finalized_block - runtime_block > normalized_drift
        ):
            raise ValidatorChainSourceV2Error(
                "runtime block is not within the finalized signing window"
            )
        canonical_hash = normalize_raw_hash(
            invoke('chain_getBlockHash', [runtime_block]),
            "canonical runtime block hash",
        )
        if canonical_hash != requested_hash:
            raise ValidatorChainSourceV2Error(
                "runtime block is not canonical at its exact height"
            )
        try:
            version = parse_runtime_version(
                invoke('state_getRuntimeVersion', ['0x' + requested_hash])
            )
        except ChainSourceV2Error as exc:
            raise ValidatorChainSourceV2Error(
                "runtime version response is invalid"
            ) from exc
        genesis_hash = normalize_raw_hash(
            invoke('chain_getBlockHash', [0]),
            "genesis block hash",
        )
        return {
            "runtime_block": runtime_block,
            "runtime_block_hash": requested_hash,
            "finalized_block": finalized_block,
            "finalized_block_hash": finalized_hash,
            "spec_version": version["spec_version"],
            "transaction_version": version["transaction_version"],
            "genesis_hash": genesis_hash,
        }

    def read_finalized_account_nonce(
        self, *, account_public_key_hex: str, finalized_block_hash: str
    ) -> int:
        """Read nonce from authenticated System.Account at one finalized hash."""

        try:
            account = bytes.fromhex(str(account_public_key_hex or ""))
        except ValueError as exc:
            raise ValidatorChainSourceV2Error("account public key is invalid") from exc
        target = normalize_raw_hash(finalized_block_hash, "finalized block hash")
        result = self._call(method='state_getStorage', params=[system_account_storage_key(account), '0x' + target], request_id=1)
        try:
            return decode_system_account_nonce(result)
        except ChainSourceV2Error as exc:
            raise ValidatorChainSourceV2Error("finalized account nonce is invalid") from exc

    def read_canonical_block_hash(self, *, block: int) -> str:
        result = self._call(method='chain_getBlockHash', params=[int(block)], request_id=1)
        return normalize_raw_hash(result, "canonical block hash")


    def prove_timelocked_reveal_transition(
        self, *, netuid: int, validator_hotkey: str,
        hotkey_public_key_hex: str, subnet_epoch_index: int,
        commitment_hex: str, reveal_round: int, inclusion_block: int,
        reveal_deadline_block: int, expected_weights: Sequence[Tuple[int, int]],
        expected_recipient_uid_hotkeys: Sequence[Mapping[str, Any]],
        chain_profile: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Prove the first present-to-absent commit transition and its weights.

        The storage transition is read at exact finalized historical blocks.
        LastUpdate records each timelocked commit and is deliberately not
        changed when commit-reveal applies the weights. Bind it to this commit
        at its inclusion block; a later valid commit may advance it before this
        commitment is revealed.
        """

        if self._archive_rpc_call is None:
            raise ValidatorChainSourceV2Error("archive RPC is required for Arena reveal proof")
        head = self.read_finalized_head()
        start = int(inclusion_block)
        end = min(int(reveal_deadline_block), int(head["block"]))
        if end <= start:
            return None
        request_id = 1
        def archive(method: str, params: Sequence[Any]) -> Any:
            nonlocal request_id
            result = self._archive_call(method=method, params=list(params), request_id=request_id)
            request_id += 1
            return result

        commit_key = timelocked_weight_commits_storage_key(
            netuid=int(netuid), subnet_epoch_index=int(subnet_epoch_index)
        )

        def block_hash(number: int) -> str:
            return normalize_raw_hash(
                archive('chain_getBlockHash', [int(number)]),
                "reveal proof block hash",
            )

        def is_present(number: int) -> bool:
            digest = block_hash(number)
            commits = decode_timelocked_weight_commits(archive('state_getStorage', [commit_key, '0x' + digest]))
            return any(
                item["hotkey_public_key"] == str(hotkey_public_key_hex)
                and item["commitment_hex"] == str(commitment_hex)
                and int(item["reveal_round"]) == int(reveal_round)
                for item in commits
            )

        if not is_present(start):
            raise ValidatorChainSourceV2Error("Arena commitment is absent at inclusion block")
        inclusion_digest = block_hash(start)
        inclusion_metagraph = decode_selective_metagraph_result(archive('state_call', [CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=int(netuid)), '0x' + inclusion_digest]))
        inclusion_matches = [
            uid for uid, hotkey in enumerate(inclusion_metagraph["hotkeys"])
            if hotkey == validator_hotkey
        ]
        if len(inclusion_matches) != 1:
            raise ValidatorChainSourceV2Error(
                "validator hotkey has no unique UID at commitment inclusion"
            )
        inclusion_uid = inclusion_matches[0]
        inclusion_updates = list(decode_last_update_storage(archive('state_getStorage', [last_update_storage_key(netuid=int(netuid)), '0x' + inclusion_digest])))
        if (inclusion_uid >= len(inclusion_updates)
                or int(inclusion_updates[inclusion_uid]) != start):
            raise ValidatorChainSourceV2Error(
                "LastUpdate does not bind the Arena commitment inclusion"
            )
        if is_present(end):
            return None
        low, high = start, end
        while high - low > 1:
            middle = (low + high) // 2
            if is_present(middle):
                low = middle
            else:
                high = middle
        transition = high
        digest = block_hash(transition)
        metagraph = decode_selective_metagraph_result(archive('state_call', [CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=int(netuid)), '0x' + digest]))
        matches = [uid for uid, hotkey in enumerate(metagraph["hotkeys"]) if hotkey == validator_hotkey]
        if len(matches) != 1:
            raise ValidatorChainSourceV2Error("validator hotkey has no unique UID at reveal transition")
        uid = matches[0]
        if uid != inclusion_uid:
            raise ValidatorChainSourceV2Error(
                "validator UID changed between commitment and reveal"
            )
        validate_rewarded_uid_ownership(
            metagraph["hotkeys"], expected_recipient_uid_hotkeys
        )
        weights = list(decode_weights_storage(archive('state_getStorage', [weights_storage_key(netuid=int(netuid), validator_uid=uid), '0x' + digest])))
        updates = list(decode_last_update_storage(archive('state_getStorage', [last_update_storage_key(netuid=int(netuid)), '0x' + digest])))
        if (uid >= len(updates)
                or not int(inclusion_block) <= int(updates[uid]) <= transition):
            raise ValidatorChainSourceV2Error(
                "LastUpdate is outside the proved commit-reveal interval"
            )
        if weights != [(int(uid_), int(weight)) for uid_, weight in expected_weights]:
            raise ValidatorChainSourceV2Error("revealed weights differ at commit transition")

        # A queue entry is removed even when automatic decryption or weight
        # application fails. Therefore disappearance alone is not a reveal
        # proof. Require the runtime's exact successful initialization event.
        pre_transition_hash = block_hash(transition - 1)
        runtime = parse_runtime_version(archive('state_getRuntimeVersion', ['0x' + pre_transition_hash]))
        metadata_value = archive('state_getMetadata', ['0x' + pre_transition_hash])
        if not isinstance(metadata_value, str) or not metadata_value.startswith("0x"):
            raise ValidatorChainSourceV2Error("reveal runtime metadata is absent")
        metadata_raw = bytes.fromhex(metadata_value[2:])
        decode_runtime_metadata_commitment(metadata_value)
        runtime_code_hash = normalize_raw_hash(archive('state_getStorageHash', [RUNTIME_CODE_STORAGE_KEY, '0x' + pre_transition_hash]), "reveal runtime code hash")
        event_profile = validate_subtensor_events_profile_v2(
            load_subtensor_events_profile_v2(
                spec_version=int(runtime["spec_version"])
            ),
            genesis_hash=str(chain_profile["genesis_hash"]),
            spec_version=int(runtime["spec_version"]),
            transaction_version=int(runtime["transaction_version"]),
            metadata_raw=metadata_raw,
            runtime_code_hash="0x" + runtime_code_hash,
        )
        events_value = archive('state_getStorage', [system_events_storage_key(), '0x' + digest])
        event_count_value = archive('state_getStorage', [system_event_count_storage_key(), '0x' + digest])
        if (not isinstance(events_value, str) or not events_value.startswith("0x")
                or not isinstance(event_count_value, str)
                or not event_count_value.startswith("0x")):
            raise ValidatorChainSourceV2Error("reveal event storage is absent")
        event_witness = prove_timelocked_weights_reveal_v2(
            bytes.fromhex(events_value[2:]), profile=event_profile,
            event_count_raw=bytes.fromhex(event_count_value[2:]),
            expected_netuid=int(netuid), expected_uid=int(uid),
            expected_account_id_hex=str(hotkey_public_key_hex),
        )
        return {
            "reveal_block": transition, "reveal_block_hash": digest,
            "validator_uid": uid, "last_update": int(updates[uid]),
            "weights": weights, "event_witness": event_witness,
            "transition_hash": sha256_json({
                "commitment_hex": str(commitment_hex), "present_at": low,
                "absent_at": transition, "weights": [list(item) for item in weights],
                "last_update": int(updates[uid]),
                "event_witness": event_witness,
                "recipient_uid_hotkeys": list(expected_recipient_uid_hotkeys),
            }),
        }

    def _read_stateful_epoch_authority(
        self,
        *,
        configuration: Mapping[str, Any],
        finalized_hash: str,
        header: Mapping[str, Any],
        netuid: int,
        settlement_epoch_id: int,

        request_id_start: int,


        historical_snapshot: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(configuration, Mapping) or set(configuration) != {
            "mode",
            "cutover_manifest",
        }:
            raise ValidatorChainSourceV2Error(
                "measured epoch authority configuration is invalid"
            )
        if configuration.get("mode") != "stateful_v1":
            raise ValidatorChainSourceV2Error("measured epoch mode is invalid")
        cutover = configuration.get("cutover_manifest")
        cutover_fields = {
            "schema_version",
            "epoch_scheme",
            "network_genesis_hash",
            "netuid",
            "cutover_block",
            "cutover_block_hash",
            "first_subnet_epoch_index",
            "first_settlement_epoch_id",
            "last_legacy_epoch_id",
            "mapping_hash",
        }
        if not isinstance(cutover, Mapping) or set(cutover) != cutover_fields:
            raise ValidatorChainSourceV2Error("measured epoch cutover is absent")
        if (
            cutover.get("schema_version") != "leadpoet.subnet_epoch_cutover.v1"
            or cutover.get("epoch_scheme") != "bittensor.subnet_epoch_index.v1"
        ):
            raise ValidatorChainSourceV2Error("measured epoch cutover schema is invalid")
        for field in (
            "netuid",
            "cutover_block",
            "first_subnet_epoch_index",
            "first_settlement_epoch_id",
            "last_legacy_epoch_id",
        ):
            if (
                not isinstance(cutover.get(field), int)
                or isinstance(cutover.get(field), bool)
                or int(cutover[field]) < 0
            ):
                raise ValidatorChainSourceV2Error(
                    "measured epoch cutover %s is invalid" % field
                )
        try:
            normalize_raw_hash(cutover["network_genesis_hash"], "network genesis")
            normalize_raw_hash(cutover["cutover_block_hash"], "cutover block hash")
        except ChainSourceV2Error as exc:
            raise ValidatorChainSourceV2Error(
                "measured epoch cutover hash is invalid"
            ) from exc
        cutover_body = {
            field: cutover[field]
            for field in (
                "schema_version",
                "epoch_scheme",
                "network_genesis_hash",
                "netuid",
                "cutover_block",
                "cutover_block_hash",
                "first_subnet_epoch_index",
                "first_settlement_epoch_id",
                "last_legacy_epoch_id",
            )
        }
        if (
            int(cutover["first_settlement_epoch_id"])
            != int(cutover["last_legacy_epoch_id"]) + 1
            or str(cutover.get("mapping_hash") or "").lower()
            != sha256_json(cutover_body)
        ):
            raise ValidatorChainSourceV2Error("measured epoch cutover mapping is invalid")
        if int(cutover.get("netuid", -1)) != int(netuid):
            raise ValidatorChainSourceV2Error("measured epoch cutover netuid differs")
        request_id = int(request_id_start)

        def invoke(
            adapter: Callable[..., Dict[str, Any]],
            method: str,
            params: Sequence[Any],

        ) -> Any:
            nonlocal request_id
            result = adapter(method=method, params=params, request_id=request_id)
            request_id += 1
            return result

        def live_call(method: str, params: Sequence[Any]) -> Any:
            return invoke(self._call, method, params)

        def archive_call(
            method: str,
            params: Sequence[Any],

        ) -> Any:
            return invoke(self._archive_call, method, params)

        genesis_hash = normalize_raw_hash(
            archive_call('chain_getBlockHash', [0]),
            "genesis block hash",
        )
        cutover_block = int(cutover["cutover_block"])
        observed_cutover_hash = normalize_raw_hash(
            archive_call('chain_getBlockHash', [cutover_block]),
            "cutover block hash",
        )
        if "0x" + genesis_hash != str(cutover["network_genesis_hash"]):
            raise ValidatorChainSourceV2Error("chain genesis differs from cutover")
        if "0x" + observed_cutover_hash != str(cutover["cutover_block_hash"]):
            raise ValidatorChainSourceV2Error("chain cutover block hash differs")
        if cutover_block <= 0:
            raise ValidatorChainSourceV2Error(
                "measured cutover block has no predecessor"
            )
        cutover_header = parse_finalized_header(
            archive_call('chain_getHeader', ['0x' + observed_cutover_hash])
        )
        if int(cutover_header["block"]) != cutover_block:
            raise ValidatorChainSourceV2Error(
                "chain cutover header block differs"
            )
        cutover_index_key = subnet_epoch_storage_key(
            storage_name="SubnetEpochIndex",
            netuid=int(netuid),
        )
        cutover_index = decode_subnet_epoch_storage(
            archive_call('state_getStorage', [cutover_index_key, '0x' + observed_cutover_hash]),
            storage_name="SubnetEpochIndex",
        )
        predecessor_hash = normalize_raw_hash(
            archive_call('chain_getBlockHash', [cutover_block - 1]),
            "cutover predecessor block hash",
        )
        predecessor_index = decode_subnet_epoch_storage(
            archive_call('state_getStorage', [cutover_index_key, '0x' + predecessor_hash]),
            storage_name="SubnetEpochIndex",
        )
        if (
            cutover_index != int(cutover["first_subnet_epoch_index"])
            or predecessor_index + 1 != cutover_index
        ):
            raise ValidatorChainSourceV2Error(
                "measured cutover is not an official epoch transition"
            )

        snapshot_call = archive_call if historical_snapshot else live_call
        decoded = {}
        for storage_name in (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        ):
            storage_value = snapshot_call(
                "state_getStorage",
                [
                    subnet_epoch_storage_key(
                        storage_name=storage_name,
                        netuid=int(netuid),
                    ),
                    "0x" + finalized_hash,
                ],
            )
            decoded[storage_name] = decode_subnet_epoch_storage(
                storage_value,
                storage_name=storage_name,
            )
        current_observed_at = _chain_timestamp(
            decode_timestamp_now_storage(
                snapshot_call(
                    "state_getStorage",
                    [timestamp_now_storage_key(), "0x" + finalized_hash],
                )
            )
        )

        current_block = int(header["block"])
        tempo = int(decoded["Tempo"])
        last_epoch_block = int(decoded["LastEpochBlock"])
        pending_epoch_at = int(decoded["PendingEpochAt"])
        subnet_epoch_index = int(decoded["SubnetEpochIndex"])
        blocks_since_last_step = int(decoded["BlocksSinceLastStep"])
        if tempo <= 0 or last_epoch_block > current_block:
            raise ValidatorChainSourceV2Error("finalized epoch schedule is invalid")
        first_index = int(cutover["first_subnet_epoch_index"])
        if subnet_epoch_index < first_index:
            raise ValidatorChainSourceV2Error(
                "finalized epoch state predates measured cutover"
            )
        mapped_epoch = int(cutover["first_settlement_epoch_id"]) + (
            subnet_epoch_index - first_index
        )
        if mapped_epoch != int(settlement_epoch_id):
            raise ValidatorChainSourceV2Error(
                "finalized subnet epoch differs from requested settlement epoch"
            )
        if current_block < cutover_block:
            raise ValidatorChainSourceV2Error("finalized block predates cutover")
        automatic_next = last_epoch_block + tempo
        if blocks_since_last_step > CHAIN_SUBTENSOR_MAX_TEMPO:
            next_epoch_block = current_block
        else:
            safety_next = current_block + (
                CHAIN_SUBTENSOR_MAX_TEMPO + 1 - blocks_since_last_step
            )
            next_epoch_block = (
                min(automatic_next, pending_epoch_at, safety_next)
                if pending_epoch_at > 0
                else min(automatic_next, safety_next)
            )
        epoch_ref = sha256_json(
            {
                "epoch_scheme": cutover["epoch_scheme"],
                "network_genesis_hash": cutover["network_genesis_hash"],
                "netuid": int(netuid),
                "subnet_epoch_index": subnet_epoch_index,
            }
        )

        def boundary_call(
            method: str,
            params: Sequence[Any],

        ) -> Any:
            nonlocal request_id
            result = self._archive_call(method=method, params=params, request_id=request_id)
            request_id += 1
            return result

        index_key = subnet_epoch_storage_key(
            storage_name="SubnetEpochIndex",
            netuid=int(netuid),
        )
        probed = {}

        def probe_index(
            block_number: int,
            known_hash: Optional[str] = None,
            *,
            live: bool = False,
        ) -> Any:
            normalized_block = int(block_number)
            cached = probed.get(normalized_block)
            if cached is not None:
                return cached
            block_hash = known_hash or normalize_raw_hash(
                boundary_call('chain_getBlockHash', [normalized_block]),
                "subnet epoch search block hash",
            )
            index_call = live_call if live else boundary_call
            observed_index = decode_subnet_epoch_storage(
            index_call(
                "state_getStorage",
                [index_key, "0x" + block_hash],
            ),
                storage_name="SubnetEpochIndex",
            )
            probed[normalized_block] = (block_hash, observed_index)
            return probed[normalized_block]

        _current_hash, current_index_for_search = probe_index(
            current_block,
            finalized_hash,
            live=not historical_snapshot,
        )
        if current_index_for_search != subnet_epoch_index:
            raise ValidatorChainSourceV2Error(
                "finalized subnet epoch changed during boundary search"
            )
        if subnet_epoch_index == first_index:
            boundary_block = cutover_block
            boundary_hash, boundary_index = probe_index(
                boundary_block,
                observed_cutover_hash,
            )
            if boundary_index != subnet_epoch_index:
                raise ValidatorChainSourceV2Error(
                    "cutover boundary subnet epoch index differs"
                )
        else:
            high = current_block
            step = 1
            low = None
            while low is None:
                candidate = max(cutover_block, current_block - step)
                _candidate_hash, candidate_index = probe_index(candidate)
                if candidate_index > subnet_epoch_index:
                    raise ValidatorChainSourceV2Error(
                        "historical subnet epoch index is not monotonic"
                    )
                if candidate_index < subnet_epoch_index:
                    low = candidate
                    break
                high = candidate
                if candidate == cutover_block:
                    raise ValidatorChainSourceV2Error(
                        "subnet epoch boundary does not follow cutover"
                    )
                step *= 2
            while high - low > 1:
                midpoint = low + (high - low) // 2
                _midpoint_hash, midpoint_index = probe_index(midpoint)
                if midpoint_index > subnet_epoch_index:
                    raise ValidatorChainSourceV2Error(
                        "historical subnet epoch index is not monotonic"
                    )
                if midpoint_index < subnet_epoch_index:
                    low = midpoint
                else:
                    high = midpoint
            boundary_block = high
            boundary_hash, boundary_index = probe_index(boundary_block)
            _predecessor_hash, predecessor_index = probe_index(boundary_block - 1)
            if (
                boundary_index != subnet_epoch_index
                or predecessor_index + 1 != subnet_epoch_index
            ):
                raise ValidatorChainSourceV2Error(
                    "subnet epoch boundary transition is invalid"
                )
        boundary_header = parse_finalized_header(
            boundary_call('chain_getHeader', ['0x' + boundary_hash])
        )
        if int(boundary_header["block"]) != boundary_block:
            raise ValidatorChainSourceV2Error(
                "subnet epoch boundary header block differs"
            )
        boundary_decoded = {}
        for storage_name in (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        ):
            boundary_decoded[storage_name] = decode_subnet_epoch_storage(
                boundary_call('state_getStorage', [subnet_epoch_storage_key(storage_name=storage_name, netuid=int(netuid)), '0x' + boundary_hash]),
                storage_name=storage_name,
            )
        boundary_observed_at = _chain_timestamp(
            decode_timestamp_now_storage(
                boundary_call('state_getStorage', [timestamp_now_storage_key(), '0x' + boundary_hash])
            )
        )
        boundary_tempo = int(boundary_decoded["Tempo"])
        boundary_last_epoch_block = int(boundary_decoded["LastEpochBlock"])
        boundary_pending_epoch_at = int(boundary_decoded["PendingEpochAt"])
        boundary_subnet_epoch_index = int(boundary_decoded["SubnetEpochIndex"])
        boundary_blocks_since_last_step = int(
            boundary_decoded["BlocksSinceLastStep"]
        )
        if (
            boundary_tempo <= 0
            or boundary_last_epoch_block != boundary_block
            or boundary_subnet_epoch_index != subnet_epoch_index
        ):
            raise ValidatorChainSourceV2Error(
                "historical subnet epoch boundary state differs"
            )
        boundary_next_epoch_block = boundary_last_epoch_block + boundary_tempo
        if boundary_blocks_since_last_step > CHAIN_SUBTENSOR_MAX_TEMPO:
            boundary_next_epoch_block = boundary_block
        else:
            boundary_safety_next = boundary_block + (
                CHAIN_SUBTENSOR_MAX_TEMPO
                + 1
                - boundary_blocks_since_last_step
            )
            boundary_next_epoch_block = (
                min(
                    boundary_next_epoch_block,
                    boundary_pending_epoch_at,
                    boundary_safety_next,
                )
                if boundary_pending_epoch_at > 0
                else min(boundary_next_epoch_block, boundary_safety_next)
            )
        boundary_snapshot = {
            "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
            "epoch_scheme": cutover["epoch_scheme"],
            "network_genesis_hash": cutover["network_genesis_hash"],
            "netuid": int(netuid),
            "head_kind": "finalized",
            "block_hash": "0x" + boundary_hash,
            "current_block": boundary_last_epoch_block,
            "last_epoch_block": boundary_last_epoch_block,
            "pending_epoch_at": boundary_pending_epoch_at,
            "subnet_epoch_index": boundary_subnet_epoch_index,
            "tempo": boundary_tempo,
            "blocks_since_last_step": boundary_blocks_since_last_step,
            "observed_at": boundary_observed_at,
            "epoch_id": boundary_subnet_epoch_index,
            "epoch_ref": epoch_ref,
            "epoch_block": 0,
            "next_epoch_block": boundary_next_epoch_block,
            "blocks_remaining": max(
                0,
                boundary_next_epoch_block - boundary_last_epoch_block,
            ),
            "settlement_epoch_id": mapped_epoch,
            "cutover_mapping_hash": cutover["mapping_hash"],
        }
        return {
            "authority": {
                "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
                "epoch_scheme": cutover["epoch_scheme"],
                "network_genesis_hash": cutover["network_genesis_hash"],
                "netuid": int(netuid),
                "head_kind": "finalized",
                "block_hash": "0x" + finalized_hash,
                "current_block": current_block,
                "last_epoch_block": last_epoch_block,
                "pending_epoch_at": pending_epoch_at,
                "subnet_epoch_index": subnet_epoch_index,
                "tempo": tempo,
                "blocks_since_last_step": blocks_since_last_step,
                "observed_at": current_observed_at,
                "epoch_id": subnet_epoch_index,
                "epoch_block": current_block - last_epoch_block,
                "next_epoch_block": next_epoch_block,
                "blocks_remaining": max(0, next_epoch_block - current_block),
                "epoch_ref": epoch_ref,
                "settlement_epoch_id": mapped_epoch,
                "cutover_mapping_hash": cutover["mapping_hash"],
            },
            "boundary_snapshot": boundary_snapshot,
            "next_request_id": request_id,
        }


    def read_finalized_snapshot(self, *, netuid: int, epoch_id: int) -> Dict[str, Any]:
        finalized = self._call(method='chain_getFinalizedHead', params=[], request_id=1)
        finalized_hash = normalize_raw_hash(finalized, "finalized head")

        header_result = self._call(method='chain_getHeader', params=['0x' + finalized_hash], request_id=2)
        header = parse_finalized_header(header_result)
        epoch_configuration = self._epoch_authority_supplier()
        epoch_authority = None
        epoch_boundary = None
        next_request_id = 3
        historical_snapshot = False
        metagraph_call = self._call
        if epoch_configuration is None:
            raise ValidatorChainSourceV2Error(
                "official SN71 epoch authority is unavailable"
            )

        # Weight publication starts near the end of an epoch. The finalized
        # head can move into the next official epoch while gateway evidence is
        # being assembled. Select the just-finished epoch's final block from
        # authenticated state instead of binding the old settlement to a new
        # epoch head.
        cutover = epoch_configuration.get("cutover_manifest")
        if (
            isinstance(epoch_configuration, Mapping)
            and epoch_configuration.get("mode") == "stateful_v1"
            and isinstance(cutover, Mapping)
        ):
            cutover_body_fields = (
                "schema_version",
                "epoch_scheme",
                "network_genesis_hash",
                "netuid",
                "cutover_block",
                "cutover_block_hash",
                "first_subnet_epoch_index",
                "first_settlement_epoch_id",
                "last_legacy_epoch_id",
            )
            try:
                cutover_body = {
                    field: cutover[field] for field in cutover_body_fields
                }
                mapping_valid = (
                    set(cutover) == set(cutover_body_fields) | {"mapping_hash"}
                    and cutover.get("schema_version")
                    == "leadpoet.subnet_epoch_cutover.v1"
                    and cutover.get("epoch_scheme")
                    == "bittensor.subnet_epoch_index.v1"
                    and int(cutover["netuid"]) == int(netuid)
                    and str(cutover.get("mapping_hash") or "").lower()
                    == sha256_json(cutover_body)
                )
            except (KeyError, TypeError, ValueError):
                mapping_valid = False
            if mapping_valid:
                if self._archive_rpc_call is None:
                    raise ValidatorChainSourceV2Error(
                        "archive RPC adapter is required for historical epoch state"
                    )
                def selector_call(
                    storage_name: str,
                    request_id: int,
                ) -> int:
                    result = self._call(method='state_getStorage', params=[subnet_epoch_storage_key(storage_name=storage_name, netuid=int(netuid)), '0x' + finalized_hash], request_id=request_id)
                    return decode_subnet_epoch_storage(
                        result, storage_name=storage_name
                    )

                finalized_index = selector_call("SubnetEpochIndex", 3)
                finalized_last_epoch_block = selector_call("LastEpochBlock", 4)
                finalized_settlement_epoch = int(
                    cutover["first_settlement_epoch_id"]
                ) + (
                    finalized_index - int(cutover["first_subnet_epoch_index"])
                )
                next_request_id = 5
                if finalized_settlement_epoch == int(epoch_id) + 1:
                    target_block = int(finalized_last_epoch_block) - 1
                    if target_block < int(cutover["cutover_block"]):
                        raise ValidatorChainSourceV2Error(
                            "requested settlement epoch has no finalized predecessor"
                        )
                    target_hash_result = self._archive_call(method='chain_getBlockHash', params=[target_block], request_id=5)
                    finalized_hash = normalize_raw_hash(
                        target_hash_result,
                        "settlement predecessor block hash",
                    )
                    target_header_result = self._archive_call(method='chain_getHeader', params=['0x' + finalized_hash], request_id=6)
                    header = parse_finalized_header(target_header_result)
                    if int(header["block"]) != target_block:
                        raise ValidatorChainSourceV2Error(
                            "settlement predecessor header differs"
                        )
                    historical_snapshot = True
                    metagraph_call = self._archive_call
                    next_request_id = 7
                elif finalized_settlement_epoch != int(epoch_id):
                    raise ValidatorChainSourceV2Error(
                        "requested settlement epoch is not current or just finalized"
                    )
        stateful = self._read_stateful_epoch_authority(configuration=epoch_configuration, finalized_hash=finalized_hash, header=header, netuid=int(netuid), settlement_epoch_id=int(epoch_id), request_id_start=next_request_id, historical_snapshot=historical_snapshot)
        epoch_authority = stateful["authority"]
        epoch_boundary = stateful["boundary_snapshot"]
        next_request_id = int(stateful["next_request_id"])

        metagraph_result = metagraph_call(method='state_call', params=[CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=int(netuid)), '0x' + finalized_hash], request_id=next_request_id)
        metagraph = decode_selective_metagraph_result(metagraph_result)
        if int(metagraph["netuid"]) != int(netuid):
            raise ValidatorChainSourceV2Error("chain metagraph netuid differs")
        if int(metagraph["block"]) != int(header["block"]):
            raise ValidatorChainSourceV2Error("metagraph and finalized block differ")
        return {
            "finalized_block_hash": finalized_hash,
            "header": header,
            "metagraph": metagraph,
            "epoch_authority": epoch_authority,
            "epoch_boundary": epoch_boundary,
        }

    def find_finalized_extrinsic_inclusion(
        self,
        *,
        expected_extrinsics: Mapping[str, str],
        expected_commitments: Mapping[str, Mapping[str, Any]],
        minimum_block: int,
        maximum_block: int,
    ) -> Dict[str, Any]:
        """Find one exact signer-built extrinsic in authenticated finalized blocks."""

        if not isinstance(expected_extrinsics, Mapping) or not expected_extrinsics:
            raise ValidatorChainSourceV2Error("expected extrinsic set is empty")
        if (
            not isinstance(expected_commitments, Mapping)
            or set(expected_commitments) != set(expected_extrinsics)
        ):
            raise ValidatorChainSourceV2Error(
                "expected commitment set differs from extrinsics"
            )
        normalized = {}
        for expected_hash, encoded in expected_extrinsics.items():
            hash_text = str(expected_hash or "").lower()
            encoded_text = str(encoded or "").lower()
            if hash_text.startswith("0x") and len(hash_text) == 66:
                pass
            else:
                raise ValidatorChainSourceV2Error("expected extrinsic hash is invalid")
            try:
                raw = bytes.fromhex(encoded_text)
            except ValueError as exc:
                raise ValidatorChainSourceV2Error(
                    "expected extrinsic bytes are invalid"
                ) from exc
            if signed_extrinsic_hash_v2(raw) != hash_text:
                raise ValidatorChainSourceV2Error(
                    "expected extrinsic hash differs from bytes"
                )
            normalized[hash_text] = raw.hex()

        start = int(minimum_block)
        requested_end = int(maximum_block)
        if start < 0 or requested_end < start:
            raise ValidatorChainSourceV2Error("finalization scan range is invalid")
        rpc_started = False

        def finalization_call(
            *,
            adapter: Optional[Callable[..., Dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            nonlocal rpc_started
            if rpc_started:
                self._finalization_sleep(FINALIZATION_RPC_PACING_SECONDS)
            result = (adapter or self._call)(**kwargs)
            rpc_started = True
            return result

        finalized = finalization_call(method='chain_getFinalizedHead', params=[], request_id=1)
        finalized_hash = normalize_raw_hash(finalized, "finalized head")
        header_result = finalization_call(method='chain_getHeader', params=['0x' + finalized_hash], request_id=2)
        finalized_header = parse_finalized_header(header_result)
        end = min(requested_end, int(finalized_header["block"]))
        if end < start:
            raise ValidatorChainSourceV2Error(
                "authorized extrinsic range is not finalized"
            )
        if end - start + 1 > CHAIN_MAX_FINALIZATION_SCAN_BLOCKS:
            raise ValidatorChainSourceV2Error(
                "finalization scan exceeds measured policy"
            )

        request_id = 3
        # Weight submissions normally land near the end of their mortal era.
        # Scan newest-first so the proof completes before the epoch rolls.
        for block_number in range(end, start - 1, -1):
            block_hash_result = finalization_call(method='chain_getBlockHash', params=[block_number], request_id=request_id)
            request_id += 1
            block_hash = normalize_raw_hash(
                block_hash_result, "finalized block hash"
            )
            block_result = finalization_call(method='chain_getBlock', params=['0x' + block_hash], request_id=request_id)
            request_id += 1
            block = parse_finalized_block_extrinsics(
                block_result, expected_block=block_number
            )
            candidate_extrinsics = block["extrinsics"]
            has_expected_extrinsic = any(
                normalized.get(
                    signed_extrinsic_hash_v2(bytes.fromhex(extrinsic_hex))
                )
                == extrinsic_hex
                for extrinsic_hex in candidate_extrinsics
            )
            if not has_expected_extrinsic and self._archive_rpc_call is not None:
                # The live endpoint anchors finality and the canonical block
                # hash. If its load-balanced block-body read is incomplete,
                # independently read that exact immutable hash from the
                # measured archive. Require identical headers before accepting
                # any archive bytes, then retain the existing exact-extrinsic
                # and state-transition checks below.
                archive_block_result = finalization_call(adapter=self._archive_call, method='chain_getBlock', params=['0x' + block_hash], request_id=request_id)
                request_id += 1
                archive_block = parse_finalized_block_extrinsics(
                    archive_block_result, expected_block=block_number
                )
                if archive_block["header"] != block["header"]:
                    raise ValidatorChainSourceV2Error(
                        "live and archive finalized block headers differ"
                    )
                candidate_extrinsics = archive_block["extrinsics"]
            for extrinsic_hex in candidate_extrinsics:
                extrinsic_hash = signed_extrinsic_hash_v2(
                    bytes.fromhex(extrinsic_hex)
                )
                if normalized.get(extrinsic_hash) == extrinsic_hex:
                    expected_commitment = expected_commitments.get(extrinsic_hash)
                    if not isinstance(expected_commitment, Mapping) or set(
                        expected_commitment
                    ) != {
                        "netuid",
                        "subnet_epoch_index",
                        "hotkey_public_key",
                        "commitment_hex",
                        "reveal_round",
                    }:
                        raise ValidatorChainSourceV2Error(
                            "expected commitment fields are invalid"
                        )
                    # Live nodes may discard exact historical state immediately
                    # after finalization. Keep the live node's canonical block
                    # hash and exact extrinsic above, but obtain the state
                    # transition at that exact hash from the measured archive.
                    state_call = (
                        self._archive_call
                        if self._archive_rpc_call is not None
                        else self._call
                    )
                    storage_result = finalization_call(adapter=state_call, method='state_getStorage', params=[timelocked_weight_commits_storage_key(netuid=int(expected_commitment['netuid']), subnet_epoch_index=int(expected_commitment['subnet_epoch_index'])), '0x' + block_hash], request_id=request_id)
                    commits = decode_timelocked_weight_commits(
                        storage_result
                    )
                    matched_commits = [
                        item
                        for item in commits
                        if item["hotkey_public_key"]
                        == str(expected_commitment["hotkey_public_key"])
                        and item["commitment_hex"]
                        == str(expected_commitment["commitment_hex"])
                        and int(item["reveal_round"])
                        == int(expected_commitment["reveal_round"])
                        and int(item["submitted_at"]) <= block_number
                    ]
                    if len(matched_commits) != 1:
                        raise ValidatorChainSourceV2Error(
                            "finalized extrinsic did not produce the expected chain state"
                        )
                    return {
                        "extrinsic_hash": extrinsic_hash,
                        "extrinsic_hex": extrinsic_hex,
                        "finalized_block": block_number,
                        "finalized_block_hash": block_hash,
                        "finalized_head": finalized_header,
                        "state_transition_hash": sha256_json(matched_commits[0]),
                    }
        raise ValidatorChainSourceV2Error(
            "no authorized weight extrinsic was found in finalized blocks"
        )

    def _call(self, **kwargs: Any) -> Any:
        return self._rpc_call(**kwargs)

    def _archive_call(self, **kwargs: Any) -> Any:
        if self._archive_rpc_call is None:
            raise ValidatorChainSourceV2Error(
                "archive RPC adapter is required for historical epoch state"
            )
        return self._archive_rpc_call(**kwargs)
